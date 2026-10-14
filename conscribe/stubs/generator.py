"""Render ``.pyi`` source for one module from collected stub information.

A ``.pyi`` file *replaces* its module for type checkers: every name a stub
uses must be bound in the stub itself.  So the renderer never copies
annotation text blindly and never binds a name to ``Any`` because it could
not import it:

* Annotations are resolved in the namespace of the class or function that
  declares them (an inherited dataclass field in its own module, a method in
  its function's globals; see :mod:`conscribe.config._annotations`) and
  rendered from the resolved objects: ``Annotated`` metadata is dropped
  (type checkers ignore it in a stub), ``NoneType`` is written ``None``,
  unions use ``|``, enum members in ``Literal`` keep their dotted form.
* Every referenced object is written by its short name with a verified
  import.  The declaring module's own import statement is reproduced when it
  binds the very same object (``from pydantic import BaseModel`` stays that,
  aliases survive); otherwise the object's ``__module__`` is used.  Names
  that are only imported under ``if TYPE_CHECKING:`` are imported the same
  way in the stub -- read from the source, never imported.
* Objects defined in the stubbed module itself are defined in the stub:
  nominal classes as ``class X(Any): ...`` placeholders, enums with their
  members, Protocols and TypedDicts structurally (their own declared
  members), type variables, ``NewType`` and type aliases faithfully.

Names are assigned once everything is rendered (same-module definitions keep
their real names; an import takes the declaring module's spelling and gets a
numeric suffix only on a clash).  Then the source is checked: it must parse,
bind every top-level name once, bind every name it uses, and import only
names that exist.  Anything conscribe cannot render faithfully raises
:class:`~conscribe.exceptions.ConfigRenderError`; an annotation naming an
undefined name raises :class:`~conscribe.exceptions.ConfigAnnotationError`.
An invalid stub is never returned.
"""
from __future__ import annotations

import abc
import ast
import builtins
import collections.abc
import contextlib
import dataclasses
import enum
import functools
import importlib
import inspect
import keyword
import re
import sys
import types
import typing
import warnings
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, ForwardRef, get_args, get_origin

from conscribe.config import _annotations as _ann
from conscribe.config._utils import dataclass_field_owner, is_generated_init, qualified_name
from conscribe.exceptions import ConfigRenderError, ExtractionWarning
from conscribe.stubs.collector import (
    ClassAttrStub,
    ClassStubInfo,
    ClassValueStub,
    InjectedAttr,
    MethodStub,
    collect_class_members,
)

try:  # a pydantic dependency; tolerate its absence anyway
    import typing_extensions as _te
except ImportError:  # pragma: no cover
    _te = None  # type: ignore[assignment]

NoneType = type(None)


# ── typing objects, by identity (typing and typing_extensions) ───


def _forms(*names: str) -> tuple[Any, ...]:
    found: list[Any] = []
    for module in (typing, _te):
        if module is None:
            continue
        for name in names:
            obj = getattr(module, name, None)
            if obj is not None and not any(obj is f for f in found):
                found.append(obj)
    return tuple(found)


def _types(*names: str) -> tuple[type, ...]:
    return tuple(t for t in _forms(*names) if isinstance(t, type))


def _is_one_of(obj: Any, forms: tuple[Any, ...]) -> bool:
    return any(obj is f for f in forms)


_ANNOTATED = _forms("Annotated")
_LITERAL = _forms("Literal")
_FINAL = _forms("Final")
_TD_WRAPPERS = _forms("Required", "NotRequired", "ReadOnly")
_TYPEVAR_TYPES = _types("TypeVar")
_PARAMSPEC_TYPES = _types("ParamSpec")
_TYPEVARTUPLE_TYPES = _types("TypeVarTuple")
_TYPEALIAS_TYPES = _types("TypeAliasType")
_NEWTYPE_TYPES = _types("NewType")
_PARAMSPEC_ARGS = _types("ParamSpecArgs")
_PARAMSPEC_KWARGS = _types("ParamSpecKwargs")
_SPECIAL_ALIAS: tuple[type, ...] = tuple(
    t for t in (getattr(typing, "_SpecialGenericAlias", None),) if isinstance(t, type)
)

_PY_BUILTINS: frozenset[str] = frozenset(dir(builtins))
_TOKEN_RE = re.compile(r"__cs_ref_(\d+)__")
_MISSING: Any = object()

#: Modules that exist only for type checkers (their imports are not verified).
_STUB_ONLY_MODULES = frozenset({"_typeshed"})

#: Values a bare ``Final`` keeps as a literal (``LIMIT: Final = 3``).
_FINAL_LITERAL_TYPES: tuple[type, ...] = (bool, int, float, complex, str, bytes, NoneType)

#: Origins of the generator annotations ``@contextmanager`` methods use.
_ITERATOR_ORIGINS: tuple[Any, ...] = (
    collections.abc.Iterator,
    collections.abc.Generator,
    collections.abc.Iterable,
    collections.abc.AsyncIterator,
    collections.abc.AsyncGenerator,
    collections.abc.AsyncIterable,
)
_ITERATOR_NAMES = frozenset(t.__name__ for t in _ITERATOR_ORIGINS)


# ── Public entry point ───────────────────────────────────────────


def generate_module_stub(
    module_name: str,
    classes: list[ClassStubInfo],
    *,
    partial_class_fallback: bool = False,
) -> str:
    """Generate the ``.pyi`` source for all *classes* of one module.

    The output keeps a module-level ``def __getattr__(name: str) -> Any``
    fallback (partial-stub pattern): names the stub does not declare resolve
    to ``Any`` for importers instead of disappearing.

    Classes are placed where the module defines them: a class nested in
    another class body (``Outer.Inner``) is declared inside ``Outer`` (as a
    nominal placeholder when ``Outer`` itself is not stubbed), in module
    order regardless of the order of *classes*.

    Args:
        module_name: The module the stub describes.
        classes: Stub info collected for that module's classes.  A class
            listed twice (e.g. registered under several keys) is rendered
            once.
        partial_class_fallback: When ``True``, each generated class gets a
            ``def __getattr__(self, name: str) -> Incomplete: ...`` so type
            checkers tolerate dynamic instance attributes.  Defaults to
            ``False`` for strict checking.

    Returns:
        The stub source, or ``""`` when *classes* is empty.

    Raises:
        ConfigRenderError: Something cannot be written into a valid stub (an
            object that can neither be imported nor defined, an unsupported
            annotation form, two different objects needing one name, a class
            defined inside a function -- not an attribute of its module);
            the message names the class and the member.
        ConfigAnnotationError: An annotation names something that exists
            nowhere (a typo or a missing import).
    """
    unique: list[ClassStubInfo] = []
    seen: set[int] = set()
    for info in classes:
        if id(info.cls) not in seen:
            seen.add(id(info.cls))
            unique.append(info)
    if not unique:
        return ""
    renderer = _ModuleRenderer(module_name, partial_class_fallback=partial_class_fallback)
    for info in sorted(unique, key=_module_order(module_name)):
        renderer.add_class(info)
    return renderer.render()


def _module_order(module_name: str) -> Any:
    """Sort key: module-level definition order; an enclosing class before its nested ones."""
    module = sys.modules.get(module_name)
    position: dict[str, int] = {}
    if module is not None:
        for index, name in enumerate(vars(module)):
            position.setdefault(name, index)

    def key(info: ClassStubInfo) -> tuple[int, int, str]:
        qualname = str(getattr(info.cls, "__qualname__", info.class_name))
        parts = qualname.split(".")
        return position.get(parts[0], len(position)), len(parts), qualname

    return key


# ── Bindings and same-module definitions ────────────────────────


@dataclass
class _Binding:
    """One top-level name of the stub: an import, a definition or a builtin."""

    index: int
    kind: str  # "from" | "import" | "home" | "builtin" | "qualified" (``builtins.X``)
    preferred: str
    module: str | None = None
    attr: str | None = None
    target: Any = None
    verified: bool = True
    name: str | None = None  # assigned by _assign_names()


@dataclass
class _Def:
    """Something the stub defines: a stubbed class or a same-module object."""

    name: str
    obj: Any
    kind: str
    binding: _Binding | None = None  # top-level definitions only
    info: ClassStubInfo | None = None  # kind == "stubbed"
    header: list[str] = field(default_factory=list)  # decorators + ``class`` line, or the assignment
    body: list[str] | None = None  # class body (``None``: assignment-style definition)
    children: dict[str, _Def] = field(default_factory=dict)
    deps: set[int] = field(default_factory=set)  # same-module bindings its header needs
    anonymous: bool = False  # a PEP 695 type parameter (no module-level name of its own)
    parent: _Def | None = None  # the enclosing class definition (nested definitions only)

    @property
    def name_text(self) -> str:
        return f"__cs_ref_{self.binding.index}__" if self.binding is not None else self.name


@dataclass(frozen=True)
class _ImportRecord:
    """``from module import attr`` (``attr`` set) or ``import module``."""

    module: str
    attr: str | None


class _Undefined(Exception):
    """A name in an annotation's source text is bound nowhere."""


class _NeedsLocals(Exception):
    """An annotation's source text needs a function-local name."""


# ── The renderer ─────────────────────────────────────────────────


class _ModuleRenderer:
    """Symbol table, renderers and the validity gate for one module stub."""

    def __init__(self, module_name: str, *, partial_class_fallback: bool) -> None:
        self.home = module_name
        self.home_module = sys.modules.get(module_name)
        self.home_globals: Mapping[str, Any] = (
            vars(self.home_module) if self.home_module is not None else {}
        )
        self.partial_class_fallback = partial_class_fallback
        self.bindings: list[_Binding] = []
        self._by_key: dict[tuple[Any, ...], _Binding] = {}
        self.defs: dict[str, _Def] = {}
        self._stubbed: dict[int, _Def] = {}
        self._stubbed_defs: list[_Def] = []  # every stubbed class (nested ones too), in add order
        self._pending: list[_Def] = []
        self._captures: list[set[int]] = []
        self._reverse: dict[int, dict[int, str]] = {}
        self._imports: dict[int, dict[str, _ImportRecord]] = {}
        self._keepalive: list[Any] = []
        # Names a binding must not take: a class body that uses the binding
        # binds the same name itself (``type = "x"`` and ``-> type[int]``).
        self._avoid: dict[int, set[str]] = {}

    # -- driver ---------------------------------------------------------------

    def add_class(self, info: ClassStubInfo) -> None:
        """Declare a stubbed class where its module defines it.

        A module-level class becomes a top-level definition; a class nested
        in a class body (``Outer.Inner``) a definition inside ``Outer``.  A
        class defined inside a function is not an attribute of its module:
        a stub cannot declare it.
        """
        cls = info.cls
        qualname = getattr(cls, "__qualname__", None)
        if not isinstance(qualname, str) or not qualname:
            qualname = info.class_name
        if "<locals>" in qualname:
            raise ConfigRenderError(
                qualified_name(cls), "<stub>", _describe(cls),
                f"it is defined inside a function, so it is not an attribute of module "
                f"{self.home} and a stub cannot declare it; define the class at module level "
                f"to get a stub for it",
            )
        parts = qualname.split(".")
        if not all(part.isidentifier() for part in parts):
            raise ConfigRenderError(
                qualified_name(cls), "<stub>", _describe(cls),
                f"its qualified name {qualname!r} is not a dotted name a stub can declare",
            )
        if len(parts) == 1:
            self._add_top_level(parts[0], info)
            return
        module = self.home_module
        chain = _walk_chain(module, parts) if module is not None else None
        if chain is None or chain[-1] is not cls:
            raise ConfigRenderError(
                qualified_name(cls), "<stub>", _describe(cls),
                f"it is nested in a class body, but {self.home}.{qualname} does not lead to "
                f"it, so the stub cannot place it; keep the enclosing classes bound under "
                f"their own names in module {self.home}",
            )
        where = qualified_name(cls)
        parent = self._home_def(parts[0], chain[0], where)
        for name, obj in zip(parts[1:-1], chain[1:-1]):
            parent = self._child_def(parent, name, obj, where)
        self._check_nestable(parent, cls, where)
        name = parts[-1]
        existing = parent.children.get(name)
        if existing is not None and existing.obj is not cls:  # pragma: no cover - chain checked
            raise ConfigRenderError(
                where, "<stub>", _describe(cls),
                f"{parent.name}.{name} would be defined twice in the stub of module {self.home}",
            )
        d = existing or _Def(name=name, obj=cls, kind="stubbed", parent=parent)
        if existing is not None and existing in self._pending:
            self._pending.remove(existing)  # was going to be a nominal placeholder
        d.kind, d.info = "stubbed", info
        parent.children[name] = d
        self._stubbed[id(cls)] = d
        self._stubbed_defs.append(d)

    def _add_top_level(self, name: str, info: ClassStubInfo) -> None:
        existing = self.defs.get(name)
        if existing is not None and existing.obj is not info.cls:
            raise ConfigRenderError(
                self.home, name, f"class {name!r}",
                f"two different classes named {name!r} would both be defined in the stub "
                f"of module {self.home} ({_describe(existing.obj)} and {_describe(info.cls)}); "
                f"give them distinct names",
            )
        if existing is not None:  # referenced earlier as a nominal placeholder
            if existing in self._pending:
                self._pending.remove(existing)
            d = existing
        else:
            d = _Def(name=name, obj=info.cls, kind="stubbed")
            d.binding = self._binding(("home", name), "home", name, target=info.cls)
            self.defs[name] = d
        d.kind, d.info = "stubbed", info
        self._stubbed[id(info.cls)] = d
        self._stubbed_defs.append(d)

    def _check_nestable(self, parent: _Def, obj: Any, where: str) -> None:
        if parent.kind not in ("stubbed", "placeholder", "enum", "protocol"):
            raise ConfigRenderError(
                where, "<reference>", _describe(obj),
                f"it is nested in {parent.name!r}, which a stub cannot nest definitions in; "
                f"move it to module level",
            )

    def render(self) -> str:
        any_ref = self.ref(typing.Any)
        for d in list(self._stubbed_defs):
            self._render_stubbed(d)
        while self._pending:
            self._render_def(self._pending.pop(0))
        for d in list(self.defs.values()):
            self._collect_shadowing(d)
        self._assign_names()

        lines = [
            f"# Generated by conscribe {_version()}",
            f"# DO NOT EDIT. Stub for module: {self.home}",
            "from __future__ import annotations",
            "",
            *self._import_lines(),
            "",
            f"def __getattr__(name: str) -> {any_ref}: ...",
            "",
        ]
        for d in self._ordered_defs():
            lines.extend(self._assemble(d, ""))
            lines.append("")
        source = self._substitute("\n".join(lines).rstrip() + "\n")
        self._check(source)
        return source

    # -- bindings -----------------------------------------------------------

    def _binding(
        self,
        key: tuple[Any, ...],
        kind: str,
        preferred: str,
        *,
        module: str | None = None,
        attr: str | None = None,
        target: Any = None,
        verified: bool = True,
    ) -> _Binding:
        existing = self._by_key.get(key)
        if existing is not None:
            existing.verified = existing.verified or verified
            return existing
        binding = _Binding(
            index=len(self.bindings), kind=kind, preferred=preferred,
            module=module, attr=attr, target=target, verified=verified,
        )
        self.bindings.append(binding)
        self._by_key[key] = binding
        self._keepalive.append(target)
        return binding

    def _token(self, binding: _Binding) -> str:
        if binding.kind == "home":
            for capture in self._captures:
                capture.add(binding.index)
        return f"__cs_ref_{binding.index}__"

    def _collect_shadowing(self, d: _Def) -> None:
        """Record which bindings *d*'s class body uses while binding the same name.

        Annotations, defaults and decorators in a class body are evaluated
        in the class scope, so a class that declares ``type`` (or a method
        ``dict``) cannot write ``type[int]`` / ``dict[str, int]`` there and
        mean the builtin.  Such a binding later gets a name the body does not
        bind (``builtins.type``, an aliased import).  A same-module
        definition cannot be renamed: that stub is refused.
        """
        scope = _body_scope(d)
        if d.body is not None and scope:
            # The body's own lines, and the headers of nested classes (their
            # bases are evaluated in this class scope too).
            lines = [*d.body, *(line for child in d.children.values() for line in child.header)]
            for index in sorted({int(m) for line in lines for m in _TOKEN_RE.findall(line)}):
                binding = self.bindings[index]
                if binding.kind == "home":
                    if binding.preferred in scope:
                        raise ConfigRenderError(
                            self.home, "<stub>", repr(binding.preferred),
                            f"class {d.name!r} binds {binding.preferred!r} in its body and "
                            f"also refers to the module-level {binding.preferred!r} there; "
                            f"in a stub the class attribute would shadow it. Rename one of them",
                        )
                elif binding.kind == "builtin":
                    if binding.preferred in scope:
                        # Spelled ``builtins.<name>`` in this body only.
                        qualified = self._binding(
                            ("qualified", binding.preferred), "qualified", binding.preferred,
                            target=binding.target,
                        )
                        self._binding(
                            ("import", "builtins"), "import", "builtins",
                            module="builtins", target=builtins,
                        )
                        old, new = self._token(binding), self._token(qualified)
                        d.body = [line.replace(old, new) for line in d.body]
                        for child in d.children.values():
                            child.header = [line.replace(old, new) for line in child.header]
                else:
                    self._avoid.setdefault(index, set()).update(scope)
        for child in d.children.values():
            self._collect_shadowing(child)

    def _assign_names(self) -> None:
        taken: dict[str, _Binding] = {}
        for binding in self.bindings:
            if binding.kind == "home":
                binding.name = binding.preferred
                taken[binding.name] = binding
        shadowed = [b for b in self.bindings if b.kind == "builtin" and b.preferred in taken]
        if shadowed:  # a same-module definition shadows a builtin the stub uses
            self._binding(("import", "builtins"), "import", "builtins", module="builtins", target=builtins)
        builtin_names = {b.preferred for b in self.bindings if b.kind == "builtin"}
        imports = sorted(
            (b for b in self.bindings if b.kind in ("from", "import")),
            key=lambda b: (b.preferred, b.module or "", b.attr or ""),
        )

        def clashes(candidate: str, binding: _Binding) -> bool:
            return (
                candidate in taken
                or keyword.iskeyword(candidate)
                or (candidate in builtin_names and binding.target is not builtins)
                or (candidate in _PY_BUILTINS and binding.target is not builtins)
                or candidate in self._avoid.get(binding.index, ())
                or self._shadows_home_global(candidate, binding)
            )

        for binding in imports:
            if _is_plain_dotted(binding):
                # ``import a.b.c`` binds ``a``; references keep the dotted
                # spelling.  Several such imports share ``a``.
                top = binding.preferred.split(".", 1)[0]
                owner = taken.get(top)
                if (owner is None and not clashes(top, binding)) or (
                    owner is not None
                    and _binds_package(owner, top)
                    and top not in self._avoid.get(binding.index, ())
                ):
                    binding.name = binding.preferred
                    taken.setdefault(top, binding)
                    continue
                base = binding.preferred.replace(".", "_")  # alias the module itself
            else:
                base = binding.preferred
            suffix, candidate = 1, base
            while clashes(candidate, binding):
                suffix += 1
                candidate = f"{base}_{suffix}"
            binding.name = candidate
            taken[candidate] = binding
        for binding in self.bindings:
            if binding.kind == "builtin":
                if binding.preferred in taken and taken[binding.preferred] is not binding:
                    binding.name = f"{self._by_key[('import', 'builtins')].name}.{binding.preferred}"
                else:
                    binding.name = binding.preferred
            elif binding.kind == "qualified":  # a builtin shadowed by a class body
                binding.name = f"{self._by_key[('import', 'builtins')].name}.{binding.preferred}"

    def _shadows_home_global(self, name: str, binding: _Binding) -> bool:
        """True when *name* is a real module global bound to something else."""
        if name not in self.home_globals:
            return False
        value = self.home_globals[name]
        if binding.kind == "import":
            # ``import a.b.c`` (unaliased) binds the top package ``a``.
            module = binding.module or ""
            bound = module.split(".", 1)[0] if name == module.split(".", 1)[0] else module
            return not (isinstance(value, types.ModuleType) and value.__name__ == bound)
        return value is not binding.target

    def _substitute(self, text: str) -> str:
        return _TOKEN_RE.sub(lambda m: str(self.bindings[int(m.group(1))].name), text)

    def _import_lines(self) -> list[str]:
        plain: set[str] = set()
        grouped: dict[str, set[str]] = {}
        for binding in self.bindings:
            if binding.kind == "import":
                if binding.name == binding.module:
                    plain.add(f"import {binding.module}")
                else:
                    plain.add(f"import {binding.module} as {binding.name}")
            elif binding.kind == "from":
                piece = binding.attr if binding.name == binding.attr else f"{binding.attr} as {binding.name}"
                grouped.setdefault(binding.module or "", set()).add(str(piece))
        lines = sorted(plain)
        for module in sorted(grouped):
            lines.append(f"from {module} import {', '.join(sorted(grouped[module]))}")
        return lines

    # -- references -----------------------------------------------------------

    def ref(
        self, obj: Any, ctx: _ann.AnnotationContext | None = None, where: str = ""
    ) -> str:
        """Expression text (with name tokens) that refers to *obj* from the stub."""
        if obj is None or obj is NoneType:
            return "None"
        stubbed = self._stubbed.get(id(obj))
        if stubbed is not None and stubbed.obj is obj:
            return self._def_ref(stubbed)
        if _is_builtin(obj):
            return self._token(
                self._binding(("builtin", obj.__name__), "builtin", obj.__name__, target=obj)
            )
        return self._locate(obj, ctx or self._home_ctx(), where)

    def _locate(self, obj: Any, ctx: _ann.AnnotationContext, where: str) -> str:
        module_name = getattr(obj, "__module__", None)
        qual = _qualname(obj)
        if isinstance(module_name, str) and qual is not None:
            parts = qual.split(".")
            module = self._module(module_name)
            chain = _walk_chain(module, parts) if module is not None else None
            if chain is not None and chain[-1] is obj:
                if module_name == self.home:
                    return self._home_ref(parts, chain, where)
                return self._import_ref(obj, module_name, parts, chain, ctx)

        # Not reachable by its canonical name (bridges, a rewritten
        # ``__module__``): a module-level binding of the very same object.
        for globalns, scope_module in self._scopes(ctx):
            name = self._binding_name(globalns, obj)
            if name is None:
                continue
            record = self._source_imports(globalns).get(name)
            if record is not None and record.attr is not None and record.module != self.home:
                source_module = self._module(record.module)
                if source_module is not None and getattr(source_module, record.attr, _MISSING) is obj:
                    return self._token(self._from_binding(record.module, record.attr, name, obj))
            if scope_module == self.home:
                return self._home_ref([name], [obj], where)
            if isinstance(scope_module, str):
                return self._token(self._from_binding(scope_module, name, name, obj))

        if module_name == self.home and _definable(obj):
            # A function-local (or otherwise unreachable) object of this module.
            name = getattr(obj, "__name__", None)
            if isinstance(name, str) and name.isidentifier():
                return self._home_ref([name], [obj], where)
        if isinstance(obj, _TYPEVAR_TYPES + _PARAMSPEC_TYPES + _TYPEVARTUPLE_TYPES):
            # A PEP 695 type parameter (``class Box[T]``, ``def f[T]``,
            # ``type A[T] = ...``): bound to no module name, so the stub
            # defines its own module-level type variable for it.
            return self._anonymous_type_param(obj, where)
        raise ConfigRenderError(
            where or self.home, "<reference>", _describe(obj),
            f"it cannot be imported by name (its __module__ is {module_name!r}) and no "
            f"module-level name in {self.home} or in the module that uses it is bound "
            f"to it; define it at module level so a stub can refer to it",
        )

    def _import_ref(
        self,
        obj: Any,
        module_name: str,
        parts: list[str],
        chain: list[Any],
        ctx: _ann.AnnotationContext,
    ) -> str:
        if len(parts) == 1:
            local = self._binding_name(ctx.globalns, obj)
            if local is not None:
                # Alias-faithful: reuse the declaring module's own import.
                record = self._source_imports(ctx.globalns).get(local)
                if record is not None and record.attr is not None:
                    module = self._module(record.module)
                    if module is not None and getattr(module, record.attr, _MISSING) is obj:
                        return self._token(self._from_binding(record.module, record.attr, local, obj))
            return self._token(self._from_binding(module_name, parts[0], local or parts[0], obj))
        binding = self._from_binding(module_name, parts[0], parts[0], chain[0])
        return self._token(binding) + "".join(f".{p}" for p in parts[1:])

    def _anonymous_type_param(self, obj: Any, where: str) -> str:
        """Define a stub-level type variable for a PEP 695 type parameter.

        Type parameters are scoped to their class / function / alias, so the
        stub may name them freely: an equivalent definition under the same
        name is shared, otherwise a numeric suffix keeps them apart.
        """
        base = obj.__name__
        name, suffix = base, 1
        while True:
            existing = self.defs.get(name)
            if existing is None and name not in self.home_globals:
                break
            if existing is not None and (
                existing.obj is obj or (existing.anonymous and _same_type_param(existing.obj, obj))
            ):
                assert existing.binding is not None
                return self._token(existing.binding)
            suffix += 1
            name = f"{base}_{suffix}"
        d = _Def(name=name, obj=obj, kind=self._home_kind(obj, where), anonymous=True)
        d.binding = self._binding(("home", name), "home", name, target=obj)
        self.defs[name] = d
        self._pending.append(d)
        return self._token(d.binding)

    def _from_binding(self, module: str, attr: str, local: str, target: Any) -> _Binding:
        return self._binding(
            ("from", module, attr), "from", local, module=module, attr=attr, target=target
        )

    def _home_ref(self, parts: list[str], chain: list[Any], where: str) -> str:
        top = self._home_def(parts[0], chain[0], where)
        current = top
        for name, obj in zip(parts[1:], chain[1:]):
            current = self._child_def(current, name, obj, where)
        assert top.binding is not None
        return self._token(top.binding) + "".join(f".{p}" for p in parts[1:])

    def _def_ref(self, d: _Def) -> str:
        """Expression text naming definition *d* (``Outer.Inner`` for a nested one)."""
        names: list[str] = []
        current = d
        while current.parent is not None:
            names.append(current.name)
            current = current.parent
        assert current.binding is not None
        return self._token(current.binding) + "".join(f".{n}" for n in reversed(names))

    def _home_def(self, name: str, obj: Any, where: str) -> _Def:
        existing = self.defs.get(name)
        if existing is not None:
            if existing.obj is obj:
                return existing
            raise ConfigRenderError(
                where or self.home, "<reference>", _describe(obj),
                f"the stub of module {self.home} would define two different objects named "
                f"{name!r} ({_describe(existing.obj)} and {_describe(obj)}); give them "
                f"distinct names",
            )
        d = _Def(name=name, obj=obj, kind=self._home_kind(obj, where))
        d.binding = self._binding(("home", name), "home", name, target=obj)
        self.defs[name] = d
        self._pending.append(d)
        return d

    def _child_def(self, parent: _Def, name: str, obj: Any, where: str) -> _Def:
        existing = parent.children.get(name)
        if existing is not None:
            return existing
        self._check_nestable(parent, obj, where or self.home)
        d = _Def(name=name, obj=obj, kind=self._home_kind(obj, where), parent=parent)
        parent.children[name] = d
        self._pending.append(d)
        return d

    def _home_kind(self, obj: Any, where: str) -> str:
        if _is_class(obj):
            if issubclass(obj, enum.Enum):
                return "enum"
            if obj.__dict__.get("_is_protocol") is True:
                return "protocol"
            if _is_typeddict(obj):
                return "typeddict"
            return "placeholder"
        if isinstance(obj, _PARAMSPEC_TYPES):
            return "paramspec"
        if _TYPEVARTUPLE_TYPES and isinstance(obj, _TYPEVARTUPLE_TYPES):
            return "typevartuple"
        if isinstance(obj, _TYPEVAR_TYPES):
            return "typevar"
        if _NEWTYPE_TYPES and isinstance(obj, _NEWTYPE_TYPES):
            return "newtype"
        if _TYPEALIAS_TYPES and isinstance(obj, _TYPEALIAS_TYPES):
            return "alias"
        raise ConfigRenderError(
            where or self.home, "<reference>", _describe(obj),
            f"it is defined in module {self.home} itself, so the stub would have to define "
            f"it, and a stub can only define classes, type variables, NewType and type "
            f"aliases here",
        )

    def _scopes(self, ctx: _ann.AnnotationContext) -> list[tuple[Mapping[str, Any], str | None]]:
        scopes: list[tuple[Mapping[str, Any], str | None]] = []
        if ctx.globalns:
            scopes.append((ctx.globalns, ctx.module))
        if self.home_globals and all(g is not self.home_globals for g, _ in scopes):
            scopes.append((self.home_globals, self.home))
        return scopes

    def _binding_name(self, globalns: Mapping[str, Any], obj: Any) -> str | None:
        """A module-level name bound to *obj* in *globalns* (its own name first)."""
        own = getattr(obj, "__name__", None)
        if isinstance(own, str) and own.isidentifier() and globalns.get(own, _MISSING) is obj:
            return own
        reverse = self._reverse.get(id(globalns))
        if reverse is None:
            reverse = {}
            for name, value in list(globalns.items()):
                if isinstance(name, str) and name.isidentifier() and not name.startswith("__"):
                    reverse.setdefault(id(value), name)
            self._reverse[id(globalns)] = reverse
            self._keepalive.append(globalns)
        name = reverse.get(id(obj))
        if name is not None and globalns.get(name, _MISSING) is obj:
            return name
        return None

    def _source_imports(self, globalns: Mapping[str, Any]) -> dict[str, _ImportRecord]:
        cached = self._imports.get(id(globalns))
        if cached is None:
            cached = _scan_imports(globalns)
            self._imports[id(globalns)] = cached
        return cached

    def _module(self, name: str) -> types.ModuleType | None:
        module = sys.modules.get(name)
        if module is not None:
            return module
        try:
            return importlib.import_module(name)
        except ImportError:  # "not importable" is the answer needed here
            return None

    def _home_ctx(self) -> _ann.AnnotationContext:
        return _ann.AnnotationContext(globalns=self.home_globals)

    # -- annotations ---------------------------------------------------------

    def annotation(
        self,
        res: _ann.ResolvedAnnotation,
        *,
        leaf: str,
        where: str,
        unwrap: tuple[Any, ...] = (),
    ) -> str | None:
        """Render one resolved annotation (``None`` when there is none)."""
        if res.status == "unannotated":
            return None
        if res.status == "resolved":
            value = res.value
            if unwrap and _is_one_of(get_origin(value), unwrap):
                value = get_args(value)[0]
            return self.render_object(value, res.context, where)
        if res.status == "error" and res.reason == "undefined_name":
            error = res.error(leaf)
            assert error is not None
            raise error
        # Names only imported under TYPE_CHECKING, names local to the function
        # that defines the class, metadata / evaluation problems: render the
        # declared source (metadata is dropped anyway).
        try:
            text = self._render_raw(res.raw, res.context, where)
        except _Undefined:
            error = res.error(leaf)
            if error is not None:
                raise error from None
            raise ConfigRenderError(  # pragma: no cover - the engine reports it first
                where, "<annotation>", res.annotation_text, "it uses an undefined name",
            ) from None
        except _NeedsLocals:
            if res.status == "degraded" and res.reason == "function_locals":
                res.warn(leaf, stacklevel=5)
            else:  # pragma: no cover - the engine classifies these as function_locals
                warnings.warn(
                    f"{leaf}.{res.name}: annotation {res.annotation_text!r} needs names local "
                    f"to the function that defines the class, which conscribe cannot see; the "
                    f"stub types it as Any",
                    ExtractionWarning,
                    stacklevel=5,
                )
            return self.ref(typing.Any)
        if unwrap:
            text = _strip_wrappers(text)
        return text

    def _render_raw(self, raw: Any, ctx: _ann.AnnotationContext, where: str) -> str:
        if isinstance(raw, str):
            return self.render_text(raw, ctx, where)
        if isinstance(raw, ForwardRef):
            return self.render_text(raw.__forward_arg__, ctx, where)
        return self.render_object(raw, ctx, where)

    def render_object(
        self, tp: Any, ctx: _ann.AnnotationContext | None, where: str, depth: int = 0
    ) -> str:
        """Render a typing object as stub source (with name tokens)."""
        if depth > 60:
            raise ConfigRenderError(where, "<annotation>", _safe_repr(tp), "it nests too deeply")
        nxt = depth + 1
        if tp is None or tp is NoneType:
            return "None"
        if tp is Ellipsis:
            return "..."
        if isinstance(tp, (str, ForwardRef)):
            text = tp if isinstance(tp, str) else tp.__forward_arg__
            return self.render_text(text, ctx or self._home_ctx(), where)
        if isinstance(tp, dataclasses.InitVar):
            return self.render_object(tp.type, ctx, where, nxt)
        if tp is dataclasses.InitVar:
            return self.ref(typing.Any)
        origin = get_origin(tp)
        args = get_args(tp)
        if _is_one_of(origin, _ANNOTATED):
            return self.render_object(args[0], ctx, where, nxt)
        if origin in _ann.UNION_ORIGINS:
            return " | ".join(self.render_object(a, ctx, where, nxt) for a in args)
        if _is_one_of(origin, _LITERAL):
            values = ", ".join(self.literal_value(v, ctx, where) for v in args)
            return f"{self.ref(origin, ctx, where)}[{values}]"
        if _PARAMSPEC_ARGS and isinstance(tp, _PARAMSPEC_ARGS):
            return f"{self.render_object(tp.__origin__, ctx, where, nxt)}.args"
        if _PARAMSPEC_KWARGS and isinstance(tp, _PARAMSPEC_KWARGS):
            return f"{self.render_object(tp.__origin__, ctx, where, nxt)}.kwargs"
        if origin is not None:
            head = self.ref(_typing_alias(tp, origin), ctx, where)
            if (_SPECIAL_ALIAS and isinstance(tp, _SPECIAL_ALIAS)) or self._is_any_def(head):
                return head  # bare ``typing.List`` and friends; ``X = Any`` takes no arguments
            if origin is collections.abc.Callable and len(args) == 2:
                params, ret = args
                if isinstance(params, list):
                    rendered = "[" + ", ".join(self.render_object(a, ctx, where, nxt) for a in params) + "]"
                else:
                    rendered = self.render_object(params, ctx, where, nxt)
                return f"{head}[{rendered}, {self.render_object(ret, ctx, where, nxt)}]"
            if origin is tuple and args in ((), ((),)):
                return f"{head}[()]"
            if not args:
                return head
            return f"{head}[{', '.join(self._render_arg(a, ctx, where, nxt) for a in args)}]"
        generic = _pydantic_generic(tp)
        if generic is not None:
            g_origin, g_args = generic
            head = self.ref(g_origin, ctx, where)
            return f"{head}[{', '.join(self._render_arg(a, ctx, where, nxt) for a in g_args)}]"
        if _is_class(tp) or _is_named_typing_object(tp):
            return self.ref(tp, ctx, where)
        raise ConfigRenderError(
            where, "<annotation>", _safe_repr(tp), "it is not a type a stub can express",
        )

    def _render_arg(self, arg: Any, ctx: _ann.AnnotationContext | None, where: str, depth: int) -> str:
        if isinstance(arg, (list, tuple)):
            return "[" + ", ".join(self.render_object(a, ctx, where, depth) for a in arg) + "]"
        return self.render_object(arg, ctx, where, depth)

    def _is_any_def(self, text: str) -> bool:
        match = _TOKEN_RE.fullmatch(text)
        if match is None:
            return False
        binding = self.bindings[int(match.group(1))]
        d = self.defs.get(binding.preferred) if binding.kind == "home" else None
        return d is not None and d.kind == "any"

    def literal_value(self, value: Any, ctx: _ann.AnnotationContext | None, where: str) -> str:
        """Render one ``Literal[...]`` argument."""
        if value is None:
            return "None"
        if isinstance(value, enum.Enum):
            enum_cls = type(value)
            if not value.name.isidentifier() or getattr(enum_cls, value.name, None) is not value:
                raise ConfigRenderError(
                    where, "<annotation>", repr(value), "the enum member cannot be named in source",
                )
            return f"{self.ref(enum_cls, ctx, where)}.{value.name}"
        if type(value) in (bool, int, str, bytes):
            return repr(value)
        raise ConfigRenderError(
            where, "<annotation>", _safe_repr(value),
            "Literal values must be None, bool, int, str, bytes or enum members",
        )

    # -- source-text annotations ---------------------------------------------

    def render_text(self, text: str, ctx: _ann.AnnotationContext, where: str) -> str:
        """Render annotation source text, rebinding every name it uses.

        Used for annotations that cannot be evaluated (names only imported
        under ``if TYPE_CHECKING:``).  ``Annotated`` metadata is dropped,
        names that resolve at runtime are rendered from their objects, the
        rest are imported from the declaring module's TYPE_CHECKING block.
        """
        try:
            node = ast.parse(text.strip(), mode="eval").body
        except SyntaxError as exc:
            raise ConfigRenderError(
                where, "<annotation>", repr(text), f"not a valid expression: {exc.msg}",
            ) from None
        return ast.unparse(self._tx(node, ctx, where, 0))

    def _tx(self, node: ast.expr, ctx: _ann.AnnotationContext, where: str, depth: int) -> ast.expr:
        if depth > 60:
            raise ConfigRenderError(where, "<annotation>", ast.unparse(node), "it nests too deeply")
        nxt = depth + 1
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str):  # a nested forward reference
                try:
                    inner = ast.parse(node.value.strip(), mode="eval").body
                except SyntaxError:
                    raise ConfigRenderError(
                        where, "<annotation>", repr(node.value), "not a valid forward reference",
                    ) from None
                return self._tx(inner, ctx, where, nxt)
            if node.value is None or node.value is Ellipsis:
                return node
            raise ConfigRenderError(where, "<annotation>", ast.unparse(node), "a constant is not a type")
        if isinstance(node, (ast.Name, ast.Attribute)):
            return self._tx_ref(node, ctx, where)
        if isinstance(node, ast.Subscript):
            found, head_obj = self._lookup_chain(node.value, ctx)
            tail = _dotted_tail(node.value)
            elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            if (found and _is_one_of(head_obj, _ANNOTATED)) or (not found and tail == "Annotated"):
                return self._tx(elements[0], ctx, where, nxt)
            if found and head_obj is dataclasses.InitVar:
                return self._tx(elements[0], ctx, where, nxt)
            head = self._tx(node.value, ctx, where, nxt)
            if (found and _is_one_of(head_obj, _LITERAL)) or (not found and tail == "Literal"):
                items = [self._tx_literal(e, ctx, where) for e in elements]
            else:
                items = [self._tx(e, ctx, where, nxt) for e in elements]
            slice_: ast.expr = (
                ast.Tuple(elts=items, ctx=ast.Load()) if isinstance(node.slice, ast.Tuple) else items[0]
            )
            return ast.Subscript(value=head, slice=slice_, ctx=ast.Load())
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return ast.BinOp(
                left=self._tx(node.left, ctx, where, nxt),
                op=ast.BitOr(),
                right=self._tx(node.right, ctx, where, nxt),
            )
        if isinstance(node, ast.List):
            return ast.List(elts=[self._tx(e, ctx, where, nxt) for e in node.elts], ctx=ast.Load())
        if isinstance(node, ast.Tuple):
            return ast.Tuple(elts=[self._tx(e, ctx, where, nxt) for e in node.elts], ctx=ast.Load())
        raise ConfigRenderError(
            where, "<annotation>", ast.unparse(node),
            "this expression is not a type annotation a stub can express",
        )

    def _tx_literal(self, node: ast.expr, ctx: _ann.AnnotationContext, where: str) -> ast.expr:
        if isinstance(node, ast.Constant) and (
            node.value is None or type(node.value) in (bool, int, str, bytes)
        ):
            return node
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and type(node.operand.value) is int
        ):
            return node
        if isinstance(node, (ast.Name, ast.Attribute)):
            found, value = self._lookup_chain(node, ctx)
            if found and isinstance(value, enum.Enum):
                return ast.parse(self.literal_value(value, ctx, where), mode="eval").body
        raise ConfigRenderError(
            where, "<annotation>", ast.unparse(node),
            "Literal values must be None, bool, int, str, bytes or enum members",
        )

    def _lookup_chain(self, node: ast.expr, ctx: _ann.AnnotationContext) -> tuple[bool, Any]:
        """Resolve a ``Name`` / dotted ``Attribute`` chain at runtime: ``(found, obj)``."""
        parts = _dotted_parts(node)
        if parts is None:
            return False, None
        found, obj = ctx.lookup(parts[0])
        if not found:
            return False, None
        for part in parts[1:]:
            try:
                obj = getattr(obj, part)
            except AttributeError:
                return False, None
        return True, obj

    def _tx_ref(self, node: ast.expr, ctx: _ann.AnnotationContext, where: str) -> ast.expr:
        parts = _dotted_parts(node)
        if parts is None:  # pragma: no cover - only Name / Attribute reach here
            raise ConfigRenderError(where, "<annotation>", ast.unparse(node), "not a dotted name")
        found, obj = ctx.lookup(parts[0])
        if found:
            return self._tx_ref_object(obj, parts, ctx, where)

        records = _ann.type_checking_imports(ctx)
        record = records.get(parts[0])
        if record is not None:
            rest = parts[1:]
            if _is_plain_import(record):
                # ``import a.b.c`` under TYPE_CHECKING: repeat the statement
                # that makes the dotted reference valid (not just ``import a``).
                module = _covering_import(records, parts) or parts[0]
                rest = parts[len(module.split(".")):]
                binding = self._binding(
                    ("import", module), "import", module, module=module, verified=False
                )
            else:
                binding = self._type_checking_binding(record, ctx, where)
            text = self._token(binding) + "".join(f".{p}" for p in rest)
            return ast.parse(text, mode="eval").body
        if ctx.function_local:
            local = self._local_class(parts[0], ctx)
            if local is None:
                raise _NeedsLocals(parts[0])
            return self._tx_ref_object(local, parts, ctx, where)
        raise _Undefined(parts[0])

    def _local_class(self, name: str, ctx: _ann.AnnotationContext) -> type | None:
        """A function-local class an annotation can only mean by *name*.

        Under PEP 563 a class defined inside a function cannot resolve its own
        name, its bases' names or its siblings' names; the declaring class,
        the classes of its MRO and the classes this stub already defines are
        the only candidates conscribe can see.
        """
        owner = ctx.owner
        if owner is not None:
            for klass in owner.__mro__:
                if klass.__name__ == name:
                    return klass
        d = self.defs.get(name)
        if d is not None and _is_class(d.obj):
            return d.obj
        return None

    def _tx_ref_object(
        self, obj: Any, parts: list[str], ctx: _ann.AnnotationContext, where: str
    ) -> ast.expr:
        for index, part in enumerate(parts[1:], start=1):
            try:
                obj = getattr(obj, part)
            except AttributeError:
                raise ConfigRenderError(
                    where, "<annotation>", ".".join(parts),
                    f"{'.'.join(parts[:index])} has no attribute {part!r}",
                ) from None
        if isinstance(obj, types.ModuleType):
            raise ConfigRenderError(where, "<annotation>", ".".join(parts), "a module is not a type")
        # Classes and named typing objects are referred to by name; a
        # module-level alias such as ``Pair = tuple[int, int]`` is inlined
        # (render_object checks for a parametrized alias before treating
        # anything as a class: ``tuple[int, int]`` is a ``type`` on 3.10).
        text = self.render_object(obj, ctx, where)
        return ast.parse(text, mode="eval").body

    def _type_checking_binding(
        self, record: _ann.TypeCheckingName, ctx: _ann.AnnotationContext, where: str
    ) -> _Binding:
        """Binding for a name bound only under ``if TYPE_CHECKING:`` (never imported)."""
        if record.module is None:
            # Defined (not imported) under TYPE_CHECKING in the declaring module:
            # type checkers can import it from there, but not from this stub.
            if ctx.module and ctx.module != self.home:
                return self._binding(
                    ("from", ctx.module, record.name), "from", record.name,
                    module=ctx.module, attr=record.name, verified=False,
                )
            existing = self.defs.get(record.name)
            if existing is not None and existing.kind != "any":
                raise ConfigRenderError(  # pragma: no cover - needs a runtime/TC name clash
                    where, "<annotation>", record.name,
                    f"{record.name!r} is bound both at runtime and under 'if TYPE_CHECKING:' "
                    f"in {self.home}",
                )
            if existing is None:
                existing = _Def(name=record.name, obj=None, kind="any")
                existing.binding = self._binding(("home", record.name), "home", record.name)
                existing.header = [f"{existing.name_text} = {self.ref(typing.Any)}"]
                self.defs[record.name] = existing
            assert existing.binding is not None
            return existing.binding
        if record.attr is None:  # ``import module [as name]``
            return self._binding(
                ("import", record.module), "import", record.name,
                module=record.module, verified=False,
            )
        return self._binding(
            ("from", record.module, record.attr), "from", record.name,
            module=record.module, attr=record.attr, verified=False,
        )

    # -- stubbed classes --------------------------------------------------------

    def _render_stubbed(self, d: _Def) -> None:
        info = d.info
        assert info is not None
        cls = info.cls
        ctx = _ann.class_context(cls)
        where = qualified_name(cls)
        transformed = _dataclass_transformed(cls)
        bases: tuple[Any, ...] = tuple(info.bases)
        if bases == tuple(cls.__bases__):
            bases = cls.__dict__.get("__orig_bases__") or bases  # keep generic arguments
        d.header = [self._class_line(d, bases, ctx, where)]

        attrs = [self._injected_line(a, ctx, where, transformed) for a in info.injected_attrs]
        class_attrs = self._class_attr_lines(info.class_attrs, transformed)
        values = self._value_lines(info.values, transformed)
        functions: list[str] = []
        if info.init is not None:
            functions.extend(self._function_lines(info.init, cls, is_init=True))
        for method in info.methods:
            functions.extend(self._function_lines(method, cls))
        if self.partial_class_fallback and not any(m.name == "__getattr__" for m in info.methods):
            incomplete = self._token(
                self._binding(
                    ("from", "_typeshed", "Incomplete"), "from", "Incomplete",
                    module="_typeshed", attr="Incomplete", verified=False,
                )
            )
            functions.append(f"def __getattr__(self, name: str) -> {incomplete}: ...")
        d.body = _join_sections(attrs, class_attrs, values, functions)
        # Classes defined in the body: the stub replaces the class, so they
        # are declared too (nominal placeholders unless stubbed themselves).
        for name, nested in info.nested:
            if name not in d.children:
                self._child_def(d, name, nested, f"{where}.{name}")

    def _class_line(
        self, d: _Def, bases: tuple[Any, ...], ctx: _ann.AnnotationContext, where: str
    ) -> str:
        capture: set[int] = set()
        self._captures.append(capture)
        try:
            rendered = [
                self.render_object(base, ctx, f"{where} (base class)")
                for base in bases
                if base is not object
            ]
        finally:
            self._captures.pop()
            d.deps |= capture
        return f"class {d.name_text}({', '.join(rendered)}):" if rendered else f"class {d.name_text}:"

    def _injected_line(
        self, attr: InjectedAttr, ctx: _ann.AnnotationContext, where: str, transformed: bool
    ) -> str:
        type_text = self.ref(attr.resolved_type, ctx, f"{where}.{attr.name}")
        default = " = ..." if transformed else ""
        comment = f"  # wired from: {attr.registry_name}" if attr.registry_name else ""
        return f"{attr.name}: {type_text}{default}{comment}"

    def _class_attr_lines(self, attrs: tuple[ClassAttrStub, ...], transformed: bool) -> list[str]:
        lines: list[str] = []
        for attr in attrs:
            where = f"{qualified_name(attr.owner)}.{attr.name}"
            res = _ann.resolve_class_annotation(attr.owner, attr.name)
            if res is None:
                res = _ann.resolve_annotation(
                    attr.annotation, _ann.class_context(attr.owner), name=attr.name
                )
            text = self.annotation(res, leaf=attr.owner.__qualname__, where=where)
            if text is None:  # pragma: no cover - own annotations are never empty
                continue
            if self._is_bare_final(res):
                lines.append(self._final_line(attr, text, where))
                continue
            default = " = ..." if transformed and attr.has_default else ""
            lines.append(f"{attr.name}: {text}{default}")
        return lines

    def _is_bare_final(self, res: _ann.ResolvedAnnotation) -> bool:
        """``x: Final = 3`` -- a ``Final`` whose type comes from the value."""
        if res.status == "resolved":
            return _is_one_of(res.value, _FINAL)
        try:
            node = ast.parse(res.annotation_text.strip(), mode="eval").body
        except SyntaxError:
            return False
        if _dotted_tail(node) != "Final":
            return False
        found, obj = self._lookup_chain(node, res.context)
        return _is_one_of(obj, _FINAL) if found else True

    def _final_line(self, attr: ClassAttrStub, final_text: str, where: str) -> str:
        """A bare ``Final`` keeps its value's type: ``LIMIT: Final = 3`` or ``Final[T]``.

        A bare ``Final`` without the value would leave type checkers nothing
        to infer the type from.
        """
        value = attr.owner.__dict__.get(attr.name, _MISSING)
        if value is _MISSING:
            return f"{attr.name}: {final_text}[{self.ref(typing.Any)}]"
        if type(value) in _FINAL_LITERAL_TYPES:
            literal = _default_to_source(value)
            if literal != "..." and len(literal) <= 80:
                return f"{attr.name}: {final_text} = {literal}"
        ctx = _ann.class_context(attr.owner)
        return f"{attr.name}: {final_text}[{self._value_type(value, ctx, where)}]"

    def _value_lines(self, values: tuple[ClassValueStub, ...], transformed: bool) -> list[str]:
        """Unannotated class values, typed from the value (a type alias for typing constructs).

        In a class a ``dataclass_transform`` applies to (pydantic models) a
        plain annotation would declare a field: the value is a ``ClassVar``.
        """
        lines: list[str] = []
        for v in values:
            where = f"{qualified_name(v.owner)}.{v.name}"
            ctx = _ann.class_context(v.owner)
            if get_origin(v.value) is not None and not _is_class(v.value):
                alias = self.render_object(v.value, ctx, where)
                lines.append(f"{v.name}: {self.ref(typing.TypeAlias)} = {alias}")
                continue
            text = self._value_type(v.value, ctx, where)
            if transformed:
                text = f"{self.ref(typing.ClassVar)}[{text}]"
            lines.append(f"{v.name}: {text}")
        return lines

    def _value_type(
        self, value: Any, ctx: _ann.AnnotationContext, where: str, depth: int = 0
    ) -> str:
        """The type of an unannotated value, as stub source (``Any`` when it cannot be named)."""
        any_text = self.ref(typing.Any)
        if depth > 3:
            return any_text
        if value is None:
            return "None"
        tp = type(value)
        if tp in (bool, int, float, complex, str, bytes):
            return self.ref(tp)
        if tp in (list, set, frozenset):
            items = list(value)
            if tp is not list:  # set order is not stable across runs
                items.sort(key=lambda item: (type(item).__module__, type(item).__qualname__))
            return f"{self.ref(tp)}[{self._joined_types(items, ctx, where, depth)}]"
        if tp is tuple:
            if not value:
                return f"{self.ref(tuple)}[()]"
            if len(value) <= 6:
                inner = ", ".join(self._value_type(item, ctx, where, depth + 1) for item in value)
                return f"{self.ref(tuple)}[{inner}]"
            return f"{self.ref(tuple)}[{self._joined_types(list(value), ctx, where, depth)}, ...]"
        if tp is dict:
            keys = self._joined_types(list(value), ctx, where, depth)
            items = self._joined_types(list(value.values()), ctx, where, depth)
            return f"{self.ref(dict)}[{keys}, {items}]"
        if _is_class(value):
            inner = self.ref(value, ctx, where) if self._nameable(value) else any_text
            return f"{self.ref(type)}[{inner}]"
        if self._nameable(tp):
            return self.ref(tp, ctx, where)
        return any_text

    def _joined_types(
        self, items: list[Any], ctx: _ann.AnnotationContext, where: str, depth: int
    ) -> str:
        """``A | B`` for the value types of *items* (``Any`` when empty or too varied)."""
        any_text = self.ref(typing.Any)
        texts: list[str] = []
        for item in items[:50]:
            text = self._value_type(item, ctx, where, depth + 1)
            if text == any_text:
                return any_text
            if text not in texts:
                texts.append(text)
            if len(texts) > 4:
                return any_text
        return " | ".join(texts) if texts else any_text

    def _nameable(self, obj: Any) -> bool:
        """True when :meth:`ref` can name *obj* (an import or a same-module definition)."""
        if _is_builtin(obj):
            return True
        module_name = getattr(obj, "__module__", None)
        qual = getattr(obj, "__qualname__", None)
        if not isinstance(module_name, str) or not isinstance(qual, str):
            return False
        parts = qual.split(".")
        if module_name == "builtins" or not all(p.isidentifier() for p in parts):
            return False
        module = self.home_module if module_name == self.home else sys.modules.get(module_name)
        chain = _walk_chain(module, parts) if module is not None else None
        return chain is not None and chain[-1] is obj

    def _function_lines(self, method: MethodStub, cls: type, *, is_init: bool = False) -> list[str]:
        lines = ["@" + self._decorator(name, method.name) for name in method.decorators]
        keyword_ = "async def" if method.is_async else "def"
        if method.kind == "opaque":
            signature = self._opaque_signature(method)
        else:
            signature = self._signature(method, cls, is_init=is_init)
        lines.append(f"{keyword_} {method.name}{signature}: ...")
        return lines

    def _opaque_signature(self, method: MethodStub) -> str:
        """A wrapper conscribe cannot see through: any arguments, ``Any`` result."""
        any_text = self.ref(typing.Any)
        decorators = method.decorators
        if "property" in decorators or "cached_property" in decorators:
            return f"(self) -> {any_text}"
        if "setter" in decorators:
            return f"(self, value: {any_text}) -> None"
        if "deleter" in decorators:
            return "(self) -> None"
        first = (
            "" if "staticmethod" in decorators
            else "cls, " if "classmethod" in decorators
            else "self, "
        )
        return f"({first}*args: {any_text}, **kwargs: {any_text}) -> {any_text}"

    def _decorator(self, decorator: str, name: str) -> str:
        if decorator in ("setter", "deleter"):
            return f"{name}.{decorator}"
        target = {
            "classmethod": classmethod,
            "staticmethod": staticmethod,
            "property": property,
            "cached_property": functools.cached_property,
            "abstractmethod": abc.abstractmethod,
        }.get(decorator)
        if target is None:  # pragma: no cover - the collector emits known names only
            raise ConfigRenderError(name, "<decorator>", decorator, "unknown decorator")
        return self.ref(target)

    def _signature(self, method: MethodStub, cls: type, *, is_init: bool) -> str:
        func = method.function
        where = f"{qualified_name(cls)}.{method.name}"
        leaf = f"{cls.__qualname__}.{method.name}"
        sig = _ann.safe_signature(func)
        if sig is None:
            warnings.warn(
                f"{where}: conscribe cannot read its signature; the stub declares it as "
                f"accepting any arguments",
                ExtractionWarning,
                stacklevel=4,
            )
            any_text = self.ref(typing.Any)
            first = (
                "" if "staticmethod" in method.decorators
                else "cls, " if "classmethod" in method.decorators
                else "self, "
            )
            return f"({first}*args: {any_text}, **kwargs: {any_text}) -> {'None' if is_init else any_text}"

        annotations, closure = _ann.function_annotation_source(func)
        fctx = _ann.function_context(func, cls, closure=closure)
        generated = is_init and is_generated_init(cls, func)
        params: list[inspect.Parameter] = []
        for param in sig.parameters.values():
            res = self._param_resolution(param, annotations, fctx, cls, generated)
            text = self.annotation(res, leaf=leaf, where=f"{where}({param.name})")
            annotation: Any = _Source(text) if text is not None else inspect.Parameter.empty
            default = param.default
            if default is not inspect.Parameter.empty:
                default = _Source(_default_to_source(default))
            params.append(param.replace(annotation=annotation, default=default))

        raw_return = sig.return_annotation
        if method.kind in ("contextmanager", "asynccontextmanager"):
            return_text = self._context_manager_return(
                method, raw_return, fctx, leaf=leaf, where=f"{where}(return)"
            )
        elif raw_return is inspect.Signature.empty:
            return_text = "None" if is_init else None
        else:
            res = _ann.resolve_annotation(raw_return, fctx, name="return")
            return_text = self.annotation(res, leaf=leaf, where=f"{where}(return)")
        return_annotation: Any = (
            _Source(return_text) if return_text is not None else inspect.Signature.empty
        )
        return str(sig.replace(parameters=params, return_annotation=return_annotation))

    def _context_manager_return(
        self,
        method: MethodStub,
        raw_return: Any,
        fctx: _ann.AnnotationContext,
        *,
        leaf: str,
        where: str,
    ) -> str:
        """``AbstractContextManager[T]`` for a ``@contextmanager`` method (async: the async ABC).

        ``T`` is what the generator yields, read from its ``Iterator[T]`` /
        ``Generator[T, ...]`` (async: ``AsyncIterator[T]`` ...) annotation;
        ``Any`` when it has none.
        """
        manager = (
            contextlib.AbstractContextManager
            if method.kind == "contextmanager"
            else contextlib.AbstractAsyncContextManager
        )
        head = self.ref(manager)
        item: str | None = None
        if raw_return is not inspect.Signature.empty:
            res = _ann.resolve_annotation(raw_return, fctx, name="return")
            item = self._yielded(res, leaf=leaf, where=where)
        return f"{head}[{item or self.ref(typing.Any)}]"

    def _yielded(self, res: _ann.ResolvedAnnotation, *, leaf: str, where: str) -> str | None:
        """The ``T`` of a generator's ``Iterator[T]``-like return annotation, rendered."""
        if res.status == "unannotated":
            return None
        if res.status == "resolved":
            value = res.value
            if _is_one_of(get_origin(value), _ANNOTATED):
                value = get_args(value)[0]
            args = get_args(value)
            if get_origin(value) in _ITERATOR_ORIGINS and args:
                return self.render_object(args[0], res.context, where)
            return None
        # Not evaluable at runtime (TYPE_CHECKING-only names): read the text.
        try:
            node = ast.parse(res.annotation_text.strip(), mode="eval").body
        except SyntaxError:
            return None
        if not isinstance(node, ast.Subscript) or _dotted_tail(node.value) not in _ITERATOR_NAMES:
            return None
        first = node.slice.elts[0] if isinstance(node.slice, ast.Tuple) else node.slice
        inner = _ann.resolve_annotation(ast.unparse(first), res.context, name=res.name)
        return self.annotation(inner, leaf=leaf, where=where)

    @staticmethod
    def _param_resolution(
        param: inspect.Parameter,
        annotations: dict[str, Any],
        fctx: _ann.AnnotationContext,
        cls: type,
        generated: bool,
    ) -> _ann.ResolvedAnnotation:
        if generated:
            fields = cls.__dict__.get("__dataclass_fields__")
            if isinstance(fields, dict) and param.name in fields:
                # A generated dataclass __init__: the annotation belongs to the
                # class body that declares the field (maybe another module).
                owner = (
                    dataclass_field_owner(cls, param.name)
                    or _ann.annotation_owner(cls, param.name)
                    or cls
                )
                res = _ann.resolve_class_annotation(owner, param.name)
                if res is not None:
                    return res
                return _ann.resolve_annotation(
                    param.annotation, _ann.class_context(owner), name=param.name
                )
        raw = annotations.get(param.name, param.annotation)
        return _ann.resolve_annotation(raw, fctx, name=param.name)

    # -- same-module definitions ------------------------------------------------

    def _render_def(self, d: _Def) -> None:
        obj = d.obj
        where = qualified_name(obj) if _is_class(obj) else f"{self.home}.{d.name}"
        if d.kind == "placeholder":
            self._placeholder(d, where)
        elif d.kind == "enum":
            self._enum(d, where)
        elif d.kind == "protocol":
            self._protocol(d, where)
        elif d.kind == "typeddict":
            self._typeddict(d, where)
        elif d.kind in ("typevar", "paramspec", "typevartuple"):
            self._typevar(d, where)
        elif d.kind == "newtype":
            supertype = self._eager(d, obj.__supertype__, where)
            d.header = [f"{d.name_text} = {self.ref(typing.NewType)}({d.name!r}, {supertype})"]
        elif d.kind == "alias":
            value = self._eager(d, obj.__value__, where)
            d.header = [f"{d.name_text}: {self.ref(typing.TypeAlias)} = {value}"]
        else:  # pragma: no cover - _home_kind() only returns the kinds above
            raise ConfigRenderError(where, "<definition>", _describe(obj), "cannot define it")

    def _eager(
        self, d: _Def, tp: Any, where: str, ctx: _ann.AnnotationContext | None = None
    ) -> str:
        """Render *tp* in a header (evaluated eagerly): record its same-module deps."""
        capture: set[int] = set()
        self._captures.append(capture)
        try:
            return self.render_object(tp, ctx or _module_ctx(d.obj, self.home_globals), where)
        finally:
            self._captures.pop()
            d.deps |= capture

    def _placeholder(self, d: _Def, where: str) -> None:
        bases = [self.ref(typing.Any)]
        params: tuple[Any, ...] = tuple(getattr(d.obj, "__parameters__", ()) or ())
        if params and issubclass(d.obj, typing.Generic):  # type: ignore[arg-type]
            rendered = ", ".join(self._eager(d, p, where) for p in params)
            bases.append(f"{self.ref(typing.Generic)}[{rendered}]")
        d.header = [f"class {d.name_text}({', '.join(bases)}):"]
        d.body = []

    def _enum(self, d: _Def, where: str) -> None:
        cls = d.obj
        d.header = [self._class_line(d, _own_bases(cls), _ann.class_context(cls), where)]
        members = [
            f"{name} = {_default_to_source(member.value)}"
            for name, member in cls.__members__.items()
            if name.isidentifier()
        ]
        skip = frozenset(cls.__members__)
        d.body = _join_sections(members, *self._own_members(cls, skip=skip, skip_sunder=True))

    def _protocol(self, d: _Def, where: str) -> None:
        cls = d.obj
        decorators = []
        if cls.__dict__.get("_is_runtime_protocol") is True:
            decorators.append("@" + self.ref(typing.runtime_checkable))
        d.header = [*decorators, self._class_line(d, _own_bases(cls), _ann.class_context(cls), where)]
        # A Protocol's ``__init__`` is typing's placeholder, never a member.
        d.body = _join_sections(*self._own_members(cls, with_init=False))

    def _own_members(
        self,
        cls: type,
        *,
        skip: frozenset[str] = frozenset(),
        skip_sunder: bool = False,
        with_init: bool = True,
    ) -> tuple[list[str], list[str]]:
        members = collect_class_members(cls, skip=skip)
        methods, class_attrs, values = members.methods, members.class_attrs, members.values
        if skip_sunder:
            methods = tuple(m for m in methods if not _is_sunder(m.name))
            class_attrs = tuple(a for a in class_attrs if not _is_sunder(a.name))
            values = tuple(v for v in values if not _is_sunder(v.name))
        transformed = _dataclass_transformed(cls)
        attr_lines = self._class_attr_lines(class_attrs, transformed)
        attr_lines += self._value_lines(values, transformed)
        functions: list[str] = []
        if members.init is not None and with_init:
            functions.extend(self._function_lines(members.init, cls, is_init=True))
        for method in methods:
            functions.extend(self._function_lines(method, cls))
        return attr_lines, functions

    def _typeddict(self, d: _Def, where: str) -> None:
        cls = d.obj
        total = bool(getattr(cls, "__total__", True))
        required = frozenset(getattr(cls, "__required_keys__", ()) or ())
        bases = [self.ref(typing.TypedDict)]
        params: tuple[Any, ...] = tuple(getattr(cls, "__parameters__", ()) or ())
        if params:
            rendered = ", ".join(self._eager(d, p, where) for p in params)
            bases.append(f"{self.ref(typing.Generic)}[{rendered}]")
        total_kw = "" if total else ", total=False"
        d.header = [f"class {d.name_text}({', '.join(bases)}{total_kw}):"]
        lines: list[str] = []
        for name in _ann.own_annotations(cls):
            if not name.isidentifier():
                continue
            res = _ann.resolve_class_annotation(cls, name)
            if res is None:  # pragma: no cover - the name came from own_annotations
                continue
            text = self.annotation(
                res, leaf=cls.__qualname__, where=f"{where}.{name}", unwrap=_TD_WRAPPERS
            )
            if name in required and not total:
                text = f"{self.ref(_typing_form('Required'))}[{text}]"
            elif name not in required and total:
                text = f"{self.ref(_typing_form('NotRequired'))}[{text}]"
            lines.append(f"{name}: {text}")
        d.body = lines

    def _typevar(self, d: _Def, where: str) -> None:
        tv = d.obj
        ctx = _module_ctx(tv, self.home_globals)
        args = [repr(d.name)]
        factory: Any = type(tv)
        if d.kind != "typevartuple":
            for constraint in getattr(tv, "__constraints__", ()) or ():
                args.append(self._eager(d, constraint, where, ctx))
            bound = getattr(tv, "__bound__", None)
            if bound is not None:
                args.append(f"bound={self._eager(d, bound, where, ctx)}")
            if getattr(tv, "__covariant__", False):
                args.append("covariant=True")
            if getattr(tv, "__contravariant__", False):
                args.append("contravariant=True")
            if getattr(tv, "__infer_variance__", False):
                args.append("infer_variance=True")
                if sys.version_info < (3, 12) and _te is not None:  # pragma: no cover - 3.10/3.11
                    factory = _te.TypeVar
        d.header = [f"{d.name_text} = {self.ref(factory)}({', '.join(args)})"]

    # -- assembly -----------------------------------------------------------------

    def _ordered_defs(self) -> list[_Def]:
        """Top-level definitions, each after the same-module objects its header uses."""
        by_index = {d.binding.index: d for d in self.defs.values() if d.binding is not None}
        ordered: list[_Def] = []
        visited: set[int] = set()

        def visit(d: _Def) -> None:
            assert d.binding is not None
            if d.binding.index in visited:
                return
            visited.add(d.binding.index)
            for dep in sorted(_all_deps(d)):
                target = by_index.get(dep)
                if target is not None:
                    visit(target)
            ordered.append(d)

        helpers = [d for d in self.defs.values() if d.kind != "stubbed"]
        stubbed = [d for d in self.defs.values() if d.kind == "stubbed"]
        for d in helpers + stubbed:
            visit(d)
        return ordered

    def _assemble(self, d: _Def, indent: str) -> list[str]:
        out = [indent + line for line in d.header]
        if d.body is None:
            return out
        inner = list(d.body)
        for child in _ordered_children(d):
            if inner:
                inner.append("")
            inner.extend(self._assemble(child, ""))
        if not inner:
            out[-1] += " ..."
            return out
        out.extend(f"{indent}    {line}" if line else "" for line in inner)
        return out

    # -- the gate -----------------------------------------------------------------

    def _check(self, source: str) -> None:
        """Refuse a stub that does not parse, binds a top-level name twice, uses
        an unbound name or imports something that does not exist."""
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:  # pragma: no cover - renderer bug guard
            raise ConfigRenderError(
                self.home, "<stub>", f"line {exc.lineno}",
                f"the generated stub does not parse: {exc.msg}",
            ) from None

        bound: set[str] = set()
        packages: set[str] = set()  # bound by unaliased ``import a`` / ``import a.b``
        for node in tree.body:
            for name in _bound_names(node):
                plain = isinstance(node, ast.Import) and _plain_import_binds(node, name)
                if name in bound and not (plain and name in packages):
                    raise ConfigRenderError(
                        self.home, "<stub>", repr(name), "the generated stub would bind it twice",
                    )
                bound.add(name)
                if plain:
                    packages.add(name)

        unbound = sorted(_unbound_names(tree, bound))
        if unbound:
            raise ConfigRenderError(
                self.home, "<stub>", ", ".join(repr(n) for n in unbound),
                "the generated stub would use names it does not bind",
            )

        shadowed = sorted(_class_shadowed_names(tree))
        if shadowed:  # pragma: no cover - renderer bug guard (_collect_shadowing renames)
            raise ConfigRenderError(
                self.home, "<stub>", ", ".join(repr(n) for n in shadowed),
                "a class body of the generated stub binds these names and also uses them in "
                "annotations, where they would mean the class attribute",
            )

        for binding in self.bindings:
            if binding.kind not in ("from", "import") or not binding.verified:
                continue
            if binding.module in _STUB_ONLY_MODULES:  # pragma: no cover - never verified
                continue
            module = self._module(binding.module or "")
            if binding.kind == "import":
                ok = module is not None
            else:
                value = getattr(module, binding.attr or "", _MISSING) if module is not None else _MISSING
                ok = value is not _MISSING and (binding.target is None or value is binding.target)
            if not ok:
                statement = (
                    f"import {binding.module}" if binding.kind == "import"
                    else f"from {binding.module} import {binding.attr}"
                )
                raise ConfigRenderError(
                    self.home, "<stub>", statement,
                    "this import would not bind the object the stub refers to",
                )


# ── Module helpers ───────────────────────────────────────────────


class _Source:
    """Carries pre-rendered source text through ``str(inspect.Signature)``.

    ``inspect`` formats annotations and defaults with ``repr()``; wrapping the
    rendered text lets ``Signature.__str__`` keep placing the ``/`` and ``*``
    markers and ``*args`` / ``**kwargs``.
    """

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return self.text


# Types whose ``repr()`` may round-trip to valid, self-contained source.
_LITERAL_TYPES: tuple[type, ...] = (
    bool, int, float, complex, str, bytes, list, tuple, set, frozenset, dict,
)


def _default_to_source(value: Any) -> str:
    """Render a default value as stub source, or ``...`` when it is not a literal.

    Only values whose ``repr()`` evaluates back to an equal value survive;
    everything else (``default_factory`` sentinels, ``FieldInfo``,
    ``WiredField`` descriptors, arbitrary instances, enums, ``inf``) becomes
    ``...`` -- the PEP 484 stub placeholder, which still marks the parameter
    as optional.
    """
    if value is None:
        return "None"
    if value is Ellipsis:
        return "..."
    # ``type(value) in`` (not isinstance): subclasses such as IntEnum may
    # have a repr that is not valid source.
    if type(value) in _LITERAL_TYPES:
        rendered = repr(value)
        try:
            if ast.literal_eval(rendered) == value:
                return rendered
        except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
            pass
    return "..."


def _join_sections(*sections: list[str]) -> list[str]:
    out: list[str] = []
    for section in sections:
        if not section:
            continue
        if out:
            out.append("")
        out.extend(section)
    return out


def _strip_wrappers(text: str) -> str:
    """Drop an outer ``Required[...]`` / ``NotRequired[...]`` / ``ReadOnly[...]``."""
    try:
        node = ast.parse(text, mode="eval").body
    except SyntaxError:  # pragma: no cover - text came from ast.unparse
        return text
    if isinstance(node, ast.Subscript) and _dotted_tail(node.value) in (
        "Required", "NotRequired", "ReadOnly",
    ):
        return ast.unparse(node.slice)
    return text


def _module_ctx(obj: Any, fallback: Mapping[str, Any]) -> _ann.AnnotationContext:
    module = sys.modules.get(getattr(obj, "__module__", "") or "")
    return _ann.AnnotationContext(globalns=vars(module) if module is not None else fallback)


def _scan_imports(globalns: Mapping[str, Any]) -> dict[str, _ImportRecord]:
    """``{local name: import}`` of a module's module-level imports (AST only)."""
    source, _path = _ann._module_source(globalns)
    if source is None:
        return {}
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return {}
    package = _ann._module_package(globalns)
    found: dict[str, _ImportRecord] = {}
    for node in _ann._module_level_statements(tree.body):
        if isinstance(node, ast.ImportFrom):
            module = _ann._absolute_module(node.module, node.level, package)
            if module is None:
                continue
            for alias in node.names:
                if alias.name != "*":
                    found.setdefault(alias.asname or alias.name, _ImportRecord(module, alias.name))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    found.setdefault(alias.asname, _ImportRecord(alias.name, None))
                else:
                    top = alias.name.split(".", 1)[0]
                    found.setdefault(top, _ImportRecord(top, None))
    return found


def _is_plain_import(record: _ann.TypeCheckingName) -> bool:
    """An unaliased ``import a`` / ``import a.b.c`` (it binds the top package)."""
    return (
        record.attr is None
        and record.module is not None
        and record.module.split(".", 1)[0] == record.name
    )


def _covering_import(records: Mapping[str, _ann.TypeCheckingName], parts: list[str]) -> str | None:
    """The longest plain ``import`` of ``parts[0]`` whose module prefixes the dotted *parts*."""
    best: str | None = None
    for record in records.values():
        if record.name != parts[0] or not _is_plain_import(record):
            continue
        module = record.module or ""
        names = module.split(".")
        if names == parts[: len(names)] and (best is None or len(names) > len(best.split("."))):
            best = module
    return best


def _is_plain_dotted(binding: _Binding) -> bool:
    """An unaliased ``import a.b.c`` binding (its preferred name is the dotted module)."""
    return (
        binding.kind == "import"
        and binding.module is not None
        and "." in binding.preferred
        and binding.preferred == binding.module
    )


def _binds_package(binding: _Binding, top: str) -> bool:
    """True when *binding* is an unaliased ``import top`` / ``import top.x`` (it binds ``top``)."""
    return (
        binding.kind == "import"
        and binding.module is not None
        and binding.name == binding.module
        and binding.module.split(".", 1)[0] == top
    )


def _is_class(obj: Any) -> bool:
    """``isinstance(obj, type)``, except parametrized builtins (``tuple[int, int]`` is
    an instance of ``type`` on Python 3.10)."""
    return isinstance(obj, type) and not isinstance(obj, types.GenericAlias)


def _walk_chain(module: Any, parts: list[str]) -> list[Any] | None:
    chain: list[Any] = []
    current = module
    for part in parts:
        try:
            current = getattr(current, part)
        except AttributeError:
            return None
        chain.append(current)
    return chain


def _qualname(obj: Any) -> str | None:
    """``__qualname__`` (else ``__name__``) when it is a plain dotted identifier."""
    for attr in ("__qualname__", "__name__"):
        try:
            value = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(value, str) and value and all(p.isidentifier() for p in value.split(".")):
            return value
    return None


def _is_builtin(obj: Any) -> bool:
    name = getattr(obj, "__name__", None)
    return (
        _is_class(obj)
        and getattr(obj, "__module__", None) == "builtins"
        and isinstance(name, str)
        and getattr(builtins, name, None) is obj
    )


def _is_named_typing_object(obj: Any) -> bool:
    """Special forms, type variables, ``NewType`` and type aliases."""
    if isinstance(obj, _TYPEVAR_TYPES + _PARAMSPEC_TYPES + _TYPEVARTUPLE_TYPES):
        return True
    if _NEWTYPE_TYPES and isinstance(obj, _NEWTYPE_TYPES):
        return True
    if _TYPEALIAS_TYPES and isinstance(obj, _TYPEALIAS_TYPES):
        return True
    return getattr(obj, "__module__", None) in ("typing", "typing_extensions") and (
        _qualname(obj) is not None
    )


def _definable(obj: Any) -> bool:
    return _is_class(obj) or _is_named_typing_object(obj)


def _is_typeddict(obj: Any) -> bool:
    if typing.is_typeddict(obj):
        return True
    return bool(_te is not None and _te.is_typeddict(obj))


def _typing_form(name: str) -> Any:
    form = getattr(typing, name, None)
    if form is None and _te is not None:  # pragma: no cover - 3.10
        form = getattr(_te, name)
    return form


def _typing_alias(tp: Any, origin: Any) -> Any:
    """The ``typing`` alias a subscripted ``typing.X[...]`` was spelled with.

    ``typing.AsyncIterator[str]`` has origin ``collections.abc.AsyncIterator``;
    rendering the head through ``typing.AsyncIterator`` keeps the spelling (and
    the import) the source used.
    """
    name = getattr(tp, "_name", None)
    if not isinstance(name, str) or getattr(tp, "__module__", None) != "typing":
        return origin
    special = getattr(typing, name, None)
    if special is None or special is origin or get_origin(special) is not origin:
        return origin
    return special


def _pydantic_generic(tp: Any) -> tuple[Any, tuple[Any, ...]] | None:
    """``(origin, args)`` of a parametrized pydantic generic model class."""
    if not _is_class(tp):
        return None
    meta = tp.__dict__.get("__pydantic_generic_metadata__")
    if not isinstance(meta, dict):
        return None
    origin = meta.get("origin")
    args = meta.get("args") or ()
    if origin is None or not args:
        return None
    return origin, tuple(args)


def _own_bases(cls: type) -> tuple[Any, ...]:
    """The bases as written (``__orig_bases__`` keeps generic arguments)."""
    return tuple(cls.__dict__.get("__orig_bases__") or cls.__bases__)


def _dataclass_transformed(cls: type) -> bool:
    """True when a ``dataclass_transform`` (e.g. pydantic's) applies to *cls*'s stub."""
    for klass in cls.__mro__[1:]:
        if "__dataclass_transform__" in klass.__dict__:
            return True
    return any("__dataclass_transform__" in meta.__dict__ for meta in inspect.getmro(type(cls)))


def _same_type_param(a: Any, b: Any) -> bool:
    """True when two type variables would be written identically."""
    if type(a) is not type(b) or getattr(a, "__name__", None) != getattr(b, "__name__", None):
        return False
    return all(
        getattr(a, attr, None) == getattr(b, attr, None)
        for attr in (
            "__bound__", "__constraints__", "__covariant__", "__contravariant__",
            "__infer_variance__",
        )
    )


def _is_sunder(name: str) -> bool:
    return len(name) > 2 and name.startswith("_") and name.endswith("_") and not name.startswith("__")


#: The name a class-body line of the stub binds (``x: T``, ``X = v``, ``def f(``, ``class C``).
_BODY_BINDING_RE = re.compile(r"(?:async\s+def|def|class)\s+([A-Za-z_]\w*)|([A-Za-z_]\w*)\s*[:=]")


def _body_scope(d: _Def) -> frozenset[str]:
    """Names the class body of *d* binds (its member lines and nested classes)."""
    names = set(d.children)
    for line in d.body or ():
        if not line or line.startswith(("@", " ", "#")):
            continue
        match = _BODY_BINDING_RE.match(line)
        if match:
            names.add(match.group(1) or match.group(2))
    return frozenset(names)


def _ordered_children(d: _Def) -> list[_Def]:
    """Nested definitions in the order the class body defines them."""
    names = list(vars(d.obj)) if _is_class(d.obj) else []
    position = {name: index for index, name in enumerate(names)}
    return sorted(d.children.values(), key=lambda child: position.get(child.name, len(position)))


def _all_deps(d: _Def) -> set[int]:
    deps = set(d.deps)
    for child in d.children.values():
        deps |= _all_deps(child)
    return deps


def _dotted_parts(node: ast.AST) -> list[str] | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return parts[::-1]


def _dotted_tail(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _describe(obj: Any) -> str:
    if _is_class(obj):
        return f"class {qualified_name(obj)}"
    return _safe_repr(obj)


def _safe_repr(obj: Any) -> str:
    try:
        return repr(obj)
    except Exception:  # noqa: BLE001 - message rendering must never fail
        return f"<{type(obj).__name__} object>"


def _version() -> str:
    from conscribe import __version__

    return __version__


def _class_shadowed_names(tree: ast.Module) -> set[str]:
    """Names a class body binds and also loads in an annotation, default or base.

    Annotations in a class body are evaluated in the class scope, so such a
    load means the class attribute, never the module-level object a
    generated stub refers to.  (Decorators may use class-level names:
    ``@size.setter``.)
    """
    found: set[str] = set()

    def visit(node: ast.ClassDef) -> None:
        bound: set[str] = set()
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(stmt.name)
            elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                bound.add(stmt.target.id)
            elif isinstance(stmt, ast.Assign):
                bound.update(t.id for t in stmt.targets if isinstance(t, ast.Name))
        exprs: list[ast.AST] = []
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                exprs.append(stmt.args)
                if stmt.returns is not None:
                    exprs.append(stmt.returns)
            elif isinstance(stmt, ast.AnnAssign):
                exprs.append(stmt.annotation)
                if stmt.value is not None:
                    exprs.append(stmt.value)
            elif isinstance(stmt, ast.ClassDef):
                exprs.extend(stmt.bases)
                visit(stmt)
        for expr in exprs:
            for sub in ast.walk(expr):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load) and sub.id in bound:
                    found.add(sub.id)

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            visit(node)
    return found


def _plain_import_binds(node: ast.Import, name: str) -> bool:
    """True when *node* binds *name* only through unaliased ``import name[.x]``."""
    return any(
        alias.asname is None and alias.name.split(".", 1)[0] == name for alias in node.names
    ) and not any(alias.asname == name for alias in node.names)


def _bound_names(node: ast.stmt) -> Iterator[str]:
    if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        yield node.name
    elif isinstance(node, ast.Import):
        for alias in node.names:
            yield alias.asname or alias.name.split(".", 1)[0]
    elif isinstance(node, ast.ImportFrom):
        if node.module != "__future__":
            for alias in node.names:
                yield alias.asname or alias.name
    elif isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                yield target.id
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        yield node.target.id


def _unbound_names(tree: ast.Module, module_bound: set[str]) -> set[str]:
    """Names the stub loads that are neither bound at module level nor builtins.

    Decorators in a class body may also use names bound earlier in that
    class body (``@x.setter``).
    """
    missing: set[str] = set()

    def check(node: ast.AST, scope: frozenset[str] = frozenset()) -> None:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                if sub.id not in scope and sub.id not in module_bound and sub.id not in _PY_BUILTINS:
                    missing.add(sub.id)

    def check_function(node: ast.FunctionDef | ast.AsyncFunctionDef, scope: frozenset[str]) -> None:
        for decorator in node.decorator_list:
            check(decorator, scope)
        check(node.args)
        if node.returns is not None:
            check(node.returns)

    def visit_class(node: ast.ClassDef) -> None:
        for expr in [*node.bases, *(k.value for k in node.keywords), *node.decorator_list]:
            check(expr)
        scope: set[str] = set()
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                check_function(stmt, frozenset(scope))
                scope.add(stmt.name)
            elif isinstance(stmt, ast.ClassDef):
                visit_class(stmt)
                scope.add(stmt.name)
            elif isinstance(stmt, ast.AnnAssign):
                check(stmt.annotation)
                if stmt.value is not None:
                    check(stmt.value)
                if isinstance(stmt.target, ast.Name):
                    scope.add(stmt.target.id)
            elif isinstance(stmt, ast.Assign):
                check(stmt.value)
                scope.update(t.id for t in stmt.targets if isinstance(t, ast.Name))
            else:
                check(stmt)

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            visit_class(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            check_function(node, frozenset())
        elif isinstance(node, ast.AnnAssign):
            check(node.annotation)
            if node.value is not None:
                check(node.value)
        elif isinstance(node, ast.Assign):
            check(node.value)
    return missing
