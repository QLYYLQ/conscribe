"""Render in-process config models as Python source.

This private module is the shared renderer behind ``codegen``.  It turns
what conscribe's in-process pydantic models carry — annotations,
``FieldInfo`` objects, defaults, default factories, ``Annotated`` metadata,
``model_config`` — into Python source text, and it owns the module-level
names of one generated file.

Design rules:

* **Faithful or loud.**  A default, factory or type that cannot be written so
  that the generated model equals the in-process one raises
  :class:`~conscribe.exceptions.ConfigRenderError` naming the model, the field
  and the value.  ``Annotated`` metadata that cannot be written (a validator
  wrapping a lambda, ``Predicate(lambda ...)``) keeps being enforced by the
  in-process model; the renderer drops it from the file and records a
  :class:`RenderNote` that the caller reports as an ``ExtractionWarning``.
* **Values go through a ladder**, every rung verified against the real
  objects: literal (``ast.literal_eval`` round trip) → Enum member →
  constructor form (``eval``; NaN-aware, see :func:`_same_value`) → JSON form
  + ``validate_default`` (field defaults only; the literal must validate back
  to the default under the whole field, see :func:`_json_form`) → importable
  reference found by identity → error.
* **Names are assigned last.**  Every reference to a module-level object (an
  imported user type, a typing/pydantic helper, a builtin, a class defined in
  the generated file) is written as a placeholder claimed in a
  :class:`ModuleNamespace`.  Once the whole file is rendered the namespace
  assigns local names — aliasing colliding names deterministically and
  independently of rendering order — and substitutes them.
* **Define vs import.**  Only conscribe-generated models are defined in the
  file (:func:`is_generated_model`); every other referenced type is imported
  by identity.  Imports from ``conscribe.*`` and from function locals are
  refused; ``__main__`` imports are allowed with a warning.
* **Validators.**  An emitted class subclasses the first user model in its
  MRO (inheriting that model's validators) and flattens conscribe's own
  classes into its body; validators those classes declare are rendered by
  the layout (``render_model_class(extra_lines=..., rendered_decorators=...)``,
  e.g. the composed top level's inline-only refusal) or else reported as a
  :class:`RenderNote` (:func:`flattened_decorators`).

Required fields render as ``name: T`` (or ``name: T = Field(<kwargs>)``),
defaults always use the keyword form ``Field(default=...)`` when other
keywords are present — PEP 681 type checkers ignore positional defaults.
"""
from __future__ import annotations

import ast
import builtins
import collections.abc
import dataclasses
import datetime
import decimal
import enum
import inspect
import json
import keyword
import math
import pathlib
import re
import sys
import types
import typing
import warnings
from collections.abc import Callable, Hashable, Iterable, Mapping
from typing import Annotated, Any, Literal, Union, get_args, get_origin

import annotated_types
import pydantic
import pydantic_core
from pydantic import BaseModel, Discriminator, Field, Tag, TypeAdapter
from pydantic.fields import FieldInfo
from pydantic.json_schema import SkipJsonSchema
from pydantic_core import PydanticUndefined

from conscribe.exceptions import ConfigRenderError

__all__ = [
    "ModuleNamespace",
    "RenderContext",
    "RenderNote",
    "declared_fields",
    "describe",
    "flattened_decorators",
    "generated_dependencies",
    "is_generated_model",
    "model_base",
    "render_field",
    "render_model_class",
    "render_type",
    "render_value",
    "split_discriminated",
]

# ``get_origin(int | None)`` is ``types.UnionType`` on 3.10–3.13 and
# ``typing.Union`` on 3.14+; both spellings must be recognised.
_UNION_ORIGINS: tuple[Any, ...] = (Union, types.UnionType)

# Keywords ``pydantic.Field`` accepts on the running pydantic version
# (``fail_fast`` exists from 2.8, ``exclude_if`` from 2.11, ...).  The
# deprecated ``**extra`` catch-all is never rendered.
_FIELD_PARAMS: frozenset[str] = frozenset(inspect.signature(Field).parameters) - {"extra"}

# annotated_types constraints that map 1:1 onto a ``Field`` keyword.
_SIMPLE_CONSTRAINTS: dict[type, str] = {
    annotated_types.Gt: "gt",
    annotated_types.Ge: "ge",
    annotated_types.Lt: "lt",
    annotated_types.Le: "le",
    annotated_types.MultipleOf: "multiple_of",
    annotated_types.MinLen: "min_length",
    annotated_types.MaxLen: "max_length",
}

# pydantic marker classes whose single attribute is a ``Field`` keyword.
_PYDANTIC_FLAG_CONSTRAINTS: dict[str, str] = {
    "Strict": "strict",
    "AllowInfNan": "allow_inf_nan",
    "FailFast": "fail_fast",
}

# The class attribute conscribe sets in the own ``__dict__`` of every model it
# builds (``conscribe.config.builder.GENERATED_MARKER``).
_GENERATED_MARKER = "__conscribe_generated__"

# Module prefix of every module that builds config models.  The "generated"
# fallback for models without the marker (the extractor's helper bases).
_GENERATED_MODULE = "conscribe.config"

# Sentinels that must never be written as a default value.
_SENTINEL_NAMES = {
    id(dataclasses.MISSING): "dataclasses.MISSING",
    id(inspect.Parameter.empty): "inspect.Parameter.empty",
    id(PydanticUndefined): "PydanticUndefined",
}
_HAS_DEFAULT_FACTORY = getattr(dataclasses, "_HAS_DEFAULT_FACTORY", None)
if _HAS_DEFAULT_FACTORY is not None:
    _SENTINEL_NAMES[id(_HAS_DEFAULT_FACTORY)] = "dataclasses <factory>"

# Placeholders are delimited by two private-use characters.  ``repr()`` of a
# string escapes them (they are not printable), so rendered literals can never
# contain a raw placeholder; comments are sanitised by ``_comment``.
_PH_OPEN = "\ue000"
_PH_CLOSE = "\ue001"
_PLACEHOLDER = re.compile(_PH_OPEN + r"(\d+)" + _PH_CLOSE)
_BUILTIN_NAMES: frozenset[str] = frozenset(dir(builtins))
_KIND_PRIORITY = {"defined": 0, "helper": 1, "builtin": 2, "import": 3, "module": 4}


_NO_SUBJECT = object()


class _Unrenderable(Exception):
    """Internal signal: the renderer has no faithful source for an object.

    ``subject`` is the object that failed, when known; the public entry points
    convert this into a ``ConfigRenderError`` (or, for metadata, a note).
    """

    def __init__(self, detail: str, subject: Any = _NO_SUBJECT) -> None:
        super().__init__(detail)
        self.detail = detail
        self.subject = subject


# ---------------------------------------------------------------------------
# Generated-model predicate
# ---------------------------------------------------------------------------


def is_generated_model(tp: Any) -> bool:
    """Whether ``tp`` is a pydantic model conscribe built (and so must define).

    Uses the ``__conscribe_generated__ = True`` marker in the class's *own*
    ``__dict__`` (``builder.GENERATED_MARKER``): a user subclass of a
    generated model inherits the attribute but is not generated -- it is
    imported, with its validators.  Falls back to "defined in a
    ``conscribe.config`` module", which covers every model the builders
    create with ``create_model`` (per-key models, nested segment submodels,
    composed rebuilds, the top-level ``ComposedConfig``) and the extractor's
    private ``_<Name>Base`` helper bases.
    """
    if not isinstance(tp, type) or tp is BaseModel:
        return False
    try:
        if not issubclass(tp, BaseModel):
            return False
    except TypeError:
        return False
    if vars(tp).get(_GENERATED_MARKER) is True:
        return True
    module = getattr(tp, "__module__", "") or ""
    return module == _GENERATED_MODULE or module.startswith(_GENERATED_MODULE + ".")


# ---------------------------------------------------------------------------
# Stable descriptions (never memory addresses)
# ---------------------------------------------------------------------------


def describe(obj: Any, _depth: int = 0) -> str:
    """Short, deterministic description of ``obj`` for messages and comments."""
    if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
        text = repr(obj)
        return text if len(text) <= 80 else text[:77] + "..."
    if isinstance(obj, enum.Enum):
        return f"{type(obj).__qualname__}.{obj.name}"
    if isinstance(obj, (type, types.FunctionType, types.BuiltinFunctionType, types.MethodType)):
        module = getattr(obj, "__module__", None) or "?"
        qualname = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", "?")
        return f"{module}.{qualname}"
    if isinstance(obj, FieldInfo):
        return "Field(...)"
    if _depth < 2 and dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        parts = []
        for f in dataclasses.fields(obj):
            if not f.repr:
                continue
            value = getattr(obj, f.name, None)
            parts.append(f"{f.name}={describe(value, _depth + 1)}")
        return f"{type(obj).__qualname__}({', '.join(parts)})"
    if isinstance(obj, (list, tuple, set, frozenset, dict)):
        return f"{type(obj).__name__} of {len(obj)} item(s)"
    text = repr(obj)
    if " at 0x" in text or len(text) > 80:
        return f"<{type(obj).__qualname__} object>"
    return text


def _comment(text: str) -> str:
    """Make ``text`` safe for a single-line ``#`` comment."""
    text = text.replace("\r", " ").replace("\n", " ")
    return text.replace(_PH_OPEN, "").replace(_PH_CLOSE, "")


# ---------------------------------------------------------------------------
# Module namespace
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Claim:
    kind: str  # "defined" | "helper" | "builtin" | "import" | "module"
    module: str
    attr: str
    obj: Any
    local: str | None = None


class ModuleNamespace:
    """Module-level names of one generated file.

    Rendering code never writes a module-level name directly: it asks the
    namespace for a placeholder (:meth:`define`, :meth:`helper`,
    :meth:`builtin`, :meth:`import_name`, :meth:`module`).  Field names are
    reported with :meth:`forbid` — inside a class body such a name would
    shadow the module-level binding, so no imported name may use it.  Names
    that a user base class's string annotations use are reported with
    :meth:`pin` — the file may bind them only to the object the base's own
    module binds.  :meth:`finalize` then assigns every local name and
    :meth:`materialize` substitutes them.

    Collisions are resolved by priority — names defined in the file, then
    typing/pydantic helpers, then builtins, then imported objects; among
    imports, the alphabetically-first module keeps the plain name — and the
    losers are aliased as ``<Name>__<module_path>``.  Builtins that collide
    are written ``builtins.<name>``.
    """

    def __init__(self) -> None:
        self._claims: list[_Claim] = []
        self._index: dict[tuple[str, str, str], int] = {}
        self._forbidden: dict[str, str] = {}  # field name -> "Model.field" declaring it
        self._assigned: set[str] = set()  # field names whose statement assigns a value
        #: name -> (object it must denote, "where" for messages)
        self._pinned: dict[str, tuple[Any, str]] = {}
        self._final = False

    # -- claiming ----------------------------------------------------------

    def _claim(self, kind: str, module: str, attr: str, obj: Any) -> str:
        space = kind if kind in ("defined", "builtin", "module") else "import"
        key = (space, module, attr)
        idx = self._index.get(key)
        if idx is None:
            if self._final:
                raise RuntimeError(f"name {module}.{attr} claimed after finalize()")
            idx = len(self._claims)
            self._claims.append(_Claim(kind, module, attr, obj))
            self._index[key] = idx
        elif kind == "helper" and self._claims[idx].kind == "import":
            self._claims[idx].kind = "helper"
        return f"{_PH_OPEN}{idx}{_PH_CLOSE}"

    def define(self, name: str, obj: Any) -> str:
        """Placeholder for ``name``, defined at module level in this file.

        Raises:
            ConfigRenderError: ``name`` is not an identifier (or a keyword).
        """
        if not _is_field_name(name):
            why = "a Python keyword" if keyword.iskeyword(name) else "not a valid Python identifier"
            raise ConfigRenderError(
                name, "(class)", repr(name),
                f"the generated module-level name {name!r} is {why}",
            )
        idx = self._index.get(("defined", "", name))
        if idx is not None and self._claims[idx].obj is not obj:
            raise RuntimeError(f"generated name {name!r} defined twice")
        return self._claim("defined", "", name, obj)

    def helper(self, module: str, name: str) -> str:
        """Placeholder for a typing/pydantic helper such as ``Field``."""
        obj = getattr(sys.modules[module], name)
        return self._claim("helper", module, name, obj)

    def builtin(self, name: str) -> str:
        """Placeholder for the builtin ``name`` (``list``, ``float``, ...)."""
        return self._claim("builtin", "builtins", name, getattr(builtins, name))

    def import_name(self, module: str, attr: str, obj: Any) -> str:
        """Placeholder for ``from <module> import <attr>`` (``obj`` is the value)."""
        return self._claim("import", module, attr, obj)

    def module(self, name: str) -> str:
        """Placeholder for the top-level module ``name`` (``import <name>``)."""
        if "." in name:
            raise ValueError(f"module() takes a top-level module name, not {name!r}")
        return self._claim("module", name, name, sys.modules[name])

    def forbid(self, name: str, where: str = "", *, assigned: bool = True) -> None:
        """No imported, helper or builtin name used in the file may be ``name``.

        Args:
            name: A field name.
            where: ``"Model.field"``, for error messages.
            assigned: The field statement assigns a value.  Such a name
                shadows the module-level binding in the class body at
                runtime, so a *definition* of the file under that name is an
                error; an annotation-only field only confuses type checkers,
                which imports avoid by aliasing.
        """
        self._forbidden.setdefault(name, where or name)
        if assigned:
            self._assigned.add(name)

    def pin(self, name: str, obj: Any, where: str) -> None:
        """The file may bind ``name`` only to ``obj`` (or leave it unbound).

        pydantic < 2.9 evaluates the string annotations of a user base class
        in the namespace of the module that subclasses it: when the generated
        file binds such a name to another object (``from datetime import
        datetime`` while the base's module did ``import datetime`` for
        ``"datetime.datetime"``), the evaluation fails and the file does not
        import.  An import, helper or builtin under a pinned name that is not
        that object is aliased instead.  Names the file defines itself keep
        their names: pydantic only uses inherited fields' hints to detect
        ``ClassVar``, so a hint resolving to a generated class is harmless.

        Args:
            name: A name the base's string annotations use.
            obj: What the base's module binds it to (:data:`_UNBOUND` when
                nothing).
            where: The base class, for error messages.
        """
        self._pinned.setdefault(name, (obj, where))

    def _blocked(self, name: str, obj: Any) -> bool:
        """``name`` may not be bound to ``obj`` (a field name, or pinned elsewhere)."""
        if name in self._forbidden:
            return True
        pinned = self._pinned.get(name)
        return pinned is not None and pinned[0] is not obj

    # -- finalizing ------------------------------------------------------------

    def finalize(self) -> None:
        """Assign every claimed name.  Idempotent."""
        if self._final:
            return
        taken: set[str] = set()
        for claim in self._claims:
            if claim.kind != "defined":
                continue
            if claim.attr in self._assigned:
                model, _, field = self._forbidden[claim.attr].rpartition(".")
                raise ConfigRenderError(
                    model or "<module>", field, repr(claim.attr),
                    f"the field name equals the generated definition {claim.attr!r}, "
                    f"which the field's class body would shadow; rename the field",
                )
            claim.local = claim.attr
            taken.add(claim.attr)

        if any(
            c.kind == "builtin" and (self._blocked(c.attr, c.obj) or c.attr in taken)
            for c in self._claims
        ):
            self._claim("module", "builtins", "builtins", builtins)
        self._final = True

        groups: dict[str, list[_Claim]] = {}
        for claim in self._claims:
            if claim.kind != "defined":
                groups.setdefault(claim.attr, []).append(claim)

        deferred: list[_Claim] = []
        for name in sorted(groups):
            members = sorted(
                groups[name], key=lambda c: (_KIND_PRIORITY[c.kind], c.module)
            )
            for claim in members:
                if claim.kind == "builtin":
                    if not self._blocked(name, claim.obj) and name not in taken:
                        claim.local = name
                        taken.add(name)
                    continue
                eligible = (
                    name not in taken
                    and not self._blocked(name, claim.obj)
                    and name not in _BUILTIN_NAMES
                    and not keyword.iskeyword(name)
                )
                if eligible:
                    claim.local = name
                    taken.add(name)
                else:
                    deferred.append(claim)

        for claim in sorted(deferred, key=lambda c: (c.attr, c.module)):
            tag = re.sub(r"\W+", "_", claim.module.strip("_")).strip("_") or "mod"
            base = f"{claim.attr}__{tag}"
            alias, n = base, 2
            while (
                alias in taken or alias in self._forbidden or alias in self._pinned
                or alias in _BUILTIN_NAMES
            ):
                alias, n = f"{base}_{n}", n + 1
            claim.local = alias
            taken.add(alias)

        module_local = None
        idx = self._index.get(("module", "builtins", "builtins"))
        if idx is not None:
            module_local = self._claims[idx].local
        for claim in self._claims:
            if claim.kind == "builtin" and claim.local is None:
                claim.local = f"{module_local}.{claim.attr}"

    def materialize(self, text: str) -> str:
        """Substitute final names for every placeholder in ``text``."""
        self.finalize()
        return _PLACEHOLDER.sub(lambda m: self._claims[int(m.group(1))].local or "", text)

    def materialize_qualified(self, text: str) -> str:
        """Substitute ``kind:module:name`` for every placeholder (signatures).

        Unlike :meth:`materialize` this needs no :meth:`finalize` and is
        independent of aliasing, so two texts are equal exactly when they
        reference the same objects.
        """

        def repl(match: re.Match[str]) -> str:
            claim = self._claims[int(match.group(1))]
            return f"<{claim.kind}:{claim.module}:{claim.attr}>"

        return _PLACEHOLDER.sub(repl, text)

    def readable(self, text: str) -> str:
        """``text`` with every placeholder replaced by its plain name (messages)."""
        return _PLACEHOLDER.sub(lambda m: self._claims[int(m.group(1))].attr, text)

    def evaluate(self, text: str) -> Any:
        """``eval`` rendered source against the real objects (verification)."""
        names: dict[str, Any] = {}

        def repl(match: re.Match[str]) -> str:
            idx = int(match.group(1))
            local = f"__cs_ref_{idx}"
            names[local] = self._claims[idx].obj
            return local

        code = _PLACEHOLDER.sub(repl, text)
        return eval(code, {"__builtins__": builtins, **names})  # noqa: S307

    def import_lines(self) -> list[str]:
        """The import block: typing, a blank line, pydantic, other modules."""
        self.finalize()
        by_module: dict[str, list[_Claim]] = {}
        module_imports: list[_Claim] = []
        for claim in self._claims:
            if claim.kind in ("helper", "import"):
                by_module.setdefault(claim.module, []).append(claim)
            elif claim.kind == "module":
                module_imports.append(claim)

        def line(module: str) -> str:
            names = []
            for claim in sorted(by_module[module], key=lambda c: c.attr):
                if claim.local == claim.attr:
                    names.append(claim.attr)
                else:
                    names.append(f"{claim.attr} as {claim.local}")
            return f"from {module} import {', '.join(names)}"

        lines: list[str] = []
        if "typing" in by_module:
            lines.append(line("typing"))
            lines.append("")
        if "pydantic" in by_module:
            lines.append(line("pydantic"))
        for module in sorted(by_module):
            if module not in ("typing", "pydantic"):
                lines.append(line(module))
        for claim in sorted(module_imports, key=lambda c: c.attr):
            if claim.local == claim.attr:
                lines.append(f"import {claim.attr}")
            else:
                lines.append(f"import {claim.attr} as {claim.local}")
        return lines


# ---------------------------------------------------------------------------
# Render context
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class RenderNote:
    """Metadata the generated file cannot carry (the in-process model still does).

    Attributes:
        model: Name of the in-process model.
        field: Field name (``"model_config"`` for a config key).
        what: Description of the dropped item, e.g. ``AfterValidator(...)``.
    """

    model: str
    field: str
    what: str

    def message(self) -> str:
        """The ``ExtractionWarning`` text for this note."""
        return (
            f"{self.model}.{self.field}: {self.what} cannot be written into the "
            f"generated source; the in-process config model keeps enforcing it, "
            f"the generated file does not"
        )


#: ``renderer(model, name, decorator, ctx) -> class-body lines`` reproducing a
#: validator that conscribe attaches to generated models (see
#: :meth:`RenderContext.register_validator`).
ValidatorRenderer = Callable[[type[BaseModel], str, Any, "RenderContext"], list[str]]


class RenderContext:
    """State shared while rendering one generated file.

    Holds the :class:`ModuleNamespace`, the table of objects the file
    *defines* (generated models, discriminator functions, union aliases), the
    notes and warnings collected so far, and the current location (model and
    field) used in error messages.
    """

    def __init__(self, namespace: ModuleNamespace | None = None) -> None:
        self.ns = namespace if namespace is not None else ModuleNamespace()
        self._models: dict[int, tuple[type, str]] = {}
        self._functions: dict[int, tuple[Any, str]] = {}
        self._aliases: dict[Hashable, str] = {}
        self._validator_renderers: dict[int, tuple[Any, ValidatorRenderer]] = {}
        #: Module-level helper blocks requested while rendering (name -> source).
        self.module_blocks: dict[str, str] = {}
        self.notes: list[RenderNote] = []
        self.warnings: list[str] = []
        #: The in-process model whose class is being rendered (its config and
        #: validators decide how defaults may be written), if any.
        self.model: type[BaseModel] | None = None
        self.model_name = ""
        self.field_name = ""

    # -- conscribe's own validators -------------------------------------------

    def register_validator(self, fn: Any, renderer: ValidatorRenderer) -> None:
        """Reproduce the validator function ``fn`` with ``renderer``.

        ``fn`` is a function conscribe attaches to generated models as a
        validator (by identity); ``renderer(model, name, decorator, ctx)``
        returns the class-body lines that reproduce the validator ``name``
        of ``model`` (unindented; placeholders allowed).
        """
        self._validator_renderers[id(fn)] = (fn, renderer)

    def validator_renderer(self, decorator: Any) -> ValidatorRenderer | None:
        """The registered renderer of a pydantic decorator record, if any."""
        func = getattr(decorator, "func", None)
        func = getattr(func, "__func__", func)  # classmethods are recorded bound
        entry = self._validator_renderers.get(id(func))
        return entry[1] if entry is not None and entry[0] is func else None

    def module_function(self, name: str, obj: Any, source: Callable[[str], str]) -> str:
        """Placeholder of a module-level helper the file defines once.

        Args:
            name: The helper's name in the file.
            obj: The in-process object it reproduces (claims the name).
            source: ``source(placeholder) -> block``; called on first use.
        """
        placeholder = self.ns.define(name, obj)
        if name not in self.module_blocks:
            self.module_blocks[name] = source(placeholder)
        return placeholder

    # -- definitions -------------------------------------------------------

    def define_model(self, model: type, name: str) -> str:
        """Record that ``model`` is defined in the file as ``name``."""
        placeholder = self.ns.define(name, model)
        self._models[id(model)] = (model, placeholder)
        return placeholder

    def define_function(self, fn: Any, name: str) -> str:
        """Record that ``fn`` is emitted as the module-level function ``name``.

        Args:
            fn: The in-process function the emitted one reproduces (a nested
                layer's compound discriminator: one cached function per
                layer, shared by every union of its models).
            name: Its module-level name in the file.
        """
        placeholder = self.ns.define(name, fn)
        self._functions[id(fn)] = (fn, placeholder)
        return placeholder

    def define_function_as(self, fn: Any, placeholder: str) -> None:
        """Render references to ``fn`` as the already-emitted function ``placeholder``.

        For a second function object with the same name, discriminator fields
        and separator as an emitted one (the same logic): the file emits it
        once.
        """
        self._functions[id(fn)] = (fn, placeholder)

    def define_alias(self, union_type: Any, name: str) -> str:
        """Record the module-level alias ``name = <union_type>``.

        Every later occurrence of the same discriminated union (same members,
        same discriminator) renders as ``name``.
        """
        placeholder = self.ns.define(name, union_type)
        base, disc, rest = split_discriminated(union_type)
        if base is not None and not rest:
            key = self._union_key(base, disc)
            if key is not None:
                self._aliases[key] = placeholder
        return placeholder

    def is_defined_model(self, model: Any) -> bool:
        """Whether ``model`` is defined in this file."""
        return id(model) in self._models

    def model_ref(self, model: type) -> str:
        """Placeholder for a model defined in this file.

        Raises:
            _Unrenderable: ``model`` is not defined in this file.
        """
        entry = self._models.get(id(model))
        if entry is None:
            raise _Unrenderable(
                f"generated model {model.__qualname__} is referenced but not "
                f"emitted (conscribe bug: it was missed when collecting submodels)"
            )
        return entry[1]

    def defined_ref(self, model: type, *, where: str = "(class)") -> str:
        """:meth:`model_ref` for layout code: a missing model is a ``ConfigRenderError``.

        Args:
            model: A model the file must define.
            where: The location named in the error (field / block label).

        Raises:
            ConfigRenderError: ``model`` is not defined in this file.
        """
        try:
            return self.model_ref(model)
        except _Unrenderable as exc:
            raise ConfigRenderError(
                getattr(model, "__name__", "<model>"), where, describe(model), exc.detail,
            ) from None

    def function_token(self, fn: Any) -> str | None:
        """Placeholder of the emitted function ``fn`` (by identity), if any.

        Every union of a nested layer's models -- the layer union, composed
        subset unions, rebuilt unions -- carries the layer's one cached
        discriminator function, so identity is the whole match.
        """
        entry = self._functions.get(id(fn))
        if entry is not None and entry[0] is fn:
            return entry[1]
        return None

    def alias_for(self, base: Any, disc: Any) -> str | None:
        """Placeholder of the alias for ``Annotated[base, <disc>]``, if any."""
        key = self._union_key(base, disc)
        if key is None:
            return None
        return self._aliases.get(key)

    def _union_key(self, base: Any, disc: Any) -> Hashable | None:
        if get_origin(base) not in _UNION_ORIGINS:
            return None
        members = get_args(base)
        if isinstance(disc, str):
            if not all(isinstance(m, type) for m in members):
                return None
            return ("flat", disc, frozenset(id(m) for m in members))
        if isinstance(disc, Discriminator):
            if (
                disc.custom_error_type is not None
                or disc.custom_error_message is not None
                or disc.custom_error_context is not None
            ):
                return None
            token = self.function_token(disc.discriminator)
            if token is None:
                return None
            tagged = []
            for member in members:
                if get_origin(member) is not Annotated:
                    return None
                inner, *meta = get_args(member)
                if len(meta) != 1 or not isinstance(meta[0], Tag):
                    return None
                tagged.append((id(inner), meta[0].tag))
            return ("callable", token, frozenset(tagged))
        return None

    # -- references ----------------------------------------------------------

    def ref(self, obj: Any) -> str:
        """Source for a reference to ``obj`` (class, function, constant).

        Generated models and emitted functions resolve to their definitions;
        builtins to their names; everything else is imported by identity.

        Raises:
            _Unrenderable: ``obj`` cannot be referenced from the generated file.
        """
        if is_generated_model(obj):
            return self.model_ref(obj)
        entry = self._functions.get(id(obj))
        if entry is not None:
            return entry[1]
        token = self.function_token(obj) if callable(obj) else None
        if token is not None:
            return token
        return _import_reference(obj, self)

    def error(self, value: Any, detail: str) -> ConfigRenderError:
        """A ``ConfigRenderError`` for the current location.

        Placeholders in ``detail`` (rendered text) are spelled out by name.
        """
        return ConfigRenderError(
            self.model_name or "<module>",
            self.field_name or "<model>",
            describe(value),
            self.ns.readable(detail),
        )

    def fail(self, exc: _Unrenderable, fallback: Any) -> ConfigRenderError:
        """Convert an internal ``_Unrenderable`` into a ``ConfigRenderError``."""
        subject = fallback if exc.subject is _NO_SUBJECT else exc.subject
        return self.error(subject, exc.detail)

    def note(self, what: str) -> None:
        """Record dropped metadata at the current location."""
        note = RenderNote(self.model_name or "<module>", self.field_name or "<model>", what)
        if note not in self.notes:
            self.notes.append(note)

    def warn(self, message: str) -> None:
        """Record an ``ExtractionWarning`` message (deduplicated)."""
        if message not in self.warnings:
            self.warnings.append(message)


# ---------------------------------------------------------------------------
# Import references (by identity)
# ---------------------------------------------------------------------------


def _resolve_path(root: Any, path: str) -> Any:
    obj = root
    for part in path.split("."):
        obj = getattr(obj, part)
    return obj


def _same(found: Any, obj: Any) -> bool:
    if found is obj:
        return True
    # Bound methods (classmethods) are re-created on every attribute access.
    return isinstance(obj, types.MethodType) and found == obj


def _public_module(module: str, attr: str, obj: Any) -> str:
    """Prefer a public re-export when ``module`` has a private segment."""
    parts = module.split(".")
    if not any(p.startswith("_") for p in parts) and parts[0] not in ("pydantic", "pydantic_core"):
        return module
    best = module
    for i in range(len(parts) - 1, 0, -1):
        candidate = ".".join(parts[:i])
        mod = sys.modules.get(candidate)
        if mod is not None and getattr(mod, attr, None) is obj:
            best = candidate
    return best


def _import_reference(obj: Any, ctx: RenderContext) -> str:
    module = getattr(obj, "__module__", None)
    qualname = getattr(obj, "__qualname__", None) or getattr(obj, "__name__", None)

    # Method descriptors of builtin types (``str.lower``) carry no module.
    objclass = getattr(obj, "__objclass__", None)
    if module is None and objclass is not None and isinstance(getattr(obj, "__name__", None), str):
        if getattr(objclass, obj.__name__, None) is obj:
            return f"{ctx.ref(objclass)}.{obj.__name__}"

    # Bound classmethods: reference through the class.
    if isinstance(obj, types.MethodType) and isinstance(obj.__self__, type):
        cls = obj.__self__
        name = obj.__func__.__name__
        if getattr(cls, name, None) == obj:
            return f"{ctx.ref(cls)}.{name}"

    if isinstance(module, str) and isinstance(qualname, str) and qualname:
        if "<locals>" in qualname or "<lambda>" in qualname:
            raise _Unrenderable(
                f"{describe(obj)} is defined inside a function or is a lambda; the "
                f"generated file can only import module-level objects — move it to "
                f"module level",
                obj,
            )
        if module == "builtins":
            if getattr(builtins, qualname, None) is obj:
                return ctx.ns.builtin(qualname)
            raise _Unrenderable(f"{describe(obj)} is not importable from builtins", obj)
        _check_module_allowed(module, obj, ctx)
        mod = sys.modules.get(module)
        if mod is None:
            raise _Unrenderable(f"{describe(obj)}: module {module!r} is not imported", obj)
        try:
            found = _resolve_path(mod, qualname)
        except AttributeError:
            found = None
        first, _, rest = qualname.partition(".")
        if found is not None and _same(found, obj) and _is_field_name(first):
            top = _resolve_path(mod, first)
            source_module = _public_module(module, first, top)
            text = ctx.ns.import_name(source_module, first, top)
            return text + (f".{rest}" if rest else "")
        if isinstance(obj, (type, types.FunctionType)):
            raise _Unrenderable(
                f"{describe(obj)} cannot be imported by its qualified name "
                f"({module}.{qualname} is a different object)",
                obj,
            )

    # Module-level constants and sentinels: search loaded modules by identity.
    found_at = _find_module_attribute(obj)
    if found_at is None:
        raise _Unrenderable(
            f"{describe(obj)} has no literal, constructor or JSON form and is not "
            f"a module-level object the generated file could import",
            obj,
        )
    module, attr = found_at
    _check_module_allowed(module, obj, ctx)
    return ctx.ns.import_name(module, attr, obj)



def _check_module_allowed(module: str, obj: Any, ctx: RenderContext) -> None:
    if module == "conscribe" or module.startswith("conscribe."):
        raise _Unrenderable(
            f"{describe(obj)} lives in conscribe's own module {module!r}; generated "
            f"files never import conscribe internals",
            obj,
        )
    if module == "__main__":
        ctx.warn(
            f"{describe(obj)} is defined in __main__; the generated file imports it "
            f"with 'from __main__ import ...', which only works while that script is "
            f"the running program — move it into an importable module"
        )


def _find_module_attribute(obj: Any) -> tuple[str, str] | None:
    """Find a module-level binding of ``obj`` (by identity) among loaded modules.

    The module that defines ``type(obj)`` is searched first; then every
    loaded module in alphabetical order (``__main__`` last).  Within a module
    the alphabetically-first name wins, so the result is deterministic.
    """
    preferred = getattr(type(obj), "__module__", None)
    names = sorted(m for m in list(sys.modules) if m != "__main__")
    order = ([preferred] if preferred and preferred != "builtins" else []) + names + ["__main__"]
    for module_name in order:
        if module_name == "conscribe" or module_name.startswith("conscribe."):
            continue
        module = sys.modules.get(module_name)
        namespace = getattr(module, "__dict__", None)
        if not isinstance(namespace, dict):
            continue
        hits = [
            k for k, v in list(namespace.items())
            if v is obj and not k.startswith("__") and _is_field_name(k)
        ]
        if hits:
            return module_name, min(hits)
    return None


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


def render_type(tp: Any, ctx: RenderContext) -> str:
    """Source for the type expression ``tp``.

    Raises:
        ConfigRenderError: ``tp`` (or something inside it) cannot be written.
    """
    try:
        return _type(tp, ctx)
    except _Unrenderable as exc:
        raise ctx.fail(exc, tp) from None


def _type(tp: Any, ctx: RenderContext) -> str:
    if tp is None or tp is type(None):
        return "None"
    if tp is Any:
        return ctx.ns.helper("typing", "Any")
    if tp is Ellipsis:
        return "..."
    if isinstance(tp, (str, typing.ForwardRef)):
        text = tp if isinstance(tp, str) else tp.__forward_arg__
        raise _Unrenderable(f"unresolved forward reference {text!r}", tp)
    if isinstance(tp, typing.TypeVar):
        raise _Unrenderable(f"unbound type variable {tp!r}", tp)

    origin = get_origin(tp)
    if origin is Annotated:
        return _annotated(tp, ctx)
    if origin is Literal:
        values = ", ".join(_value(v, ctx) for v in get_args(tp))
        return f"{ctx.ns.helper('typing', 'Literal')}[{values}]"
    if origin in _UNION_ORIGINS:
        args = get_args(tp)
        if len(args) == 2 and args[1] is type(None):
            return f"{ctx.ns.helper('typing', 'Optional')}[{_type(args[0], ctx)}]"
        return f"{ctx.ns.helper('typing', 'Union')}[{', '.join(_type(a, ctx) for a in args)}]"
    if origin is tuple:
        if not hasattr(tp, "__args__"):
            return ctx.ns.builtin("tuple")  # bare typing.Tuple
        args = get_args(tp)
        if args in ((), ((),)):
            return f"{ctx.ns.builtin('tuple')}[()]"
        return f"{ctx.ns.builtin('tuple')}[{', '.join(_type(a, ctx) for a in args)}]"
    if origin is collections.abc.Callable:
        args = get_args(tp)
        head = _class_ref(collections.abc.Callable, ctx)
        if not args:
            return head
        params, result = args[0], args[-1]
        if params is Ellipsis:
            params_src = "..."
        else:
            params_src = "[" + ", ".join(_type(a, ctx) for a in params) + "]"
        return f"{head}[{params_src}, {_type(result, ctx)}]"
    if origin is not None:
        head = _class_ref(origin, ctx)
        if not hasattr(tp, "__args__"):
            return head  # bare typing.List / typing.Dict
        args = get_args(tp)
        if not args:
            return head
        return f"{head}[{', '.join(_type(a, ctx) for a in args)}]"

    if isinstance(tp, type):
        return _class_ref(tp, ctx)

    special = getattr(tp, "_name", None)
    if isinstance(special, str) and getattr(typing, special, None) is tp:
        return ctx.ns.helper("typing", special)
    # NewType, TypeAliasType and other importable type-like objects.
    return _import_reference(tp, ctx)


def _class_ref(cls: Any, ctx: RenderContext) -> str:
    if is_generated_model(cls):
        return ctx.model_ref(cls)
    if getattr(builtins, getattr(cls, "__name__", ""), None) is cls:
        return ctx.ns.builtin(cls.__name__)
    generic = getattr(cls, "__pydantic_generic_metadata__", None) if isinstance(cls, type) else None
    if generic and generic.get("origin") is not None:
        # ``Page[int]``: pydantic also binds such classes in the caller's
        # module under the non-identifier name "Page[int]"; never import that.
        head = _class_ref(generic["origin"], ctx)
        return f"{head}[{', '.join(_type(a, ctx) for a in generic['args'])}]"
    return _import_reference(cls, ctx)


def split_discriminated(tp: Any) -> tuple[Any, Any, list[Any]]:
    """Split ``Annotated[Union[...], <disc>, *rest]`` into ``(base, disc, rest)``.

    ``disc`` is the discriminator (a ``str`` from ``Field(discriminator=...)``
    or a ``Discriminator``); ``(None, None, [])`` when ``tp`` is not a
    discriminated ``Annotated`` union with a plain discriminator item.
    """
    if get_origin(tp) is not Annotated:
        return None, None, []
    base, *meta = get_args(tp)
    if get_origin(base) not in _UNION_ORIGINS:
        return None, None, []
    disc = None
    rest: list[Any] = []
    for item in meta:
        if disc is None and isinstance(item, Discriminator):
            disc = item
        elif disc is None and isinstance(item, FieldInfo) and item.discriminator is not None:
            if _only_discriminator(item):
                disc = item.discriminator
            else:
                rest.append(item)
        else:
            rest.append(item)
    if disc is None:
        return None, None, []
    return base, disc, rest


def _only_discriminator(fi: FieldInfo) -> bool:
    """A FieldInfo that sets nothing but ``discriminator``."""
    if fi.metadata or fi.default is not PydanticUndefined or fi.default_factory is not None:
        return False
    for attr in ("alias", "validation_alias", "serialization_alias", "title", "description",
                 "examples", "json_schema_extra", "frozen", "validate_default", "exclude",
                 "deprecated"):
        if getattr(fi, attr, None) is not None:
            return False
    return getattr(fi, "repr", True) is True


def _annotated(tp: Any, ctx: RenderContext) -> str:
    base, disc, rest = split_discriminated(tp)
    if base is not None and not rest:
        alias = ctx.alias_for(base, disc)
        if alias is not None:
            return alias
    inner, *metadata = get_args(tp)
    return _wrap_metadata(_type(inner, ctx), metadata, ctx)


#: pydantic markers spelled as a subscript: ``Marker[T]`` is
#: ``Annotated[T, Marker()]``.  Their instances carry no state and do not
#: compare equal, so no value rung reproduces them; the subscript form is also
#: the one type checkers understand.
_SUBSCRIPT_MARKERS: dict[Any, tuple[str, str]] = {
    SkipJsonSchema: ("pydantic.json_schema", "SkipJsonSchema"),
    pydantic.Json: ("pydantic", "Json"),
    pydantic.ImportString: ("pydantic", "ImportString"),
}


def _subscript_marker(meta: Any) -> tuple[str, str] | None:
    """``(module, name)`` when ``meta`` is a stateless subscript marker instance."""
    spelling = _SUBSCRIPT_MARKERS.get(type(meta))
    if spelling is None or getattr(meta, "__dict__", None):
        return None
    return spelling


def _wrap_metadata(type_src: str, metadata: list[Any], ctx: RenderContext) -> str:
    """``type_src`` wrapped in its ``Annotated`` metadata (if any), in order.

    Subscript markers (``SkipJsonSchema()``, ``Json()``, ``ImportString()``)
    are written in the subscript form pydantic documents --
    ``Annotated[T, a, Json(), b]`` becomes ``Annotated[Json[Annotated[T, a]], b]``
    (the same metadata, in the same order).
    """
    src = type_src
    pending: list[Any] = []
    for meta in metadata:
        spelling = _subscript_marker(meta)
        if spelling is None:
            pending.append(meta)
            continue
        src = f"{ctx.ns.helper(*spelling)}[{_annotate_items(src, pending, ctx)}]"
        pending = []
    return _annotate_items(src, pending, ctx)


def _annotate_items(type_src: str, metadata: list[Any], ctx: RenderContext) -> str:
    items = _metadata_items(metadata, ctx)
    if not items:
        return type_src
    return f"{ctx.ns.helper('typing', 'Annotated')}[{type_src}, {', '.join(items)}]"


# ---------------------------------------------------------------------------
# Metadata (constraints, validators, markers)
# ---------------------------------------------------------------------------


def _constraint_kwargs(meta: Any) -> list[tuple[str, Any]] | None:
    """``Field`` keywords equivalent to one metadata item, or ``None``."""
    if meta is None:
        return []
    kind = type(meta)
    if kind in _SIMPLE_CONSTRAINTS:
        name = _SIMPLE_CONSTRAINTS[kind]
        return [(name, getattr(meta, name))]
    if kind is annotated_types.Interval:
        return [(k, getattr(meta, k)) for k in ("gt", "ge", "lt", "le") if getattr(meta, k) is not None]
    if kind is annotated_types.Len:
        out: list[tuple[str, Any]] = []
        if meta.min_length:
            out.append(("min_length", meta.min_length))
        if meta.max_length is not None:
            out.append(("max_length", meta.max_length))
        return out
    module = getattr(kind, "__module__", "") or ""
    if module.startswith("pydantic"):
        flag = _PYDANTIC_FLAG_CONSTRAINTS.get(kind.__name__)
        if flag is not None and dataclasses.is_dataclass(meta):
            return [(flag, getattr(meta, flag))] if flag in _FIELD_PARAMS else None
        if kind.__name__ == "_PydanticGeneralMetadata":
            items = list(vars(meta).items())
            if all(k in _FIELD_PARAMS for k, _ in items):
                return items
            return None
    return None


def _kwargs_src(pairs: Iterable[tuple[str, Any]], ctx: RenderContext) -> list[str]:
    """``key=value`` sources for constraint keywords.

    A constraint value with no source form (say a compiled pattern) is metadata
    the in-process model keeps enforcing: it is dropped with a note.
    """
    out: list[str] = []
    for key, value in pairs:
        try:
            out.append(f"{key}={_value(value, ctx)}")
        except _Unrenderable:
            ctx.note(f"{key}={describe(value)}")
    return out


def _metadata_items(metadata: Iterable[Any], ctx: RenderContext) -> list[str]:
    """Render ``Annotated`` metadata in order; constraint runs become ``Field(...)``."""
    items: list[str] = []
    run: list[tuple[str, Any]] = []

    def flush() -> None:
        if run:
            items.append(f"{ctx.ns.helper('pydantic', 'Field')}({', '.join(_kwargs_src(run, ctx))})")
            run.clear()

    for meta in metadata:
        if meta is None:
            continue
        if isinstance(meta, FieldInfo):
            flush()
            items.extend(_fieldinfo_items(meta, ctx))
            continue
        pairs = _constraint_kwargs(meta)
        if pairs is not None:
            for key, value in pairs:
                if any(k == key for k, _ in run):
                    flush()
                run.append((key, value))
            continue
        flush()
        rendered = _metadata_object(meta, ctx)
        if rendered is not None:
            items.append(rendered)
    flush()
    return items


def _metadata_object(meta: Any, ctx: RenderContext) -> str | None:
    """Render one non-constraint metadata item; ``None`` (with a note) if impossible."""
    try:
        if isinstance(meta, Tag):
            return f"{ctx.ns.helper('pydantic', 'Tag')}({_value(meta.tag, ctx)})"
        if isinstance(meta, Discriminator):
            return _discriminator_call(meta, ctx)
        return _value(meta, ctx)
    except _Unrenderable:
        ctx.note(describe(meta))
        return None


def _discriminator_call(disc: Discriminator, ctx: RenderContext) -> str:
    target = disc.discriminator
    target_src = _value(target, ctx) if isinstance(target, str) else ctx.ref(target)
    parts = [target_src]
    for attr in ("custom_error_type", "custom_error_message", "custom_error_context"):
        value = getattr(disc, attr)
        if value is not None:
            parts.append(f"{attr}={_value(value, ctx)}")
    return f"{ctx.ns.helper('pydantic', 'Discriminator')}({', '.join(parts)})"


# ---------------------------------------------------------------------------
# Values (the default ladder)
# ---------------------------------------------------------------------------


def render_value(value: Any, ctx: RenderContext) -> str:
    """Source for ``value`` (literal → Enum → constructor → reference), verified.

    Raises:
        ConfigRenderError: no faithful source exists.
    """
    try:
        return _value(value, ctx)
    except _Unrenderable as exc:
        raise ctx.fail(exc, value) from None
    except RecursionError:
        # A value that contains itself (``x = []; x.append(x)``) has no
        # finite source form.
        raise ctx.fail(
            _Unrenderable("the value contains itself, so it has no finite source form"),
            value,
        ) from None


def _value(value: Any, ctx: RenderContext) -> str:
    rendered, cause = _source_form(value, ctx)
    if rendered is not None:
        return rendered
    return _reference_or_cause(value, ctx, cause)


def _source_form(value: Any, ctx: RenderContext) -> tuple[str | None, _Unrenderable | None]:
    """The literal, Enum and constructor rungs.

    Returns ``(source, None)``, or ``(None, cause)`` where ``cause`` is the
    failure of a nested element (kept for a precise error message).
    """
    try:
        literal = _literal(value, ctx)
        if literal is not None:
            return literal, None
        if isinstance(value, enum.Enum):
            return _enum_member(value, ctx), None
    except _Unrenderable as exc:
        return None, exc
    return _constructor(value, ctx)


def _reference_or_cause(value: Any, ctx: RenderContext, cause: _Unrenderable | None) -> str:
    """The reference rung; on failure prefer the nested ``cause``."""
    try:
        return _reference_value(value, ctx)
    except _Unrenderable:
        if cause is not None:
            raise cause from None
        raise


def _literal(value: Any, ctx: RenderContext) -> str | None:
    kind = type(value)
    if value is None or kind in (bool, int, str, bytes):
        return repr(value)
    if value is Ellipsis:
        return "..."
    if kind is float:
        if math.isfinite(value):
            return repr(value)
        return f"{ctx.ns.builtin('float')}({str(value)!r})"
    if kind is complex:
        if math.isfinite(value.real) and math.isfinite(value.imag):
            return repr(value)
        return None
    if kind is list:
        return "[" + ", ".join(_value(v, ctx) for v in value) + "]"
    if kind is tuple:
        inner = [_value(v, ctx) for v in value]
        if len(inner) == 1:
            return f"({inner[0]},)"
        return "(" + ", ".join(inner) + ")"
    if kind is dict:
        return "{" + ", ".join(f"{_value(k, ctx)}: {_value(v, ctx)}" for k, v in value.items()) + "}"
    if kind is set:
        if not value:
            return f"{ctx.ns.builtin('set')}()"
        return "{" + ", ".join(_sorted_members(value, ctx)) + "}"
    if kind is frozenset:
        head = ctx.ns.builtin("frozenset")
        if not value:
            return f"{head}()"
        return f"{head}({{" + ", ".join(_sorted_members(value, ctx)) + "})"
    return None


def _sorted_members(value: set[Any] | frozenset[Any], ctx: RenderContext) -> list[str]:
    """Rendered members of a set, in an order independent of ``PYTHONHASHSEED``.

    Sorted by their text with every placeholder spelled out by the object it
    names (:meth:`ModuleNamespace.materialize_qualified`): placeholder numbers
    follow claim order, which for members of not-yet-claimed classes is the
    set's iteration order.
    """
    rendered = [_value(v, ctx) for v in value]
    return sorted(rendered, key=ctx.ns.materialize_qualified)


def _enum_member(value: enum.Enum, ctx: RenderContext) -> str:
    cls = type(value)
    name = value.name
    if (
        isinstance(name, str)
        and name.isidentifier()
        and not keyword.iskeyword(name)
        and getattr(cls, name, None) is value
    ):
        return f"{_class_ref(cls, ctx)}.{name}"
    expr = f"{_class_ref(cls, ctx)}({_value(value.value, ctx)})"
    if _verified(expr, value, ctx):
        return expr
    raise _Unrenderable(f"enum member {describe(value)} cannot be reconstructed")


def _verified(expr: str, value: Any, ctx: RenderContext) -> bool:
    """``eval(expr)`` reproduces ``value`` (:func:`_same_value`).

    A candidate expression is arbitrary user construction code (a dataclass
    ``__post_init__``, a model validator, ...), so any exception it raises
    only means "this rung does not reproduce the value" — the ladder moves on
    and ends in a ``ConfigRenderError`` if nothing reproduces it.
    """
    try:
        result = ctx.ns.evaluate(expr)
    except Exception:  # noqa: BLE001 — see docstring: a failed candidate, not a user error
        return False
    return _same_value(result, value)


def _same_value(a: Any, b: Any, _depth: int = 0) -> bool:
    """``a`` and ``b`` are the same value: same type at every level, and equal.

    ``NaN`` (float, ``Decimal``, inside complex numbers and containers) equals
    ``NaN``.  Containers compare member by member (``dict`` and sequences in
    order: rendering preserves it); ``datetime`` / ``time`` / ``timezone``
    values must also print the same (aware values in different zones compare
    equal as instants, and timezones compare by offset only); pydantic models
    compare like pydantic does (field values, extras, private attributes),
    member by member.  An object is always the same as itself; other values
    whose ``==`` raises are not the same.
    """
    if a is b:
        return True
    if type(a) is not type(b):
        return False
    if _depth > 50:
        return _safe_equal(a, b)
    depth = _depth + 1
    if isinstance(a, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    if isinstance(a, complex):
        return _same_value(a.real, b.real, depth) and _same_value(a.imag, b.imag, depth)
    if isinstance(a, decimal.Decimal):
        if a.is_nan() or b.is_nan():
            return str(a) == str(b)
        return _safe_equal(a, b)
    if isinstance(a, (datetime.datetime, datetime.time, datetime.timezone)):
        return _safe_equal(a, b) and repr(a) == repr(b)
    if type(a) in (list, tuple):
        return len(a) == len(b) and all(_same_value(x, y, depth) for x, y in zip(a, b))
    if type(a) is dict:
        return len(a) == len(b) and all(
            _same_value(ka, kb, depth) and _same_value(va, vb, depth)
            for (ka, va), (kb, vb) in zip(a.items(), b.items())
        )
    if type(a) in (set, frozenset):
        if len(a) != len(b):
            return False
        rest = list(b)
        for item in a:
            match = next((i for i, other in enumerate(rest) if _same_value(item, other, depth)), None)
            if match is None:
                return False
            rest.pop(match)
        return True
    if isinstance(a, BaseModel):
        return (
            all(
                _same_value(a.__dict__.get(name, _NO_SUBJECT), b.__dict__.get(name, _NO_SUBJECT), depth)
                for name in type(a).model_fields
            )
            and _same_value(a.__pydantic_extra__, b.__pydantic_extra__, depth)
            and _same_value(
                getattr(a, "__pydantic_private__", None), getattr(b, "__pydantic_private__", None), depth,
            )
        )
    return _safe_equal(a, b)


def _constructor(value: Any, ctx: RenderContext) -> tuple[str | None, _Unrenderable | None]:
    """Constructor form (``Cls(...)``) verified by ``eval``.

    Returns ``(source, None)``, ``(None, None)`` when no constructor form
    fits, or ``(None, cause)`` when one would fit but a part of it (the class,
    an argument) cannot be referenced.
    """
    if isinstance(value, type) or (callable(value) and not dataclasses.is_dataclass(value)):
        return None, None
    try:
        if isinstance(value, BaseModel):
            return _model_constructor(value, ctx), None
        if isinstance(value, (pydantic.AliasPath, pydantic.AliasChoices)):
            # Dataclasses with a positional-only ``__init__(first, *rest)``.
            return _alias_constructor(value, ctx), None
        if dataclasses.is_dataclass(value):
            return _dataclass_constructor(value, ctx), None
        if isinstance(value, pathlib.PurePath):
            return _path_constructor(value, ctx), None
        if type(value) is re.Pattern:
            return _pattern_constructor(value, ctx), None
        if type(value) is datetime.timezone:
            return _timezone_constructor(value, ctx), None
        return _repr_constructor(value, ctx), None
    except _Unrenderable as exc:
        return None, exc


def _model_constructor(value: BaseModel, ctx: RenderContext) -> str | None:
    cls = type(value)
    fields = cls.model_fields
    if any(not n.isidentifier() or keyword.iskeyword(n) for n in fields):
        return None
    if any(fi.alias is not None and fi.alias != n for n, fi in fields.items()):
        return None
    head = _class_ref(cls, ctx)
    extra = getattr(value, "__pydantic_extra__", None) or {}
    if any(not (isinstance(k, str) and k.isidentifier()) for k in extra):
        return None
    for names in (sorted(value.model_fields_set, key=list(fields).index), list(fields)):
        parts = [f"{n}={_value(getattr(value, n), ctx)}" for n in names]
        parts += [f"{k}={_value(v, ctx)}" for k, v in extra.items()]
        expr = f"{head}({', '.join(parts)})"
        if _verified(expr, value, ctx):
            return expr
    return None


def _dataclass_constructor(value: Any, ctx: RenderContext) -> str | None:
    cls = type(value)
    parts = []
    for f in dataclasses.fields(value):
        if not f.init:
            continue
        current = getattr(value, f.name)
        if f.default is not dataclasses.MISSING:
            try:
                if type(current) is type(f.default) and current == f.default:
                    continue
            except Exception:  # noqa: BLE001 — incomparable: keep the argument
                pass
        parts.append(f"{f.name}={_value(current, ctx)}")
    expr = f"{_class_ref(cls, ctx)}({', '.join(parts)})"
    return expr if _verified(expr, value, ctx) else None


def _path_constructor(value: pathlib.PurePath, ctx: RenderContext) -> str | None:
    cls = type(value)
    if isinstance(value, pathlib.Path) and type(pathlib.Path(str(value))) is cls:
        head = _class_ref(pathlib.Path, ctx)
    else:
        head = _class_ref(cls, ctx)
    expr = f"{head}({str(value)!r})"
    return expr if _verified(expr, value, ctx) else None


def _alias_constructor(value: Any, ctx: RenderContext) -> str | None:
    if isinstance(value, pydantic.AliasPath):
        args = [_value(p, ctx) for p in value.path]
    else:
        args = [_value(c, ctx) for c in value.choices]
    expr = f"{_class_ref(type(value), ctx)}({', '.join(args)})"
    return expr if _verified(expr, value, ctx) else None


#: ``re`` flags in the order they are written (``re.compile(p, re.I | re.M)``).
_RE_FLAG_NAMES: tuple[str, ...] = (
    "ASCII", "IGNORECASE", "LOCALE", "UNICODE", "MULTILINE", "DOTALL", "VERBOSE", "DEBUG",
)


def _pattern_constructor(value: re.Pattern[Any], ctx: RenderContext) -> str | None:
    """``re.compile(<pattern>[, <flags>])`` for a compiled pattern.

    The implicit ``re.UNICODE`` of ``str`` patterns is left out; other flags
    are written by name (``re.IGNORECASE | re.MULTILINE``).
    """
    flags = value.flags
    if isinstance(value.pattern, str):
        flags &= ~re.UNICODE
    mod = ctx.ns.module("re")
    args = [_value(value.pattern, ctx)]
    if flags:
        names = []
        for name in _RE_FLAG_NAMES:
            bit = int(getattr(re, name))
            if flags & bit:
                names.append(f"{mod}.{name}")
                flags &= ~bit
        if flags:  # bits without a name (internal flags): keep them as a number
            names.append(str(flags))
        args.append(" | ".join(names))
    expr = f"{mod}.compile({', '.join(args)})"
    return expr if _verified(expr, value, ctx) else None


def _timezone_constructor(value: datetime.timezone, ctx: RenderContext) -> str | None:
    """``timezone.utc`` or ``timezone(<offset>[, <name>])``.

    ``timezone`` compares by offset only, so :func:`_same_value` also
    compares the reprs: the name is written when the unnamed form does not
    reproduce the value (``timezone(timedelta(0), 'UTC')`` is not
    ``timezone.utc``).
    """
    head = _class_ref(datetime.timezone, ctx)
    if value is datetime.timezone.utc:
        return f"{head}.utc"
    offset = _value(value.utcoffset(None), ctx)
    for args in ([offset], [offset, _value(value.tzname(None), ctx)]):
        expr = f"{head}({', '.join(args)})"
        if _verified(expr, value, ctx):
            return expr
    return None


def _repr_constructor(value: Any, ctx: RenderContext) -> str | None:
    """Re-emit ``repr(value)`` when it is ``Cls(<args>)`` for ``type(value)``.

    Positional arguments must be literals.  A keyword argument that is not a
    literal is written from the value's attribute of the same name, through
    the whole ladder (``datetime(..., tzinfo=timezone.utc)``: ``tzinfo`` is
    ``value.tzinfo``).  The expression is verified by ``eval``.
    """
    cls = type(value)
    try:
        node = ast.parse(repr(value), mode="eval").body
    except (SyntaxError, ValueError):
        return None
    if not isinstance(node, ast.Call):
        return None
    callee = ast.unparse(node.func)
    accepted = {cls.__qualname__, cls.__name__, f"{cls.__module__}.{cls.__qualname__}"}
    if callee not in accepted:
        return None
    args: list[str] = []
    for arg in node.args:
        if isinstance(arg, ast.Starred):
            return None
        try:
            args.append(_value(ast.literal_eval(arg), ctx))
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            return None
    for kw in node.keywords:
        if kw.arg is None:
            return None
        try:
            args.append(f"{kw.arg}={_value(ast.literal_eval(kw.value), ctx)}")
        except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
            attr = getattr(value, kw.arg, _NO_SUBJECT)
            if attr is _NO_SUBJECT:
                return None
            args.append(f"{kw.arg}={_value(attr, ctx)}")
    expr = f"{_class_ref(cls, ctx)}({', '.join(args)})"
    return expr if _verified(expr, value, ctx) else None


def _reference_value(value: Any, ctx: RenderContext) -> str:
    sentinel = _SENTINEL_NAMES.get(id(value))
    if sentinel is not None:
        raise _Unrenderable(f"the {sentinel} sentinel is not a value", value)
    return ctx.ref(value)


def _json_form(value: Any, fi: FieldInfo, ctx: RenderContext) -> str | None:
    """JSON-compatible literal the generated field validates back to ``value``.

    The generated file writes ``Field(default=<literal>, validate_default=True)``,
    so the literal is validated with everything the field declares --
    constraints, ``strict``, validators in its metadata, its discriminator --
    under the model's config, on every instantiation.  A probe model built
    that way (:func:`_probe_default`) must turn the literal into a value
    equal to (and of the same type as) the in-process default; otherwise the
    rung does not apply.  Fields that a model-level ``@field_validator``
    covers never take it: those validators would run on the generated
    default only.
    """
    model = ctx.model
    if model is None or _has_field_validators(model, ctx.field_name):
        return None
    try:
        dumped = TypeAdapter(fi.annotation).dump_python(value, mode="json")
    except (pydantic.PydanticSchemaGenerationError, pydantic.PydanticUserError,
            pydantic_core.PydanticSerializationError, TypeError, ValueError):
        return None
    produced = _probe_default(dumped, fi, model)
    if produced is _PROBE_FAILED:
        return None
    expected = value
    validates = fi.validate_default
    if validates is None:  # the field defers to the model's config
        validates = bool(model.model_config.get("validate_default"))
    if validates:
        # The in-process model validates its default too: both must agree.
        expected = _probe_default(value, fi, model)
        if expected is _PROBE_FAILED:
            return None
    if not _same_value(produced, expected):
        return None
    try:
        return _literal(dumped, ctx)
    except _Unrenderable:
        return None


#: :func:`_probe_default` result when validating the default fails.
_PROBE_FAILED: Any = object()


def _probe_default(default: Any, fi: FieldInfo, model: type[BaseModel]) -> Any:
    """The value a field like ``fi`` with ``validate_default=True`` gets from ``default``.

    The probe declares the field the way the generated file does --
    ``Annotated[<annotation>, *<metadata>]`` with ``Field(default=...,
    validate_default=True)`` and the discriminator -- under ``model``'s
    config.  Validation runs user code (validators in the metadata), so any
    exception only means "this default is not reproduced", like a failed
    candidate in :func:`_verified`.
    """
    metadata = [m for m in fi.metadata if m is not None]
    annotation = Annotated[(fi.annotation, *metadata)] if metadata else fi.annotation  # type: ignore[valid-type]
    field_kwargs: dict[str, Any] = {"default": default, "validate_default": True}
    if fi.discriminator is not None:
        field_kwargs["discriminator"] = fi.discriminator
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            probe = pydantic.create_model(
                "_ConscribeDefaultProbe",
                __config__=pydantic.ConfigDict(**dict(model.model_config)),  # type: ignore[typeddict-item]
                value=(annotation, Field(**field_kwargs)),
            )
            return getattr(probe(), "value")
    except Exception:  # noqa: BLE001 — see docstring: a failed candidate, not a user error
        return _PROBE_FAILED


def _has_field_validators(model: type[BaseModel], name: str) -> bool:
    """Whether a ``@field_validator`` / ``@validator`` of ``model`` covers field ``name``."""
    decorators = getattr(model, "__pydantic_decorators__", None)
    for kind in ("field_validators", "validators"):
        for decorator in (getattr(decorators, kind, None) or {}).values():
            fields = getattr(getattr(decorator, "info", None), "fields", ()) or ()
            if name in fields or "*" in fields:
                return True
    return False


def _sentinel_check(value: Any, ctx: RenderContext) -> None:
    name = _SENTINEL_NAMES.get(id(value))
    if name is not None:
        raise ctx.error(
            value,
            f"the {name} sentinel reached the config model as a default; this is a "
            f"conscribe extraction bug — please report it",
        )
    module = getattr(type(value), "__module__", "") or ""
    if is_generated_model(type(value)):
        return
    if module == "conscribe" or module.startswith("conscribe."):
        detail = (
            f"the default is an object of conscribe's own type "
            f"{module}.{type(value).__qualname__}, which is not a config value; "
            f"generated files never import conscribe internals"
        )
        if type(value).__name__ == "WiredField":
            detail += (
                " (a WiredField reaching a default means the class declares a wired "
                "attribute that conscribe also injected)"
            )
        raise ctx.error(value, detail)


def _factory(factory: Any, ctx: RenderContext) -> str:
    try:
        if is_generated_model(factory):
            return ctx.model_ref(factory)
        if callable(factory) and getattr(builtins, getattr(factory, "__name__", ""), None) is factory:
            return ctx.ns.builtin(factory.__name__)
        return ctx.ref(factory)
    except _Unrenderable as exc:
        raise _Unrenderable(
            f"default_factory cannot be written into the generated file ({exc.detail}); "
            f"use a plain default value or a module-level factory function",
            factory,
        ) from None


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------


def _fieldinfo_items(fi: FieldInfo, ctx: RenderContext) -> list[str]:
    """``Annotated`` items for a ``FieldInfo`` found inside a type expression.

    The FieldInfo's own attributes (and its constraints, when they are all
    expressible as keywords) become one ``Field(...)``; any other metadata it
    carries follows as separate items, in order.
    """
    head: list[str] = []
    try:
        if fi.default_factory is not None:
            head.append(f"default_factory={_factory(fi.default_factory, ctx)}")
        elif fi.default is not PydanticUndefined and fi.default is not Ellipsis:
            head.append(f"default={_value(fi.default, ctx)}")
    except _Unrenderable:
        ctx.note(f"default of a nested Field ({describe(fi.default)})")
        head = []
    head.extend(_attribute_kwargs(fi, ctx, skip=()))

    metadata = [m for m in fi.metadata if m is not None]
    converted = [_constraint_kwargs(m) for m in metadata]

    def call(parts: list[str]) -> list[str]:
        if not parts:
            return []
        return [f"{ctx.ns.helper('pydantic', 'Field')}({', '.join(parts)})"]

    if all(c is not None for c in converted):
        pairs = [p for c in converted for p in c]  # type: ignore[union-attr]
        if len({k for k, _ in pairs}) == len(pairs):
            return call(head + _kwargs_src(pairs, ctx))
    return call(head) + _metadata_items(metadata, ctx)


def _attribute_kwargs(fi: FieldInfo, ctx: RenderContext, *, skip: Iterable[str]) -> list[str]:
    """``Field`` keywords for the FieldInfo attributes that are set."""
    skipped = set(skip)
    out: list[str] = []

    def add(name: str, value: Any) -> None:
        if name in skipped or name not in _FIELD_PARAMS:
            return
        try:
            out.append(f"{name}={_attribute_value(value, ctx)}")
        except _Unrenderable:
            ctx.note(f"{name}={describe(value)}")

    alias = fi.alias
    if alias is not None:
        add("alias", alias)
    if fi.validation_alias is not None and fi.validation_alias != alias:
        add("validation_alias", fi.validation_alias)
    if fi.serialization_alias is not None and fi.serialization_alias != alias:
        add("serialization_alias", fi.serialization_alias)
    for name in ("title", "description", "examples", "json_schema_extra", "discriminator",
                 "deprecated", "frozen", "validate_default", "exclude",
                 "field_title_generator", "exclude_if"):
        value = getattr(fi, name, None)
        if value is not None:
            add(name, value)
    if getattr(fi, "repr", True) is False:
        add("repr", False)
    return out


def _attribute_value(value: Any, ctx: RenderContext) -> str:
    if isinstance(value, Discriminator):
        return _discriminator_call(value, ctx)
    return _value(value, ctx)


def _is_field_name(name: str) -> bool:
    return name.isidentifier() and not keyword.iskeyword(name)


def render_field(
    name: str,
    fi: FieldInfo,
    ctx: RenderContext,
    *,
    base: type = BaseModel,
    comments: Iterable[str] = (),
) -> str:
    """Render one field line (D5 rules).

    * required, nothing else to say  → ``name: T``
    * required with keywords          → ``name: T = Field(<kwargs>)``
    * default, nothing else to say    → ``name: T = <expr>``
    * default with keywords           → ``name: T = Field(default=<expr>, <kwargs>)``
    * default factory                 → ``name: T = Field(default_factory=<ref>, ...)``

    A name that shadows an attribute of ``base`` (``validate``, ``json``,
    ``copy``, ...) always gets an explicit ``Field(...)``: on pydantic < 2.12 a
    bare annotation would pick up the inherited attribute as its default.
    Type checkers report such a field as an incompatible override of the
    inherited method / attribute; the line carries a ``# pyright: ignore``
    for that rule.  Every field name is reported to the namespace, so no
    import, helper or builtin in the file uses it: an assigned name would
    shadow the module-level binding in the class body at runtime, and type
    checkers scope even an annotation-only name to the class body.  A
    definition of the file (a generated class) under the name of an
    assigned field is an error.

    Raises:
        ConfigRenderError: the type, default or factory cannot be written.
    """
    ctx.field_name = name
    notes_before = len(ctx.notes)
    try:
        if not _is_field_name(name):
            raise ctx.error(name, _field_name_problem(name))
        line, assigned, shadowed = _field_line(name, fi, ctx, base)
    except _Unrenderable as exc:
        raise ctx.fail(exc, fi.annotation) from None
    except RecursionError:
        # A default (or metadata value) that contains itself, e.g.
        # ``x = []; x.append(x)``, has no finite source form.
        raise ctx.fail(
            _Unrenderable("a value of this field contains itself, so it has no finite source form"),
            fi.default,
        ) from None
    finally:
        ctx.field_name = ""
    ctx.ns.forbid(name, f"{ctx.model_name or '<module>'}.{name}", assigned=assigned)
    trailing = [_comment(c) for c in comments if c]
    dropped = [n.what for n in ctx.notes[notes_before:]]
    if dropped:
        trailing.append(_comment("not rendered: " + ", ".join(dropped)))
    if trailing:
        line += "  # " + "; ".join(trailing)
    if shadowed is not None:
        line += f"  # pyright: ignore[{shadowed}]"
    return line


def _field_name_problem(name: str) -> str:
    """Why ``name`` cannot be a class attribute in the generated file, and the fix."""
    if isinstance(name, str) and keyword.iskeyword(name):
        why = f"the field name {name!r} is a Python keyword"
    else:
        why = f"the field name {name!r} is not a valid Python identifier"
    return (
        f"{why}, so it cannot be written as a class attribute; rename the field and "
        f"keep {name!r} as its config key with Field(alias={name!r})"
    )


def _override_rule(base: type, name: str) -> str | None:
    """The type-checker rule a field ``name`` declared on a subclass of ``base`` breaks.

    ``None`` when ``base`` has no such attribute.  Methods and properties
    (``BaseModel.copy``, ``.json``, ``.schema``, ...) are reported as
    ``reportIncompatibleMethodOverride``, other class attributes as
    ``reportIncompatibleVariableOverride``.
    """
    try:
        attr = inspect.getattr_static(base, name)
    except AttributeError:
        return None
    if hasattr(type(attr), "__get__"):  # functions, class/static methods, properties
        return "reportIncompatibleMethodOverride"
    return "reportIncompatibleVariableOverride"


def _field_line(
    name: str, fi: FieldInfo, ctx: RenderContext, base: type,
) -> tuple[str, bool, str | None]:
    """``(line, assigned, rule)``: the field statement, whether it assigns a
    value, and the type-checker rule it breaks by shadowing an attribute of
    ``base`` (``None`` when it shadows nothing)."""
    annotation = fi.annotation
    metadata = [m for m in fi.metadata if m is not None]
    skip: set[str] = set()

    # A discriminated union equal to a whole layer renders as that layer's alias
    # (which carries the discriminator itself).
    type_src = None
    if get_origin(annotation) in _UNION_ORIGINS:
        if isinstance(fi.discriminator, (str, Discriminator)):
            alias = ctx.alias_for(annotation, fi.discriminator)
            if alias is not None:
                type_src = alias
                skip.add("discriminator")
        if type_src is None:
            discs = [m for m in metadata if isinstance(m, Discriminator)]
            if len(discs) == 1:
                alias = ctx.alias_for(annotation, discs[0])
                if alias is not None:
                    type_src = alias
                    metadata = [m for m in metadata if m is not discs[0]]
    if type_src is None:
        type_src = _type(annotation, ctx)

    # Constraint metadata becomes Field keywords; anything else (validators,
    # markers, callable discriminators) is kept, in order, inside Annotated.
    converted = [_constraint_kwargs(m) for m in metadata]
    constraint_kwargs: list[str] = []
    if all(c is not None for c in converted):
        pairs = [p for c in converted for p in c]  # type: ignore[union-attr]
        if len({k for k, _ in pairs}) == len(pairs):
            constraint_kwargs = _kwargs_src(pairs, ctx)
            metadata = []
    if metadata:
        type_src = _wrap_metadata(type_src, metadata, ctx)

    kwargs = _attribute_kwargs(fi, ctx, skip=skip) + constraint_kwargs

    default_kw: str | None = None
    plain_expr: str | None = None
    if fi.default_factory is not None:
        default_kw = f"default_factory={_factory(fi.default_factory, ctx)}"
    elif fi.default is not PydanticUndefined and fi.default is not Ellipsis:
        _sentinel_check(fi.default, ctx)
        default_kw, plain_expr, kwargs = _default_parts(fi.default, fi, ctx, kwargs)

    shadows = hasattr(base, name)
    rule = _override_rule(base, name) if shadows else None
    if default_kw is None:
        if not kwargs and not shadows:
            return f"{name}: {type_src}", False, None
        return f"{name}: {type_src} = {ctx.ns.helper('pydantic', 'Field')}({', '.join(kwargs)})", True, rule
    if plain_expr is not None and not kwargs and not shadows:
        return f"{name}: {type_src} = {plain_expr}", True, None
    field = ctx.ns.helper("pydantic", "Field")
    return f"{name}: {type_src} = {field}({', '.join([default_kw, *kwargs])})", True, rule


def _default_parts(
    default: Any, fi: FieldInfo, ctx: RenderContext, kwargs: list[str],
) -> tuple[str, str | None, list[str]]:
    """Render a static default: ``(keyword part, plain expression or None, kwargs)``.

    Ladder: literal → Enum member → constructor form → JSON form with
    ``validate_default=True`` (verified against the whole field, see
    :func:`_json_form`) → importable reference.  A generated submodel
    instance equal to a fresh instance becomes ``default_factory=<Sub>``.
    """
    cls = type(default)
    if is_generated_model(cls) and ctx.is_defined_model(cls):
        # The JSON-schema default the instance contributes in-process is kept
        # through json_schema_extra, so both schemas stay equal.
        extra = fi.json_schema_extra
        if (extra is None or isinstance(extra, dict)) and _equals_fresh(cls, default):
            merged: dict[str, Any] = {}
            schema_default = _schema_default(default, ctx)
            if schema_default is not _NO_SCHEMA_DEFAULT:
                merged["default"] = schema_default
            merged.update(extra or {})
            kwargs = [k for k in kwargs if not k.startswith("json_schema_extra=")]
            if merged:
                kwargs.append(f"json_schema_extra={_value(merged, ctx)}")
            return f"default_factory={ctx.model_ref(cls)}", None, kwargs

    expr, cause = _source_form(default, ctx)
    if expr is None:
        json_expr = _json_form(default, fi, ctx)
        if json_expr is not None:
            kwargs = [k for k in kwargs if not k.startswith("validate_default=")]
            kwargs.append("validate_default=True")
            return f"default={json_expr}", None, kwargs
        expr = _reference_or_cause(default, ctx, cause)
    if not _verified(expr, default, ctx):
        raise _Unrenderable(
            f"the rendered default {ctx.ns.readable(expr)} does not evaluate back to the value",
            default,
        )
    return f"default={expr}", expr, kwargs


def _equals_fresh(cls: type[BaseModel], value: Any) -> bool:
    try:
        return _same_value(cls(), value)
    except pydantic.ValidationError:
        return False


#: :func:`_schema_default` result when the JSON schema shows no default.
_NO_SCHEMA_DEFAULT: Any = object()


def _schema_default(value: Any, ctx: RenderContext) -> Any:
    """The ``default`` the published JSON schema shows for a field defaulting to ``value``.

    Computed exactly as for the in-process model: a probe model with the
    current model's ``ser_json_*`` settings and a field defaulting to
    ``value``, through the same generator as every published schema
    (:class:`~conscribe.config.json_schema._DeterministicJsonSchema`, which
    writes set-valued members sorted at any depth -- also inside model
    instances -- so the text does not depend on ``PYTHONHASHSEED``).

    Returns:
        The JSON value, or :data:`_NO_SCHEMA_DEFAULT` when pydantic leaves
        the default out (it is not JSON-serializable).
    """
    from conscribe.config.json_schema import _DeterministicJsonSchema

    config = ctx.model.model_config if ctx.model is not None else {}
    ser_config = pydantic.ConfigDict(**{k: v for k, v in config.items() if k.startswith("ser_json_")})  # type: ignore[typeddict-item]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probe = pydantic.create_model(
            "_ConscribeSchemaDefaultProbe", __config__=ser_config, value=(Any, value),
        )
        schema = probe.model_json_schema(schema_generator=_DeterministicJsonSchema)
    return schema["properties"]["value"].get("default", _NO_SCHEMA_DEFAULT)


# ---------------------------------------------------------------------------
# Model classes
# ---------------------------------------------------------------------------


def _own_annotations(klass: type) -> dict[str, Any]:
    """The annotations a class declares itself (PEP 649-safe; strings stay strings)."""
    if sys.version_info >= (3, 14):  # pragma: no cover - exercised on 3.14 only
        import annotationlib

        return dict(annotationlib.get_annotations(klass, format=annotationlib.Format.FORWARDREF))
    return dict(inspect.get_annotations(klass))


def _own_annotation_names(klass: type) -> set[str]:
    """Names a class annotates itself (PEP 649-safe)."""
    return set(_own_annotations(klass))


#: What :func:`_base_annotation_names` records for a name nothing binds.
_UNBOUND: Any = object()


def _base_annotation_names(base: type) -> dict[str, tuple[Any, str]]:
    """Names the string annotations along ``base``'s MRO use, with their bindings.

    pydantic < 2.9 evaluates the string (and forward-reference) annotations
    of every class a model inherits from in the namespace of the module
    that defines the subclass -- for an emitted class subclassing a user
    model, the generated file.  Each name maps to ``(object, class)``: what
    the declaring class's module binds it to (its globals, then builtins;
    :data:`_UNBOUND` when neither does) and the class, so the file binds the
    name to that object or not at all (:meth:`ModuleNamespace.pin`).
    """
    out: dict[str, tuple[Any, str]] = {}
    for klass in base.__mro__:
        if klass is BaseModel or not (isinstance(klass, type) and issubclass(klass, BaseModel)):
            continue
        texts: list[str] = []
        try:
            annotations = _own_annotations(klass)
        except (TypeError, ValueError, NameError, AttributeError):
            continue  # pydantic could not have read them either
        for value in annotations.values():
            _annotation_strings(value, texts, 0)
        if not texts:
            continue
        module = sys.modules.get(getattr(klass, "__module__", "") or "")
        namespace = getattr(module, "__dict__", None) or {}
        where = f"{klass.__module__}.{klass.__qualname__}"
        for text in texts:
            for name in sorted(_expression_names(text, 0)):
                if name in out:
                    continue
                if name in namespace:
                    out[name] = (namespace[name], where)
                elif hasattr(builtins, name):
                    out[name] = (getattr(builtins, name), where)
                else:
                    out[name] = (_UNBOUND, where)
    return out


def _annotation_strings(value: Any, out: list[str], depth: int) -> None:
    """Collect the unevaluated parts of an annotation (strings, forward refs)."""
    if depth > 20:
        return
    if isinstance(value, str):
        out.append(value)
        return
    if isinstance(value, typing.ForwardRef):
        out.append(value.__forward_arg__)
        return
    if get_origin(value) is Literal:
        return
    for arg in get_args(value):
        if isinstance(arg, (list, tuple)):
            for inner in arg:
                _annotation_strings(inner, out, depth + 1)
        else:
            _annotation_strings(arg, out, depth + 1)


def _expression_names(text: str, depth: int) -> set[str]:
    """Free names of the annotation expression ``text`` (nested strings too).

    ``Literal[...]`` arguments are values, not annotations, and are skipped.
    """
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except (SyntaxError, ValueError):
        return set()
    names: set[str] = set()

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            names.add(node.id)
            return
        if isinstance(node, ast.Subscript):
            visit(node.value)
            head = node.value
            tail = head.attr if isinstance(head, ast.Attribute) else getattr(head, "id", None)
            if tail != "Literal":
                visit(node.slice)
            return
        if isinstance(node, ast.Constant):
            if isinstance(node.value, str) and depth < 5:
                names.update(_expression_names(node.value, depth + 1))
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return names


def model_base(model: type[BaseModel]) -> type[BaseModel]:
    """The class the emitted class must subclass.

    conscribe's own classes -- generated models and the extractor's private
    ``_<Name>Base`` helpers (:func:`is_generated_model`) -- are flattened into
    the emitted class; the first other model in the MRO -- e.g. a
    ``__config_schema__`` class the per-key model inherits validators from --
    is imported and subclassed, whatever it declares (a user base may carry
    nothing but ``model_config`` and a ``__get_pydantic_json_schema__``
    hook, which flattening would lose).  ``BaseModel`` otherwise.
    """
    for klass in model.__mro__[1:]:
        if klass is BaseModel:
            return BaseModel
        if not (isinstance(klass, type) and issubclass(klass, BaseModel)):
            continue
        if is_generated_model(klass):
            continue
        return klass
    return BaseModel


def declared_fields(model: type[BaseModel], base: type[BaseModel]) -> list[str]:
    """Fields the emitted class must declare, in ``model_fields`` order."""
    if base is BaseModel:
        return list(model.model_fields)
    names: set[str] = set()
    for klass in model.__mro__:
        if klass is base:
            break
        names |= _own_annotation_names(klass)
    return [n for n in model.model_fields if n in names]


#: ``__pydantic_decorators__`` collections (validators, serializers, computed
#: fields); attributes missing on older pydantic versions are skipped.
_DECORATOR_KINDS: tuple[str, ...] = (
    "validators",
    "field_validators",
    "root_validators",
    "model_validators",
    "field_serializers",
    "model_serializers",
    "computed_fields",
)


def flattened_decorators(
    model: type[BaseModel], base: type[BaseModel],
) -> list[tuple[str, str, Any]]:
    """Decorators declared on the classes the emitted class flattens.

    The emitted class subclasses ``base`` -- inheriting its validators,
    serializers and computed fields -- and redeclares the fields of every
    class between ``model`` and ``base`` (conscribe's generated models and
    helper bases).  Whatever those classes *declare* themselves is not
    carried by inheritance; it must be rendered into the class body.

    A class declares a decorator when the decorated name is in its own
    ``__dict__`` (pydantic lists inherited decorators in every subclass's
    ``__pydantic_decorators__``, under the subclass's own reference).

    Returns:
        ``(kind, name, decorator)`` triples, nearest class first, one per
        name (a name redeclared further down the MRO is overridden).
    """
    skip = set(base.__mro__)
    seen: set[str] = set()
    out: list[tuple[str, str, Any]] = []
    for klass in model.__mro__:
        if klass in skip or not (isinstance(klass, type) and issubclass(klass, BaseModel)):
            continue
        infos = klass.__dict__.get("__pydantic_decorators__")
        if infos is None:
            continue
        own = vars(klass)
        for kind in _DECORATOR_KINDS:
            for name, decorator in (getattr(infos, kind, None) or {}).items():
                if name in own and name not in seen:
                    seen.add(name)
                    out.append((kind, name, decorator))
    return out


def _decorator_note(kind: str, name: str, decorator: Any) -> tuple[str, str]:
    """``(field, what)`` describing a decorator the file does not carry."""
    info = getattr(decorator, "info", None)
    fields = getattr(info, "fields", None)
    field = ", ".join(fields) if isinstance(fields, tuple) and fields else "<model>"
    label = kind[:-1].replace("_", " ")  # "field_validators" -> "field validator"
    mode = getattr(info, "mode", None)
    what = f"{label} {name!r}" + (f" (mode={mode!r})" if isinstance(mode, str) else "")
    return field, what


def render_model_class(
    model: type[BaseModel],
    ctx: RenderContext,
    *,
    field_comments: Mapping[str, Iterable[str]] | None = None,
    title: str | None = None,
    extra_lines: Iterable[str] = (),
    rendered_decorators: Iterable[str] = (),
    class_comments: Iterable[str] | None = None,
) -> str:
    """Render ``model`` (defined in ``ctx``) as a class statement.

    Validators, serializers and computed fields declared on the classes the
    emitted class flattens (see :func:`flattened_decorators`) are rendered by
    the renderer registered for their function
    (:meth:`RenderContext.register_validator`), or by the layout itself
    (``extra_lines`` + ``rendered_decorators``); any other one is recorded as
    a :class:`RenderNote` -- the generated file lacks it, the in-process
    model keeps enforcing it.

    Args:
        model: A generated model registered with ``ctx.define_model``.
        ctx: The render context.
        field_comments: Extra trailing comments per field
            (``# wired from: ...``, ``# degraded from: ...``).
        title: Schema title to pin when the emitted class name differs from
            the in-process name, so both JSON schemas keep the same title.
        extra_lines: Source lines appended to the class body, unindented
            (e.g. a validator the layout emits itself); they may contain
            placeholders from ``ctx.ns``.
        rendered_decorators: Names of the flattened decorators that
            ``extra_lines`` reproduces.
        class_comments: Comment lines (without ``#``) opening the class body.
            ``None`` uses the default: a note on ``extra="allow"`` models
            that undeclared fields are accepted.

    Raises:
        ConfigRenderError: something in the model cannot be written.
    """
    ctx.model = model
    ctx.model_name = model.__name__
    ctx.field_name = ""
    comments = field_comments or {}
    try:
        name_src = ctx.model_ref(model)
        base = model_base(model)
        if base is BaseModel:
            base_src = ctx.ns.helper("pydantic", "BaseModel")
        else:
            ctx.field_name = "(base class)"
            base_src = _type(base, ctx)
            ctx.field_name = ""
    except _Unrenderable as exc:
        raise ctx.fail(exc, model_base(model)) from None
    if base is not BaseModel:
        for name, (obj, where) in _base_annotation_names(base).items():
            ctx.ns.pin(name, obj, where)

    lines = [f"class {name_src}({base_src}):"]
    doc = model.__dict__.get("__doc__")
    if isinstance(doc, str) and doc:
        lines.append(f"    {doc!r}")
        lines.append("")

    config = dict(model.model_config)
    base_config = dict(base.model_config) if base is not BaseModel else {}
    if class_comments is None:
        class_comments = (
            ["This implementation accepts extra parameters (**kwargs).",
             "Undeclared fields will not cause errors."]
            if config.get("extra") == "allow" else []
        )
    lines.extend(f"    # {_comment(text)}" for text in class_comments)
    config_items = _config_items(config, base_config, ctx, title)
    if config_items:
        lines.append(f"    model_config = {ctx.ns.helper('pydantic', 'ConfigDict')}({', '.join(config_items)})")
        lines.append("")

    fields = declared_fields(model, base)
    for field_name in fields:
        fi = model.model_fields[field_name]
        line = render_field(field_name, fi, ctx, base=base, comments=comments.get(field_name, ()))
        lines.append(f"    {line}")
    rendered = set(rendered_decorators)
    extra: list[str] = []
    for kind, name, decorator in flattened_decorators(model, base):
        if name in rendered:
            continue
        renderer = ctx.validator_renderer(decorator)
        if renderer is not None:
            extra.extend([*([""] if extra else []), *renderer(model, name, decorator, ctx)])
            continue
        ctx.field_name, what = _decorator_note(kind, name, decorator)
        ctx.note(what)
    ctx.field_name = ""
    own_extra = list(extra_lines)
    if own_extra:
        extra.extend([*([""] if extra else []), *own_extra])
    if extra:
        lines.append("")
        lines.extend(f"    {line}" if line else "" for line in extra)
    if not fields and not config_items and not extra:
        lines.append("    pass")
    lines.append("")
    ctx.model_name = ""
    ctx.model = None
    return "\n".join(lines)


def _config_items(
    config: Mapping[str, Any],
    base_config: Mapping[str, Any],
    ctx: RenderContext,
    title: str | None,
) -> list[str]:
    items: list[str] = []
    if "extra" in config:
        items.append(f"extra={json.dumps(config['extra'])}")
    for key in sorted(config):
        if key == "extra":
            continue
        value = config[key]
        if key in base_config and _safe_equal(base_config[key], value):
            continue
        if not _is_field_name(key):
            ctx.field_name = "model_config"
            ctx.note(f"{key!r}={describe(value)}")
            ctx.field_name = ""
            continue
        ctx.field_name = "model_config"
        try:
            items.append(f"{key}={_value(value, ctx)}")
        except _Unrenderable:
            ctx.note(f"{key}={describe(value)}")
        finally:
            ctx.field_name = ""
    if title is not None and "title" not in config:
        items.append(f"title={title!r}")
    return items


def _safe_equal(a: Any, b: Any) -> bool:
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001 — incomparable values count as different
        return False


# ---------------------------------------------------------------------------
# Planning helpers (which generated models a file must define)
# ---------------------------------------------------------------------------


def generated_dependencies(
    roots: Iterable[type[BaseModel]],
    *,
    exclude: Iterable[type[BaseModel]] = (),
) -> list[type[BaseModel]]:
    """Generated models that ``roots`` reference, deepest first.

    A model is referenced through a field type (at any depth: ``Optional``,
    ``list``, ``Annotated``, unions), a default value (an instance of a
    generated model, also inside containers) or a ``default_factory`` that is
    a generated model.  The result is de-duplicated by identity and ordered
    post-order, so every model comes after the models its class body needs.
    ``roots`` and ``exclude`` are never part of the result.
    """
    ordered: list[type[BaseModel]] = []
    seen: set[int] = {id(m) for m in roots} | {id(m) for m in exclude}

    def visit(model: type[BaseModel]) -> None:
        for fi in model.model_fields.values():
            for dep in _field_dependencies(fi):
                if id(dep) in seen:
                    continue
                seen.add(id(dep))
                visit(dep)
                ordered.append(dep)

    for root in roots:
        visit(root)
    return ordered


def _field_dependencies(fi: FieldInfo) -> list[type[BaseModel]]:
    found: list[type[BaseModel]] = []
    _type_dependencies(fi.annotation, found, 0)
    _value_dependencies(fi.default, found, 0)
    if is_generated_model(fi.default_factory):
        found.append(fi.default_factory)  # type: ignore[arg-type]
    for meta in fi.metadata:
        if isinstance(meta, FieldInfo):
            found.extend(_field_dependencies(meta))
    return found


def _type_dependencies(tp: Any, found: list[type[BaseModel]], depth: int) -> None:
    if depth > 50:
        return
    if is_generated_model(tp):
        found.append(tp)
        return
    origin = get_origin(tp)
    if origin is Literal:
        return
    if origin is Annotated:
        base, *meta = get_args(tp)
        _type_dependencies(base, found, depth + 1)
        for item in meta:
            if isinstance(item, FieldInfo):
                found.extend(_field_dependencies(item))
        return
    for arg in get_args(tp):
        if isinstance(arg, (list, tuple)):
            for inner in arg:
                _type_dependencies(inner, found, depth + 1)
        else:
            _type_dependencies(arg, found, depth + 1)
    generic = getattr(tp, "__pydantic_generic_metadata__", None) if isinstance(tp, type) else None
    if generic:
        for arg in generic.get("args", ()):
            _type_dependencies(arg, found, depth + 1)


def _value_dependencies(value: Any, found: list[type[BaseModel]], depth: int) -> None:
    if depth > 20 or value is PydanticUndefined:
        return
    if isinstance(value, BaseModel) and is_generated_model(type(value)):
        found.append(type(value))
        return
    if type(value) in (list, tuple, set, frozenset):
        for item in value:
            _value_dependencies(item, found, depth + 1)
    elif type(value) is dict:
        for item in value.values():
            _value_dependencies(item, found, depth + 1)
