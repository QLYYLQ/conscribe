"""Constructor discovery and MRO parameter collection for ``**kwargs`` chains.

When a child class's ``__init__`` accepts ``**kwargs`` and passes them
to ``super().__init__(**kwargs)``, this module walks the MRO upwards to
collect parent parameters, producing a complete config schema.

Two views share one walker, :func:`walk_init_chain`:

* **runtime view** (``scope=None``, the default): every name ``cls(...)``
  accepts, following the ``**kwargs`` forwarding chain to its natural end no
  matter where the parents are installed.  :mod:`conscribe.introspect`
  answers from it.
* **config view** (``scope=`` / ``depth=``): the same walk, cut where the
  scope or depth settings say parents stop contributing config knobs.
  :func:`collect_mro_params` is the 1.x wrapper of that view.

Walk rules (both views):

1. The constructor's definer is :func:`~conscribe.config._utils.constructor_definer`
   (Protocol placeholders and copied ``__init__`` functions do not count).
   Fast paths: a pydantic model whose constructor chain only forwards
   ``**data`` up to ``BaseModel.__init__`` (cooperative mixins included)
   contributes its ``model_fields``; a pydantic dataclass contributes its
   fields.
2. If the definer's ``__init__`` takes ``**kwargs``, the walk continues up the
   MRO.  Classes without an ``__init__`` of their own (``Generic``, ``ABC``,
   Protocols, bridges, plain mixins) are skipped **before** scope and depth
   are checked, so they never cut the chain.
3. ``pydantic.BaseModel`` (which contributes the leaf model's fields) and a
   pydantic dataclass (which contributes its fields) end the chain
   naturally, also before scope and depth are checked: wherever pydantic is
   installed, its constructor is where the fields are validated.
4. A class excluded by *scope*, or the *depth* limit, cuts the chain
   (``fully_resolved=False``).
5. An ``__init__`` whose signature cannot be read, or a placeholder that
   swallows its arguments, also cuts it (``fully_resolved=False``): conscribe
   cannot know what lies beyond.  It is never mistaken for the natural end.
6. A parent whose ``__init__`` has no ``**kwargs`` ends the chain naturally
   (``fully_resolved=True``); so does ``object``.

Scope controls which classes in the MRO are included:
- ``"local"``: only classes NOT in site-packages (default)
- ``"third_party"``: include site-packages, exclude stdlib
- ``"all"``: all classes except ``object``
- ``list[str]``: local classes + only the listed third-party packages
  (e.g., ``["pydantic", "my_sdk"]``).  Stdlib is always excluded.
"""
from __future__ import annotations

import dataclasses
import difflib
import inspect
import pathlib
import sys
import sysconfig
import types
import weakref
from dataclasses import dataclass
from typing import Any, Literal, Union

from pydantic import BaseModel
from pydantic_core import PydanticUndefined

from conscribe.config import _annotations as _ann
from conscribe.config._utils import (
    constructor_definer,
    dataclass_field_owner,
    find_init_definer,
    is_bridge_class,
    is_generated_init,
    is_pydantic_dataclass,
    is_pydantic_model,
    is_swallowing_placeholder,
    model_init_only_forwards,
    named_params,
    own_init,
)

MROScope = Union[Literal["local", "third_party", "all"], list[str]]

#: Where a constructor parameter comes from.
ParamSource = Literal["init", "dataclass", "generated", "model", "pydantic_dataclass"]
#: What kind of constructor a class has.
ChainKind = Literal["init", "dataclass", "generated", "model", "pydantic_dataclass", "none"]
#: Why a ``**kwargs`` chain walk stopped before its natural end.
StopReason = Literal["scope", "depth", "unreadable", "swallowing_placeholder"]

#: Default of a parameter whose default comes from a ``default_factory``
#: (the same sentinel stdlib dataclasses put in a generated ``__init__``).
HAS_DEFAULT_FACTORY: Any = getattr(dataclasses, "_HAS_DEFAULT_FACTORY", None)


def is_no_default(value: Any) -> bool:
    """True for a "no default" marker: ``inspect.Parameter.empty``,
    ``dataclasses.MISSING``, pydantic's ``PydanticUndefined``, ``...`` (a
    required pydantic field), attrs ``NOTHING`` -- or a ``WiredField``
    placeholder that leaked into a signature (it is not a value)."""
    if (
        value is inspect.Parameter.empty
        or value is dataclasses.MISSING
        or value is Ellipsis
        or value is PydanticUndefined
    ):
        return True
    kind = type(value)
    if kind.__name__ == "_Nothing" and kind.__module__.startswith("attr"):
        return True  # attrs NOTHING
    from conscribe.wiring import WiredField

    return isinstance(value, WiredField)


def _declared_value(value: Any) -> DeclaredDefault:
    """A :class:`DeclaredDefault` for a plain default value (sentinels mapped)."""
    if value is HAS_DEFAULT_FACTORY or is_no_default(value):
        return DeclaredDefault()
    if _is_field_info(value):
        return DeclaredDefault(field_info=value)
    return DeclaredDefault(value=value)


def _is_field_info(value: Any) -> bool:
    from pydantic.fields import FieldInfo

    return isinstance(value, FieldInfo)


@dataclass(frozen=True)
class DeclaredDefault:
    """The default a constructor parameter itself declares.

    See :meth:`InitParam.declared_default`.  Never holds a sentinel
    (``<factory>``, ``NOTHING``, ``PydanticUndefined``, ``MISSING``).

    Attributes:
        value: The default value; ``inspect.Parameter.empty`` when there is
            none (also when the default comes from a factory).
        factory: The zero-argument ``default_factory``
            (``dataclasses.field``, attrs ``Factory``, a pydantic field's
            ``default_factory``), else ``None``.
        field_info: The pydantic ``FieldInfo`` behind the parameter: a
            ``Field(...)`` written as the default value (``x: int =
            Field(1, ge=0)`` declares the field, as in pydantic) or the field
            of a pydantic model / dataclass.  Its own default / factory is
            *not* copied into :attr:`value` / :attr:`factory`.
        takes_self: An attrs ``Factory(takes_self=True)``: the constructor
            has a default, but no zero-argument form of it exists.
    """

    value: Any = inspect.Parameter.empty
    factory: Any = None
    field_info: Any = None
    takes_self: bool = False


@dataclass(frozen=True, slots=True)
class InitParam:
    """One named constructor parameter, as the runtime sees it.

    Attributes:
        name: Parameter name (for pydantic models: the field name).
        parameter: The ``inspect.Parameter`` (kind, default, annotation *as
            declared*).  Pydantic fields get a synthesized keyword parameter
            whose default is the field default, ``HAS_DEFAULT_FACTORY`` for
            a ``default_factory`` field, or ``inspect.Parameter.empty``.
            Use :meth:`declared_default` for the real default.
        declared_on: The class whose namespace the annotation belongs to:
            a hand-written ``__init__``'s class, the dataclass that declares
            the field (found by identity of ``__dataclass_fields__`` entries),
            the attrs class that annotates the attribute, or the model class
            that annotates the field.
        init_owner: The class whose constructor accepts the parameter
            (the ``__init__`` definer at this chain level; for pydantic
            fields the model / pydantic dataclass).
        source: ``"init"`` (hand-written ``__init__``), ``"dataclass"``
            (stdlib dataclass-generated), ``"generated"`` (another code
            generator, e.g. attrs), ``"model"`` (pydantic ``model_fields``)
            or ``"pydantic_dataclass"``.
        level: 0 for the class's own constructor, n for the n-th class
            reached through ``**kwargs`` forwarding.
        field: The ``dataclasses.Field``, attrs ``Attribute`` or pydantic
            ``FieldInfo`` behind the parameter (for real defaults /
            factories), else ``None``.
        function: The hand-written ``__init__`` (unwrapped) for
            ``source == "init"``.
        attribute: The attribute the parameter sets when its name differs
            from the parameter's: an attrs field whose ``__init__`` argument
            is an alias (``_budget`` -> ``budget``, ``alias="count"``).  The
            annotation is declared under this name.  ``None`` otherwise.
    """

    name: str
    parameter: inspect.Parameter
    declared_on: type
    init_owner: type
    source: ParamSource
    level: int = 0
    field: Any = dataclasses.field(default=None, repr=False, compare=False)
    function: Any = dataclasses.field(default=None, repr=False, compare=False)
    attribute: str | None = None

    @property
    def default(self) -> Any:
        """The signature default (``inspect.Parameter.empty`` when required).

        May be a sentinel (dataclass ``<factory>``, attrs ``NOTHING``); use
        :meth:`declared_default` for the real default.
        """
        return self.parameter.default

    @property
    def keyword(self) -> bool:
        """True when the parameter can be passed by keyword."""
        return self.parameter.kind is not inspect.Parameter.POSITIONAL_ONLY

    def declared_default(self) -> DeclaredDefault:
        """The default the parameter itself declares, read from its real source.

        The ``dataclasses.Field`` (``default`` / ``default_factory``), the
        attrs ``Attribute`` (``default`` / ``Factory``), the pydantic
        ``FieldInfo`` (returned as :attr:`DeclaredDefault.field_info`) or the
        signature default of a hand-written ``__init__`` -- never the
        signature sentinel standing in for it.  ``Field(...)`` metadata in the
        *annotation* is not considered here (it needs the resolved
        annotation).
        """
        info = self.field
        if self.source in ("model", "pydantic_dataclass") and _is_field_info(info):
            return DeclaredDefault(field_info=info)
        if isinstance(info, dataclasses.Field):
            if info.default_factory is not dataclasses.MISSING:
                return DeclaredDefault(factory=info.default_factory)
            if info.default is dataclasses.MISSING:
                return DeclaredDefault()
            return _declared_value(info.default)
        if self.source == "generated" and info is not None and hasattr(info, "default"):
            value = info.default
            if type(value).__name__ == "Factory" and hasattr(value, "factory"):
                if getattr(value, "takes_self", False):
                    return DeclaredDefault(takes_self=True)
                return DeclaredDefault(factory=value.factory)
            return _declared_value(value)
        return _declared_value(self.parameter.default)

    def annotation_source(self) -> tuple[Any, _ann.AnnotationContext]:
        """The declared annotation in its best evaluable form, and its scope.

        Hand-written ``__init__`` parameters resolve in the function's
        closure, ``__globals__`` and builtins (the class namespace only as a
        last resort); everything else in the module of :attr:`declared_on`,
        under the :attr:`attribute` name when the parameter is an alias.  On
        Python 3.14 the annotation is the native value where it evaluates,
        else its source text.
        """
        if self.source == "init" and self.function is not None:
            annotations, closure = _ann.function_annotation_source(self.function)
            raw = annotations.get(self.name, self.parameter.annotation)
            return raw, _ann.function_context(self.function, self.init_owner, closure=closure)
        annotations, closure = _ann.class_annotation_source(self.declared_on)
        raw = annotations.get(self.attribute or self.name, self.parameter.annotation)
        return raw, _ann.class_context(self.declared_on, closure=closure)

    def resolve(self) -> _ann.ResolvedAnnotation:
        """Resolve the annotation in its declaring scope (never raises; see
        :func:`conscribe.config._annotations.resolve_annotation`)."""
        raw, context = self.annotation_source()
        return _ann.resolve_annotation(raw, context, name=self.name)


@dataclass(frozen=True)
class InitChain:
    """Everything ``cls(...)`` accepts, found by :func:`walk_init_chain`.

    Attributes:
        cls: The class that was inspected.
        kind: ``"init"`` / ``"dataclass"`` / ``"generated"`` (constructor of
            the definer), ``"model"`` (pydantic fast path),
            ``"pydantic_dataclass"`` or ``"none"`` (no ``__init__`` but
            ``object``'s: the constructor takes no arguments).
        definer: The class whose ``__init__`` runs first (``None`` for
            ``"none"``; for ``"model"`` the model class whose ``__init__``
            forwards, or ``pydantic.BaseModel``).
        params: Every named parameter, child first, unique by name (the
            nearest declaration wins).  Includes positional-only ones.
        var_keyword: The constructor takes arbitrary keywords by signature
            (the definer's ``**kwargs``; always true for pydantic models and
            dataclasses, whose own ``extra`` setting decides what happens).
        var_positional: The definer's ``__init__`` takes ``*args``.
        fully_resolved: True when the ``**kwargs`` chain was followed to its
            natural end, so :attr:`params` lists everything the constructor
            accepts by name.
        stop_reason: Why the walk stopped early (``"scope"``, ``"depth"``,
            ``"unreadable"``, ``"swallowing_placeholder"``), else ``None``.
        stopped_at: The class at which it stopped early.
        unreadable: Classes whose ``__init__`` signature could not be read.
        definers: Classes that contributed at least one parameter, in walk
            order.
        terminal: The class whose constructor ended a ``**kwargs`` walk
            naturally: ``pydantic.BaseModel`` (the leaf model's fields), a
            pydantic dataclass, or a class whose ``__init__`` takes no
            ``**kwargs``.  ``None`` when the walk reached ``object``, was
            cut, or the constructor forwards nothing.
        declarations: Every named parameter actually read, in walk order,
            including declarations shadowed by a nearer constructor. Unlike
            :attr:`params`, names are not deduplicated. Nothing beyond an
            early stop or a non-forwarding constructor is inferred.
    """

    cls: type
    kind: ChainKind
    definer: type | None
    params: tuple[InitParam, ...] = ()
    var_keyword: bool = False
    var_positional: bool = False
    fully_resolved: bool = True
    stop_reason: StopReason | None = None
    stopped_at: type | None = None
    unreadable: tuple[type, ...] = ()
    definers: tuple[type, ...] = ()
    terminal: type | None = None
    declarations: tuple[InitParam, ...] = ()

    def names(self) -> tuple[str, ...]:
        """Names the constructor accepts *by keyword*, child first."""
        return tuple(p.name for p in self.params if p.keyword)

    def get(self, name: str) -> InitParam | None:
        """The parameter called *name*, or ``None``."""
        for param in self.params:
            if param.name == name:
                return param
        return None


@dataclass(frozen=True)
class MROCollectionResult:
    """Result of collecting parameters from the MRO chain.

    Attributes:
        params: Parent-class parameters collected (child-first order, deduped).
        init_definers: Classes that contributed parameters.
        hints: Resolved annotations of :attr:`params`, each evaluated in its
            declaring class's namespace (unresolvable ones stay raw strings).
        fully_resolved: True if the chain terminated naturally (a parent
            had no ``**kwargs``), False if truncated by scope/depth or by an
            ``__init__`` whose signature cannot be read.
        declared_on: ``{param name: declaring class}`` for :attr:`params`.
        unreadable: Classes whose ``__init__`` signature could not be read.
        stop_reason: Why the walk stopped early, if it did (see
            :class:`InitChain`).
        init_params: :attr:`params` with their full provenance.
    """

    params: list[inspect.Parameter] = dataclasses.field(default_factory=list)
    init_definers: list[type] = dataclasses.field(default_factory=list)
    hints: dict[str, Any] = dataclasses.field(default_factory=dict)
    fully_resolved: bool = True
    declared_on: dict[str, type] = dataclasses.field(default_factory=dict)
    unreadable: tuple[type, ...] = ()
    stop_reason: str | None = None
    init_params: tuple[InitParam, ...] = ()


# ---------------------------------------------------------------------------
# Scope / depth settings
# ---------------------------------------------------------------------------

#: The string values an MRO scope accepts (besides a list of package names).
MRO_SCOPES: tuple[str, ...] = ("local", "third_party", "all")


def mro_scope_problem(value: Any) -> str | None:
    """Why *value* is not a valid MRO scope, or ``None`` when it is.

    Valid: ``"local"``, ``"third_party"``, ``"all"``, or a list / tuple /
    set of non-empty package-name strings.
    """
    if isinstance(value, str):
        if value in MRO_SCOPES:
            return None
        close = difflib.get_close_matches(value, MRO_SCOPES, n=1)
        hint = f" (did you mean {close[0]!r}?)" if close else ""
        return (
            f"unknown MRO scope {value!r}{hint}; use 'local', 'third_party', 'all' or a "
            f"list of package names"
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        bad = [item for item in value if not (isinstance(item, str) and item)]
        if bad:
            return (
                f"an MRO scope list holds package names (non-empty strings); got {bad!r} "
                f"in {value!r}"
            )
        return None
    return (
        f"an MRO scope is 'local', 'third_party', 'all' or a list of package names, "
        f"got {type(value).__name__} {value!r}"
    )


def mro_depth_problem(value: Any) -> str | None:
    """Why *value* is not a valid MRO depth (``None`` or an int >= 0), or ``None``."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        return (
            f"an MRO depth is a non-negative int or None, got {type(value).__name__} "
            f"{value!r}"
        )
    if value < 0:
        return f"an MRO depth is a non-negative int or None, got {value}"
    return None


# ---------------------------------------------------------------------------
# Scope classification
# ---------------------------------------------------------------------------


def classify_class_scope(cls: type) -> Literal["local", "third_party", "stdlib"]:
    """Classify a class as local, third-party, or stdlib.

    Uses ``inspect.getfile()`` and ``sysconfig.get_paths()`` to
    determine origin.  Classes created by ``LayerRegistrar.bridge()`` are
    ``"local"`` whatever the install location of conscribe itself.  A class
    without a source file is ``"stdlib"`` only when its module is a builtin
    or standard-library module; classes of dynamically created modules are
    ``"local"``.
    """
    if is_bridge_class(cls):
        return "local"
    try:
        source_file = inspect.getfile(cls)
    except (TypeError, OSError):
        return "stdlib" if _is_stdlib_module(getattr(cls, "__module__", None)) else "local"

    # Check site-packages FIRST (more specific than stdlib paths,
    # which may be a prefix of site-packages dirs).
    if "site-packages" in source_file or "dist-packages" in source_file:
        return "third_party"

    # Check stdlib paths
    stdlib_paths = _get_stdlib_paths()
    for p in stdlib_paths:
        if source_file.startswith(p):
            return "stdlib"

    return "local"


def _is_stdlib_module(module_name: Any) -> bool:
    if not isinstance(module_name, str) or not module_name:
        return True
    top = module_name.split(".", 1)[0]
    return top in sys.stdlib_module_names or top in sys.builtin_module_names


# ---------------------------------------------------------------------------
# The walker
# ---------------------------------------------------------------------------


def walk_init_chain(
    cls: type,
    *,
    scope: MROScope | None = None,
    depth: int | None = None,
) -> InitChain:
    """Find every parameter ``cls(...)`` accepts (see the module docstring).

    Pure reflection: signatures are read with
    :func:`~conscribe.config._annotations.safe_signature` and annotations are
    never resolved (on 3.14 the declared annotation of a pydantic field is
    read in ``Format.FORWARDREF``, which cannot raise), so it is safe at class
    creation time.  Nothing is imported; only per-function readings are
    cached (never per class), so the answer reflects the class at call time (inside a metaclass ``__new__`` a ``@dataclass``
    decorator has not run yet).  Use :meth:`InitParam.resolve` to evaluate a
    parameter's annotation.

    Args:
        cls: The class to inspect.
        scope: ``None`` (default) follows the chain wherever it leads (the
            runtime view).  Any :data:`MROScope` value cuts the walk at the
            first parent ``__init__`` outside that scope (the config view).
        depth: Maximum number of parent ``__init__`` definers to visit
            (``None`` = unlimited, ``0`` = do not follow ``**kwargs``).

    Returns:
        An :class:`InitChain`.

    Raises:
        ValueError: *scope* or *depth* is not a valid setting (see
            :func:`mro_scope_problem` / :func:`mro_depth_problem`).
    """
    if scope is not None:
        problem = mro_scope_problem(scope)
        if problem is not None:
            raise ValueError(problem)
    if depth is not None:
        problem = mro_depth_problem(depth)
        if problem is not None:
            raise ValueError(problem)
    if is_pydantic_model(cls) and model_init_only_forwards(cls):
        definer = constructor_definer(cls) or BaseModel
        params = tuple(_model_params(cls, level=0, seen=set()))
        return InitChain(
            cls=cls,
            kind="model",
            definer=definer,
            params=params,
            declarations=params,
            var_keyword=True,
            definers=(cls,) if params else (),
        )

    definer = constructor_definer(cls)
    if definer is None:
        return InitChain(cls=cls, kind="none", definer=None)

    if is_pydantic_dataclass(definer):
        params = tuple(_pydantic_dataclass_params(definer, level=0, seen=set()))
        return InitChain(
            cls=cls,
            kind="pydantic_dataclass",
            definer=definer,
            params=params,
            declarations=params,
            var_keyword=True,
            definers=(definer,) if params else (),
        )

    init = own_init(definer)
    if is_swallowing_placeholder(init):
        return InitChain(
            cls=cls,
            kind="init",
            definer=definer,
            var_keyword=True,
            var_positional=True,
            fully_resolved=False,
            stop_reason="swallowing_placeholder",
            stopped_at=definer,
        )
    reading = _read_init(definer, init)
    if reading is None:
        return InitChain(
            cls=cls,
            kind="init",
            definer=definer,
            fully_resolved=False,
            stop_reason="unreadable",
            stopped_at=definer,
            unreadable=(definer,),
        )

    kind: ChainKind = reading.kind
    params: list[InitParam] = list(reading.params(0))
    declarations = list(params)
    seen = {p.name for p in params}
    definers: list[type] = [definer] if params else []
    var_keyword = reading.var_keyword
    var_positional = reading.var_positional

    fully_resolved = True
    stop_reason: StopReason | None = None
    stopped_at: type | None = None
    unreadable: list[type] = []
    terminal: type | None = None

    if var_keyword:
        mro = cls.__mro__
        try:
            start = mro.index(definer)
        except ValueError:  # pragma: no cover - definer always comes from cls.__mro__
            start = len(mro)
        visited = 0
        for klass in mro[start + 1:]:
            if klass is object:
                break  # natural end: object.__init__ takes no keywords
            parent_init = own_init(klass)
            if parent_init is None:
                continue  # init-less classes never cut the chain
            # pydantic's constructors are the natural end of the chain
            # wherever pydantic is installed -- they validate fields and
            # forward nothing -- so they come before the scope / depth cut.
            if _is_base_model(klass):
                declared = _model_params(cls, level=visited + 1, seen=set())
                declarations.extend(declared)
                added = [p for p in declared if p.name not in seen]
                params.extend(added)
                seen.update(p.name for p in added)
                if added:
                    definers.append(klass)
                terminal = klass
                break
            if is_pydantic_dataclass(klass):
                declared = _pydantic_dataclass_params(klass, level=visited + 1, seen=set())
                declarations.extend(declared)
                added = [p for p in declared if p.name not in seen]
                params.extend(added)
                seen.update(p.name for p in added)
                if added:
                    definers.append(klass)
                terminal = klass
                break
            if scope is not None and not _should_include_class(klass, scope):
                fully_resolved, stop_reason, stopped_at = False, "scope", klass
                break
            if depth is not None and visited >= depth:
                fully_resolved, stop_reason, stopped_at = False, "depth", klass
                break
            visited += 1
            if is_swallowing_placeholder(parent_init):
                fully_resolved, stop_reason, stopped_at = False, "swallowing_placeholder", klass
                break
            parent = _read_init(klass, parent_init)
            if parent is None:
                fully_resolved, stop_reason, stopped_at = False, "unreadable", klass
                unreadable.append(klass)
                break

            declared = list(parent.params(visited))
            declarations.extend(declared)
            added = [p for p in declared if p.name not in seen]
            params.extend(added)
            seen.update(p.name for p in added)
            if added:
                definers.append(klass)
            if not parent.var_keyword:
                terminal = klass
                break  # natural end of the forwarding chain

    return InitChain(
        cls=cls,
        kind=kind,
        definer=definer,
        params=tuple(params),
        var_keyword=var_keyword,
        var_positional=var_positional,
        fully_resolved=fully_resolved,
        stop_reason=stop_reason,
        stopped_at=stopped_at,
        unreadable=tuple(unreadable),
        definers=tuple(definers),
        terminal=terminal,
        declarations=tuple(declarations),
    )


def keyword_param_names(cls: type) -> tuple[str, ...]:
    """``walk_init_chain(cls).names()``, faster for the common case.

    Constructor chains made only of plain functions (hand-written or
    dataclass / attrs generated, without ``__wrapped__`` / ``__signature__``)
    are read from the functions' code objects -- no ``inspect.Signature``, no
    :class:`InitParam`.  Anything else (pydantic models and dataclasses,
    Protocol placeholders, decorated or builtin ``__init__`` methods) takes
    the full :func:`walk_init_chain`, so the answer is always the same.
    """
    fast = _fast_keyword_names(cls)
    return fast if fast is not None else walk_init_chain(cls).names()


def _fast_keyword_names(cls: type) -> tuple[str, ...] | None:
    if is_pydantic_model(cls):
        return None
    definer = constructor_definer(cls)
    if definer is None:
        return ()
    mro = cls.__mro__
    names: list[str] = []
    seen: set[str] = set()
    for klass in mro[mro.index(definer):]:
        if klass is object:
            break
        init = own_init(klass)
        if init is None:
            continue
        read = _code_param_names(klass, init)
        if read is None:
            return None
        named, posonly, var_keyword = read
        for name in named:
            if name not in seen:
                seen.add(name)  # positional-only names shadow parents too
                if name not in posonly:
                    names.append(name)
        if not var_keyword:
            break
    return tuple(names)


def _code_param_names(
    klass: type, init: Any
) -> tuple[tuple[str, ...], tuple[str, ...], bool] | None:
    """``(named parameters, positional-only ones, takes **kwargs)`` of a plain
    ``__init__`` read from its code object; ``None`` when the full walk is
    needed."""
    if type(init) is not types.FunctionType or is_pydantic_dataclass(klass) or klass is BaseModel:
        return None
    if is_swallowing_placeholder(init):
        return None
    extra = init.__dict__
    if "__wrapped__" in extra or "__signature__" in extra or "__text_signature__" in extra:
        return None
    code = init.__code__
    positional = code.co_varnames[: code.co_argcount]
    keyword_only = code.co_varnames[code.co_argcount : code.co_argcount + code.co_kwonlyargcount]
    # The instance parameter is the first positional one (as named_params()).
    start = 1 if positional else 0
    posonly = positional[start : code.co_posonlyargcount]
    return positional[start:] + keyword_only, posonly, bool(code.co_flags & inspect.CO_VARKEYWORDS)


def collect_mro_params(
    cls: type,
    scope: MROScope = "local",
    depth: Union[int, None] = None,
) -> MROCollectionResult:
    """Collect parameters from parent classes along the MRO ``**kwargs`` chain.

    The config view of :func:`walk_init_chain` (see the module docstring for
    the walk rules), reported relative to the 1.x init definer
    (:func:`~conscribe.config._utils.find_init_definer`): when that definer's
    ``__init__`` accepts ``**kwargs``, :attr:`MROCollectionResult.params` is
    every parameter of the chain that the definer's own signature does not
    already declare.  Parameters already seen (by name) are skipped — child
    wins.

    Args:
        cls: The class whose MRO to walk.
        scope: Which classes to include.  String values ``"local"``,
            ``"third_party"``, or ``"all"`` behave as before.  A
            ``list[str]`` of package names includes local classes plus
            only third-party classes whose top-level package name is in
            the list (e.g., ``["pydantic", "my_sdk"]``).  Stdlib is
            always excluded when a list is used.
        depth: Maximum number of MRO levels to traverse.  ``None`` means
            unlimited.  ``0`` disables MRO traversal entirely.

    Returns:
        An ``MROCollectionResult`` with collected parameters and metadata.
    """
    if depth is not None and depth == 0:
        return MROCollectionResult()

    legacy_definer = find_init_definer(cls)
    if legacy_definer is None:
        return MROCollectionResult()
    legacy_sig = _ann.safe_signature(legacy_definer.__dict__["__init__"])
    if legacy_sig is None:
        return MROCollectionResult(fully_resolved=False, unreadable=(legacy_definer,))
    if not _has_kind(legacy_sig, inspect.Parameter.VAR_KEYWORD):
        return MROCollectionResult()

    chain = walk_init_chain(cls, scope=scope, depth=depth)
    own_names = {p.name for p in named_params(legacy_sig)}
    contributed = [p for p in chain.params if p.name not in own_names]

    hints: dict[str, Any] = {}
    for param in contributed:
        resolved = param.resolve()
        if resolved.status == "resolved":
            hints[param.name] = resolved.value
        elif resolved.status != "unannotated":
            hints[param.name] = resolved.raw

    definers: list[type] = []
    for param in contributed:
        if param.init_owner is not legacy_definer and param.init_owner not in definers:
            definers.append(param.init_owner)

    return MROCollectionResult(
        params=[p.parameter for p in contributed],
        init_definers=definers,
        hints=hints,
        fully_resolved=chain.fully_resolved,
        declared_on={p.name: p.declared_on for p in contributed},
        unreadable=chain.unreadable,
        stop_reason=chain.stop_reason,
        init_params=tuple(contributed),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _has_kind(sig: inspect.Signature, kind: Any) -> bool:
    return any(p.kind is kind for p in sig.parameters.values())


class _InitReading:
    """What one ``__init__`` of one class says, read once.

    Cached per function object (``_READINGS``) while the function's signature
    is cached (see :func:`conscribe.config._annotations.safe_signature`): a
    replaced ``__init__`` is a different function, and an entry whose
    function was mutated (code, defaults...) or that now belongs to another
    definer is read again.  The ``InitParam`` tuples it hands out are shared
    and immutable.
    """

    __slots__ = ("key", "definer", "init", "sig", "kind", "var_keyword", "var_positional", "levels")

    def __init__(self, definer: type, init: Any, sig: inspect.Signature, key: Any) -> None:
        self.key = key
        self.definer = definer
        self.init = init
        self.sig = sig
        generated = is_generated_init(definer, init)
        if generated and "__dataclass_fields__" in definer.__dict__:
            self.kind: ChainKind = "dataclass"
        elif generated:
            self.kind = "generated"
        else:
            self.kind = "init"
        self.var_keyword = _has_kind(sig, inspect.Parameter.VAR_KEYWORD)
        self.var_positional = _has_kind(sig, inspect.Parameter.VAR_POSITIONAL)
        self.levels: dict[int, tuple[InitParam, ...]] = {}

    def params(self, level: int) -> tuple[InitParam, ...]:
        """The named parameters at chain *level* (built once per level)."""
        found = self.levels.get(level)
        if found is None:
            found = tuple(_signature_params(self.definer, self.init, self.sig, level=level))
            self.levels[level] = found
        return found


_READINGS: weakref.WeakKeyDictionary[Any, _InitReading] = weakref.WeakKeyDictionary()


def _read_init(definer: type, init: Any) -> _InitReading | None:
    """The (cached) reading of *definer*'s ``__init__``; ``None`` if unreadable."""
    cacheable = type(init) is types.FunctionType
    key = _ann._signature_key(init) if cacheable else None
    if cacheable:
        cached = _READINGS.get(init)
        if cached is not None and cached.definer is definer:
            entry = _ann._SIGNATURES.get(init)  # a reading lives only as long as its signature
            if (
                entry is not None
                and entry[1] is cached.sig
                and all(a is b for a, b in zip(cached.key, key))  # type: ignore[arg-type]
            ):
                return cached
    sig = _ann.safe_signature(init)
    if sig is None:
        return None
    reading = _InitReading(definer, init, sig, key)
    entry = _ann._SIGNATURES.get(init) if cacheable else None
    if entry is not None and entry[1] is sig:  # only while the signature itself is cached
        _READINGS[init] = reading
    return reading


def _is_base_model(klass: type) -> bool:
    return klass is BaseModel


def _signature_params(
    definer: type, init: Any, sig: inspect.Signature, *, level: int
) -> list[InitParam]:
    """``InitParam`` objects for the named parameters of *definer*'s ``__init__``."""
    params: list[InitParam] = []
    if is_generated_init(definer, init):
        fields = definer.__dict__.get("__dataclass_fields__")
        aliases = _attrs_init_aliases(definer)
        for p in named_params(sig):
            if isinstance(fields, dict) and p.name in fields:
                owner = (
                    dataclass_field_owner(definer, p.name)
                    or _ann.annotation_owner(definer, p.name)
                    or definer
                )
                params.append(
                    InitParam(p.name, p, owner, definer, "dataclass", level, field=fields[p.name])
                )
            else:
                attribute = aliases.get(p.name)
                attr_name = getattr(attribute, "name", None) or p.name
                owner = _ann.annotation_owner(definer, attr_name) or definer
                params.append(
                    InitParam(
                        p.name, p, owner, definer, "generated", level, field=attribute,
                        attribute=attr_name if attr_name != p.name else None,
                    )
                )
        return params
    function = _ann.unwrap_function(init)
    for p in named_params(sig):
        params.append(InitParam(p.name, p, definer, definer, "init", level, function=function))
    return params


def _attrs_init_aliases(klass: type) -> dict[str, Any]:
    """``{__init__ argument name: attrs Attribute}`` of an attrs class.

    attrs passes a field to ``__init__`` under its alias: the explicit
    ``alias=`` or, for a private attribute, the name without its leading
    underscores (``_budget`` -> ``budget``; older attrs versions have no
    ``Attribute.alias`` and always strip).  Empty for non-attrs classes.
    """
    attributes = getattr(klass, "__attrs_attrs__", None)
    if not attributes:
        return {}
    aliases: dict[str, Any] = {}
    for attribute in attributes:
        if getattr(attribute, "init", True) is False:
            continue
        name = getattr(attribute, "name", None)
        if not isinstance(name, str):
            continue
        alias = getattr(attribute, "alias", None)
        aliases[alias if isinstance(alias, str) and alias else name.lstrip("_")] = attribute
    return aliases


def _field_default(info: Any) -> Any:
    """Parameter default for a pydantic ``FieldInfo``."""
    from pydantic_core import PydanticUndefined

    if getattr(info, "default_factory", None) is not None:
        return HAS_DEFAULT_FACTORY
    default = getattr(info, "default", PydanticUndefined)
    if default is PydanticUndefined or default is Ellipsis:
        return inspect.Parameter.empty
    return default


def _model_params(model: type, *, level: int, seen: set[str]) -> list[InitParam]:
    """Keyword parameters for the fields of a pydantic model."""
    result: list[InitParam] = []
    for name, info in model.model_fields.items():
        if name in seen or getattr(info, "init", None) is False:
            continue
        owner = _ann.annotation_owner(model, name) or model
        raw = _ann.own_annotations(owner).get(name, info.annotation)
        parameter = inspect.Parameter(
            name,
            inspect.Parameter.KEYWORD_ONLY,
            default=_field_default(info),
            annotation=raw,
        )
        result.append(InitParam(name, parameter, owner, model, "model", level, field=info))
    return result


def _pydantic_dataclass_params(klass: type, *, level: int, seen: set[str]) -> list[InitParam]:
    """Parameters for the fields of a pydantic dataclass."""
    result: list[InitParam] = []
    infos = getattr(klass, "__pydantic_fields__", None)
    if isinstance(infos, dict) and infos:
        items = [(name, info, getattr(info, "kw_only", None)) for name, info in infos.items()]
    else:  # not complete yet: fall back to the stdlib dataclass fields
        items = [
            (f.name, f, f.kw_only)
            for f in dataclasses.fields(klass)
        ]
    for name, info, kw_only in items:
        if name in seen or getattr(info, "init", None) is False:
            continue
        owner = (
            dataclass_field_owner(klass, name)
            or _ann.annotation_owner(klass, name)
            or klass
        )
        raw = _ann.own_annotations(owner).get(name, getattr(info, "annotation", inspect.Parameter.empty))
        if isinstance(info, dataclasses.Field):
            if info.default_factory is not dataclasses.MISSING:
                default = HAS_DEFAULT_FACTORY
            elif info.default is not dataclasses.MISSING:
                default = info.default
            else:
                default = inspect.Parameter.empty
        else:
            default = _field_default(info)
        kind = (
            inspect.Parameter.KEYWORD_ONLY
            if kw_only is True
            else inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        parameter = inspect.Parameter(name, kind, default=default, annotation=raw)
        result.append(
            InitParam(name, parameter, owner, klass, "pydantic_dataclass", level, field=info)
        )
    return result


def _extract_package_name(cls: type) -> Union[str, None]:
    """Extract the top-level package name for a third-party class.

    Uses ``inspect.getfile()`` to locate the source file, then looks for
    ``"site-packages"`` in the path components.  The component immediately
    after ``site-packages`` is taken as the top-level package name (with
    ``.py`` stripped for single-file packages).

    Returns:
        The top-level package name (e.g., ``"pydantic"``), or ``None``
        if the class is not from site-packages or has no source file.
    """
    try:
        source_file = inspect.getfile(cls)
    except (TypeError, OSError):
        return None

    parts = pathlib.PurePath(source_file).parts
    for marker in ("site-packages", "dist-packages"):
        if marker in parts:
            idx = parts.index(marker)
            break
    else:
        return None

    if idx + 1 >= len(parts):
        return None

    pkg = parts[idx + 1]
    # Single-file packages: e.g., "six.py" → "six"
    if pkg.endswith(".py"):
        pkg = pkg[:-3]
    return pkg


def _should_include_class(cls: type, scope: MROScope) -> bool:
    """Check whether a class should be included given the scope."""
    # String scopes: existing logic unchanged
    if isinstance(scope, str):
        if scope == "all":
            return True

        cls_scope = classify_class_scope(cls)

        if cls_scope == "stdlib":
            return False

        if scope == "local":
            return cls_scope == "local"

        # scope == "third_party": include local + third_party
        return cls_scope in ("local", "third_party")

    # List scope: local + specified packages only
    cls_scope = classify_class_scope(cls)
    if cls_scope == "local":
        return True
    if cls_scope == "stdlib":
        return False
    # third_party: check package name
    pkg = _extract_package_name(cls)
    return pkg is not None and pkg in scope


_stdlib_paths_cache: Union[list[str], None] = None


def _get_stdlib_paths() -> list[str]:
    """Get stdlib directory paths, cached for performance."""
    global _stdlib_paths_cache  # noqa: PLW0603
    if _stdlib_paths_cache is None:
        paths = sysconfig.get_paths()
        _stdlib_paths_cache = [
            p
            for key in ("stdlib", "platstdlib", "purelib")
            if (p := paths.get(key))
            and "site-packages" not in p
        ]
    return _stdlib_paths_cache
