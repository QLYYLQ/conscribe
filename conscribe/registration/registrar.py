"""LayerRegistrar — unified entry point for layer registration.

Binds Protocol + Registry + AutoRegistrar + KeyTransform into a single
API surface. Provides three registration paths:

- Path A (inheritance): ``class Foo(Base)`` via metaclass — auto-registered.
- Path B (bridge): ``bridge(ExternalClass)`` — one-time bridge, subclasses auto-register.
- Path C (manual): ``@register("key")`` — decorator with protocol check.

Every path ends in the same tail (``conscribe.registration.auto._register_class``):
it grammar-checks the class's merged ``__wiring__``
(``conscribe.wiring.check_wiring_declaration``) before the first
``registry.add``, so a malformed declaration fails at class creation, then
adds, stamps and installs the ``WiredField`` placeholders.  Keys are
validated before anything is registered.  Registration is atomic on every
path: a failure after the class was added to a registry leaves no binding
behind (see ``auto.py`` for Path A/B; ``register()`` undoes exactly its own
work, see its docstring).
"""
from __future__ import annotations

from typing import (
    Any,
    Callable,
    Generic,
    TypeVar,
)

from conscribe.exceptions import InvalidRegistryKeyError
from conscribe.registration import registry as _registry_mod
from conscribe.registration.auto import (
    REGISTRY_ORIGIN_ATTR,
    create_auto_registrar,
    _register_class,
    _registry_keys,
    _transformed_key,
)
from conscribe.registration.filters import (
    RegistrationContext,
    build_filter_chain,
    should_skip_registration,
)
from conscribe.registration.key_transform import (
    KeyTransform,
    default_key_transform,
    make_key_transform,
)
from conscribe.registration.registry import (
    _MISSING,
    LayerRegistry,
    _purge_class,
    _rebind_recreated,
    _record,
)

P = TypeVar("P")


class LayerRegistrar(Generic[P]):
    """Layer-specific registrar. Created via ``create_registrar()``.

    All public methods are classmethods. Consumers call them on the class:
        ``MyRegistrar.get("foo")``
        ``MyRegistrar.bridge(ExtAgent)``

    Subclass attributes (filled by ``create_registrar``):
        protocol: The @runtime_checkable Protocol for this layer.
        _registry: The underlying LayerRegistry instance.
        Meta: The AutoRegistrar metaclass.
        _key_transform: Key inference function.
        discriminator_field: Config union discriminator field name (flat mode).
        discriminator_fields: Config union discriminator field names (nested mode).
        inline_only: Whether the layer is inline-only (its instances may only
            be written inside other layers' wiring receptors; no top-level
            section in a composed config).  ``False`` unless
            ``create_registrar(..., inline_only=True)``.
        _key_separator: Key separator for hierarchical keys.
        _skip_pydantic_generic: Whether to skip Pydantic Generic intermediates.
        _skip_filter: Optional custom skip filter callable.
    """

    protocol: type
    _registry: LayerRegistry
    Meta: type
    _key_transform: KeyTransform
    discriminator_field: str
    discriminator_fields: list[str] | None
    inline_only: bool = False
    _key_separator: str
    _mro_scope: str | list[str]
    _mro_depth: int | None
    _skip_pydantic_generic: bool
    _skip_filter: Callable[[type], bool] | None
    _filters: list

    # ── Query API ──

    @classmethod
    def get(cls, key: str) -> type:
        """Look up a registered class by key.

        Raises:
            KeyNotFoundError: If key is not registered.
        """
        return cls._registry.get(key)

    @classmethod
    def get_or_none(cls, key: str) -> type | None:
        """Safe lookup — returns None if key is not found."""
        return cls._registry.get_or_none(key)

    @classmethod
    def get_all(cls) -> dict[str, type]:
        """Return a snapshot dict of all {key: class} mappings."""
        return dict(cls._registry.items())

    @classmethod
    def keys(cls) -> list[str]:
        """Return all registered keys."""
        return cls._registry.keys()

    @classmethod
    def unregister(cls, key: str, *, missing_ok: bool = False) -> type | None:
        """Remove the registration of ``key`` and return the removed class.

        Only that key is removed: a multi-key class stays registered under
        its other keys, and the keys below a hierarchical prefix are separate
        entries.  For test isolation prefer
        ``conscribe.testing.isolated_registries()`` (or the
        ``conscribe_isolated`` pytest fixture), which also restores what a
        test removed and undoes registrars the test created.

        Args:
            key: The registration key.
            missing_ok: If True, an unregistered key is not an error and
                ``None`` is returned, so teardown code can run twice.

        Returns:
            The class that was registered under ``key``, or ``None`` when the
            key was absent and ``missing_ok`` is true.

        Raises:
            KeyNotFoundError: If key is not registered and ``missing_ok`` is
                false (a ``KeyError`` subclass).
        """
        return cls._registry.remove(key, missing_ok=missing_ok)

    # ── Hierarchical Query API ──

    @classmethod
    def children(cls, prefix: str) -> dict[str, type]:
        """Return all entries whose key starts with ``prefix + separator``.

        Only meaningful when ``key_separator`` is set.
        """
        return cls._registry.children(prefix)

    @classmethod
    def tree(cls) -> dict:
        """Return a nested dict representing the key hierarchy.

        Only meaningful when ``key_separator`` is set.
        """
        return cls._registry.tree()

    # ── Path B: bridge() ──

    @classmethod
    def bridge(cls, external_class: type, *, name: str | None = None) -> type:
        """Create a bridge base class for an external class.

        Automatically resolves metaclass conflicts. The returned bridge class
        is marked ``__abstract__=True`` and will NOT be registered.

        Args:
            external_class: External class (not using our metaclass).
            name: Bridge class name. Defaults to ``f"{external_class.__name__}Bridge"``.

        Returns:
            A bridge base class whose subclasses auto-register via metaclass.
        """
        bridge_name = name or f"{external_class.__name__}Bridge"
        ext_meta = type(external_class)

        # Metaclass conflict resolution (4 strategies, by priority)
        if ext_meta is type:
            # Strategy 1: plain class — most common
            meta = cls.Meta
        elif issubclass(cls.Meta, ext_meta):
            # Strategy 2: our Meta already satisfies ext_meta
            meta = cls.Meta
        elif issubclass(ext_meta, cls.Meta):
            # Strategy 3: ext_meta is more specific
            meta = ext_meta
        else:
            # Strategy 4: incompatible — create combined metaclass
            try:
                meta = type(f"{bridge_name}Meta", (cls.Meta, ext_meta), {})
            except TypeError as exc:
                raise TypeError(
                    f"Cannot create combined metaclass for bridge '{bridge_name}': "
                    f"{cls.Meta.__name__} and {ext_meta.__name__} are incompatible. "
                    f"Consider using base_metaclass= in create_registrar() to resolve "
                    f"the metaclass conflict."
                ) from exc

        return meta(bridge_name, (external_class,), {"__abstract__": True})

    # ── Path C: register() ──

    @classmethod
    def register(
        cls,
        key: str | None = None,
        *,
        propagate: bool = False,
    ) -> Callable[[type], type]:
        """Manual registration decorator. Always runs protocol_check.

        The key is validated and the class's merged ``__wiring__`` is
        grammar-checked first (``conscribe.wiring.check_wiring_declaration``,
        no registry lookups), so a bad key or a malformed declaration fails
        before anything changes.  Then the class is added, and the decorator
        stamps it and installs its ``WiredField`` injection placeholders.

        The registration is atomic in its own scope: if anything fails after
        the class was added (attribute writes, descriptor injection,
        installing the propagation hook), the binding this call made and
        every attribute it wrote are undone before the error propagates.
        Bindings the class has in other registries are left alone -- also
        those made by another ``@register`` stacked below this one on the
        same class statement: each decorator undoes only its own work, so
        when an outer ``@R2.register(...)`` fails, the inner
        ``@R1.register(...)`` stays registered (unregister it, or let
        ``conscribe.testing.isolated_registries()`` roll it back).

        Place ``@register(...)`` *above* ``@dataclass(slots=True)`` /
        ``attrs.define``: those decorators replace the class with a new one,
        and a registration applied to the replaced class is not moved over.

        Args:
            key: Registration key (a non-empty str). If None, inferred via
                _key_transform.
            propagate: If True, inject __init_subclass__ so that subclasses
                of the decorated class also auto-register (see
                :meth:`_inject_auto_registration`).

        Returns:
            A decorator that registers the target class and returns it.

        Raises:
            InvalidRegistryKeyError: If *key* (or the inferred key) is not a
                non-empty str (nothing is registered).
            WiringResolutionError: If the target class's merged
                ``__wiring__`` is malformed (nothing is registered).
            DuplicateKeyError: If key already exists (from registry.add).
            ProtocolViolationError: If target class fails Protocol check.
        """

        def decorator(target_cls: type) -> type:
            if key is not None:
                (actual_key,) = _single_key(key, target_cls)
            else:
                actual_key = _transformed_key(cls._key_transform, target_cls.__name__, target_cls)
            registry = cls._registry

            before = dict(target_cls.__dict__)
            try:
                # check -> add -> stamp -> inject; a malformed __wiring__
                # fails before the class touches the registry.
                _register_class(registry, target_cls, [actual_key], protocol_check=True)
                if propagate:
                    cls._inject_auto_registration(target_cls)
            except BaseException:
                registry._discard_binding(actual_key, target_cls)
                _restore_attrs(target_cls, before)
                raise

            if _registry_mod._ISOLATION_STACK:
                # isolated_registries() undoes these writes if the
                # registration is rolled back (target_cls may predate the
                # block).
                old, new = _attr_changes(target_cls, before)
                if old:
                    _record("attrs", target_cls, ((registry, actual_key),), old, new)
            return target_cls

        return decorator

    # ── Propagation implementation ──

    @classmethod
    def _inject_auto_registration(cls, target_cls: type) -> None:
        """Inject __init_subclass__ for definition-time auto-registration.

        Called when ``register(propagate=True)`` is used.  The hook

        - finds the class that holds it at call time (the class whose own
          ``__dict__`` contains this hook), so it keeps working after
          ``@dataclass(slots=True)`` / ``attrs.define`` re-created the
          decorated class;
        - recognises a subclass re-created from a registered subclass's
          namespace and rebinds its keys instead of registering it again
          (it reads the re-creation stamp *before* chaining to the hook it
          replaced, so two ``register(propagate=True)`` on one class -- one
          registry each -- both recognise the copy);
        - installs the ``WiredField`` placeholders of every subclass,
          registered or skipped by the filters (subclasses inherit them);
        - registers a subclass through the shared tail (key validation,
          ``check_wiring_declaration`` before the first add, stamp,
          placeholders), so a bad key or a malformed declaration fails the
          subclass statement; ``__registry_key__`` may be a list of keys, as
          with the registration metaclass;
        - purges the subclass from every live registry if registering it
          fails part-way.

        Limitation: the hook registers the subclass while ``type.__new__``
        runs ``__init_subclass__``; if another ``__init_subclass__`` in the
        MRO raises *after* this hook returned (a validating mixin placed
        before the decorated class), the class statement fails but the
        subclass stays registered -- nothing runs after that error to undo
        it.  Put such validation in the subclass's own hook before calling
        ``super().__init_subclass__()``, or unregister the key.

        Args:
            target_cls: The class to inject __init_subclass__ into.
        """
        registry = cls._registry
        kt = cls._key_transform
        path_c_filters = tuple(cls._filters)

        # Save the ORIGINAL __init_subclass__ from this class's own dict
        # (NOT inherited). Use __dict__.get(), NOT getattr().
        original = target_cls.__dict__.get("__init_subclass__")

        def __init_subclass__(sub_cls: type, **kwargs: Any) -> None:
            # The re-creation stamp as the class statement wrote it: a hook
            # chained below (another registry's) re-stamps a recognised copy.
            origin = sub_cls.__dict__.get(REGISTRY_ORIGIN_ATTR)

            # Chain-call original hook
            if original is not None:
                getattr(original, "__func__", original)(sub_cls, **kwargs)
            else:
                owner = _hook_owner(sub_cls, hook, target_cls)
                super(owner, sub_cls).__init_subclass__(**kwargs)

            try:
                # Re-creation from a registered subclass's namespace
                if (
                    type(origin) is int
                    and origin != id(sub_cls)
                    and _rebind_recreated(origin, sub_cls, sub_cls.__dict__)
                ):
                    setattr(sub_cls, REGISTRY_ORIGIN_ATTR, id(sub_cls))
                    return

                from conscribe.wiring import inject_wired_descriptors

                # Build context and check filters
                ctx = RegistrationContext(
                    cls=sub_cls,
                    name=sub_cls.__name__,
                    bases=sub_cls.__bases__,
                    namespace=sub_cls.__dict__,  # type: ignore[arg-type]
                    registry_name=registry.name,
                )
                if should_skip_registration(path_c_filters, ctx):
                    inject_wired_descriptors(sub_cls)
                    return

                # Keys from the subclass's own __dict__, NOT getattr (avoid
                # inheriting the parent's key)
                explicit = sub_cls.__dict__.get("__registry_key__")
                if explicit is not None:
                    keys = _registry_keys(explicit, sub_cls, source="__registry_key__")
                else:
                    keys = [_transformed_key(kt, sub_cls.__name__, sub_cls)]
                _register_class(registry, sub_cls, keys, protocol_check=True)
            except BaseException:
                # The subclass statement fails: no registry may keep it.
                _purge_class(sub_cls)
                raise

        hook = classmethod(__init_subclass__)
        target_cls.__init_subclass__ = hook  # type: ignore[attr-defined]

    # ── Config API ──

    @classmethod
    def build_config(cls) -> Any:
        """Build discriminated union config from all registered classes.

        Returns:
            A ``LayerConfigResult`` with the union type and per-key models.
        """
        from conscribe.config.builder import build_layer_config

        return build_layer_config(cls)

    @classmethod
    def config_union_type(cls) -> type:
        """Get the discriminated union type for config validation.

        Returns:
            The union type (or single model if only one key).
        """
        return cls.build_config().union_type

    @classmethod
    def get_config_schema(cls, key: str) -> Any:
        """Get the config schema for a specific registered key.

        Args:
            key: The registration key.

        Returns:
            A Pydantic ``BaseModel`` subclass, or ``None`` if no
            extractable parameters.
        """
        from conscribe.config.extractor import extract_config_schema

        target_cls = cls._registry.get(key)
        return extract_config_schema(
            target_cls,
            mro_scope=cls._mro_scope,
            mro_depth=cls._mro_depth,
        )


def create_registrar(
    name: str,
    protocol: type,
    *,
    discriminator_field: str = "",
    discriminator_fields: list[str] | None = None,
    key_separator: str = "",
    strip_suffixes: list[str] | None = None,
    strip_prefixes: list[str] | None = None,
    key_transform: KeyTransform | None = None,
    base_metaclass: type = type,
    mro_scope: str | list[str] = "local",
    mro_depth: int | None = None,
    skip_pydantic_generic: bool = True,
    skip_filter: Callable[[type], bool] | None = None,
    inline_only: bool = False,
) -> type[LayerRegistrar[Any]]:
    """One-line factory to create a Layer-specific Registrar class.

    This is the recommended entry point.

    Args:
        name: Layer name (e.g. "agent", "llm", "provider").
        protocol: @runtime_checkable Protocol class for this layer.
        discriminator_field: Config union discriminator field name (flat mode).
        discriminator_fields: Config union discriminator field names (nested mode).
            Mutually exclusive with ``discriminator_field``. When set,
            ``key_separator`` must also be set.
        key_separator: Separator for hierarchical keys (e.g. ``"."``).
            Empty string means flat keys (backward compatible).
        strip_suffixes: Suffixes to strip during key inference.
        strip_prefixes: Prefixes to strip during key inference.
        key_transform: Fully custom key inference function (highest priority).
        base_metaclass: Parent metaclass for AutoRegistrar.
        mro_scope: Scope for MRO parameter collection.  Accepts
            ``"local"``, ``"third_party"``, ``"all"``, or a
            ``list[str]`` of package names (local + listed packages).
        mro_depth: Max MRO levels to traverse for parameter collection.
            ``None`` means unlimited.
        skip_pydantic_generic: If True (default), skip classes whose name
            contains ``[`` — these are Pydantic Generic intermediates
            (e.g. ``BaseModel[str]``) that should not pollute the
            registry.
        skip_filter: Optional callable ``(cls) -> bool``. If it returns
            True for a class, that class is skipped (not registered).
        inline_only: Declare the layer inline-only.  Its instances may only
            be written inside the wiring receptors of other layers: a
            composed config builds and emits the layer's models (receptors
            use them) but has no top-level section for it, and writing the
            layer name at top level is a validation error that names the
            receptors where the instances belong.  Exposed as
            ``Registrar.inline_only`` (and ``Registrar.Meta.inline_only``);
            registration itself is unaffected.

    Returns:
        A LayerRegistrar subclass with all class-level attributes populated.

    Raises:
        InvalidProtocolError: If protocol is not @runtime_checkable.
        ValueError: If ``discriminator_fields`` is set but ``key_separator``
            is empty, or if both ``discriminator_field`` and
            ``discriminator_fields`` are set.
        TypeError: If ``inline_only`` is not a bool.
    """
    # Validate mutually exclusive discriminator params
    if discriminator_fields and discriminator_field:
        raise ValueError(
            "Cannot set both discriminator_field and discriminator_fields. "
            "Use discriminator_field for flat mode, discriminator_fields for nested mode."
        )
    if discriminator_fields and not key_separator:
        raise ValueError(
            "discriminator_fields requires key_separator to be set "
            "(e.g. key_separator='.')."
        )
    if not isinstance(inline_only, bool):
        raise TypeError(
            f"create_registrar('{name}'): inline_only must be a bool, got "
            f"{type(inline_only).__name__} ({inline_only!r})."
        )

    # 1. Determine key transform
    if key_transform is not None:
        kt: KeyTransform = key_transform
    elif strip_suffixes or strip_prefixes:
        kt = make_key_transform(
            suffixes=strip_suffixes, prefixes=strip_prefixes
        )
    else:
        kt = default_key_transform  # type: ignore[assignment]

    # 2. Create registry
    registry = LayerRegistry(name, protocol, separator=key_separator)

    # 3. Create AutoRegistrar metaclass
    meta = create_auto_registrar(
        registry,
        kt,
        base_metaclass=base_metaclass,
        skip_pydantic_generic=skip_pydantic_generic,
        skip_filter=skip_filter,
        key_separator=key_separator,
        inline_only=inline_only,
    )

    # 4. Build filter chain (shared between Path A and Path C)
    filters = build_filter_chain(
        skip_pydantic_generic=skip_pydantic_generic,
        skip_filter=skip_filter,
        registry_name=name,
    )

    # 5. Dynamically create LayerRegistrar subclass
    registrar_cls = type(
        f"{name.title()}Registrar",
        (LayerRegistrar,),
        {
            "protocol": protocol,
            "_registry": registry,
            "Meta": meta,
            "_key_transform": staticmethod(kt),
            "discriminator_field": discriminator_field,
            "discriminator_fields": discriminator_fields,
            "inline_only": inline_only,
            "_key_separator": key_separator,
            "_mro_scope": mro_scope,
            "_mro_depth": mro_depth,
            "_skip_pydantic_generic": skip_pydantic_generic,
            "_skip_filter": skip_filter,
            "_filters": filters,
        },
    )
    return registrar_cls


# ── Helpers ──


def _single_key(key: object, cls: type) -> list[str]:
    """Validate the key given to ``register(key=...)`` (one key)."""
    if isinstance(key, list):
        raise InvalidRegistryKeyError(
            cls.__qualname__, key, "register(key=...)",
            "register() takes one key; to register a class under several keys, "
            "apply register() once per key, or set __registry_key__ = [...] on a "
            "class the registration metaclass creates",
        )
    return _registry_keys(key, cls, source="register(key=...)")


def _hook_owner(sub_cls: type, hook: object, fallback: type) -> type:
    """The class in ``sub_cls``'s MRO whose own ``__dict__`` holds ``hook``.

    A propagation hook is installed on one class, but a class decorator such
    as ``@dataclass(slots=True)`` may later replace that class with a copy
    whose ``__dict__`` holds the same hook object.  ``super()`` must start
    from whichever of the two ``sub_cls`` actually inherits from.
    """
    for klass in sub_cls.__mro__[1:]:
        if klass.__dict__.get("__init_subclass__") is hook:
            return klass
    return fallback  # pragma: no cover - only for a direct call on the owner itself


def _attr_changes(cls: type, before: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Attributes of ``cls`` whose value changed since ``before`` was taken.

    Returns ``(old, new)``: each changed name mapped to its value before and
    after (``_MISSING`` when absent).  Values are compared by identity.
    """
    after = dict(cls.__dict__)
    names = [
        name
        for name in [*before, *(n for n in after if n not in before)]
        if before.get(name, _MISSING) is not after.get(name, _MISSING)
    ]
    old = {name: before.get(name, _MISSING) for name in names}
    new = {name: after.get(name, _MISSING) for name in names}
    return old, new


def _restore_attrs(cls: type, before: dict[str, Any]) -> None:
    """Undo every attribute change made to ``cls`` since ``before`` was taken."""
    old, _ = _attr_changes(cls, before)
    _registry_mod._apply_attrs(cls, old)