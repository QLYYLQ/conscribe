"""LayerRegistrar — unified entry point for layer registration.

Binds Protocol + Registry + AutoRegistrar + KeyTransform into a single
API surface. Provides three registration paths:

- Path A (inheritance): ``class Foo(Base)`` via metaclass — auto-registered.
- Path B (bridge): ``bridge(ExternalClass)`` — one-time bridge, subclasses auto-register.
- Path C (manual): ``@register("key")`` — decorator with protocol check.
"""
from __future__ import annotations

from typing import (
    Any,
    Callable,
    Generic,
    Optional,
    TypeVar,
    Union,
)

from conscribe.registration.auto import create_auto_registrar
from conscribe.registration.key_transform import (
    KeyTransform,
    default_key_transform,
    make_key_transform,
)
from conscribe.registration.registry import LayerRegistry

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
        discriminator_field: Config union discriminator field name.
    """

    protocol: type
    _registry: LayerRegistry
    Meta: type
    _key_transform: KeyTransform
    discriminator_field: str
    _mro_scope: Union[str, list[str]]
    _mro_depth: Union[int, None]

    # ── Query API ──

    @classmethod
    def get(cls, key: str) -> type:
        """Look up a registered class by key.

        Raises:
            KeyNotFoundError: If key is not registered.
        """
        return cls._registry.get(key)

    @classmethod
    def get_or_none(cls, key: str) -> Optional[type]:
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
    def unregister(cls, key: str) -> None:
        """Remove a registration. Intended for test isolation."""
        cls._registry.remove(key)

    # ── Path B: bridge() ──

    @classmethod
    def bridge(cls, external_class: type, *, name: Optional[str] = None) -> type:
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
            meta = type(f"{bridge_name}Meta", (cls.Meta, ext_meta), {})

        return meta(bridge_name, (external_class,), {"__abstract__": True})

    # ── Path C: register() ──

    @classmethod
    def register(
        cls,
        key: Optional[str] = None,
        *,
        propagate: bool = False,
    ) -> Callable[[type], type]:
        """Manual registration decorator. Always runs protocol_check.

        Args:
            key: Registration key. If None, inferred via _key_transform.
            propagate: If True, inject __init_subclass__ so that subclasses
                of the decorated class also auto-register.

        Returns:
            A decorator that registers the target class and returns it.

        Raises:
            DuplicateKeyError: If key already exists (from registry.add).
            ProtocolViolationError: If target class fails Protocol check.
        """

        def decorator(target_cls: type) -> type:
            actual_key = key if key is not None else cls._key_transform(
                target_cls.__name__
            )
            cls._registry.add(actual_key, target_cls, protocol_check=True)
            target_cls.__registry_key__ = actual_key  # type: ignore[attr-defined]

            if propagate:
                cls._inject_auto_registration(target_cls)

            return target_cls

        return decorator

    # ── Propagation implementation ──

    @classmethod
    def _inject_auto_registration(cls, target_cls: type) -> None:
        """Inject __init_subclass__ for definition-time auto-registration.

        Called when ``register(propagate=True)`` is used.

        Args:
            target_cls: The class to inject __init_subclass__ into.
        """
        registry = cls._registry
        kt = cls._key_transform

        # Save the ORIGINAL __init_subclass__ from this class's own dict
        # (NOT inherited). Use __dict__.get(), NOT getattr().
        original = target_cls.__dict__.get("__init_subclass__")

        @classmethod  # type: ignore[misc]
        def __init_subclass__(sub_cls: type, **kwargs: Any) -> None:
            # Chain-call original hook
            if original:
                original.__func__(sub_cls, **kwargs)
            else:
                super(target_cls, sub_cls).__init_subclass__(**kwargs)

            # Skip abstract subclasses — use __dict__ NOT getattr
            # to avoid inheriting parent's __abstract__=True
            if sub_cls.__dict__.get("__abstract__", False):
                return

            # Infer key from own __dict__, NOT getattr (avoid inheriting parent key)
            sub_key = sub_cls.__dict__.get("__registry_key__") or kt(
                sub_cls.__name__
            )
            registry.add(sub_key, sub_cls, protocol_check=True)
            sub_cls.__registry_key__ = sub_key  # type: ignore[attr-defined]

        target_cls.__init_subclass__ = __init_subclass__  # type: ignore[attr-defined]

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
    strip_suffixes: Optional[list[str]] = None,
    strip_prefixes: Optional[list[str]] = None,
    key_transform: Optional[KeyTransform] = None,
    base_metaclass: type = type,
    mro_scope: Union[str, list[str]] = "local",
    mro_depth: Optional[int] = None,
) -> type:
    """One-line factory to create a Layer-specific Registrar class.

    This is the recommended entry point.

    Args:
        name: Layer name (e.g. "agent", "llm", "provider").
        protocol: @runtime_checkable Protocol class for this layer.
        discriminator_field: Config union discriminator field name.
        strip_suffixes: Suffixes to strip during key inference.
        strip_prefixes: Prefixes to strip during key inference.
        key_transform: Fully custom key inference function (highest priority).
        base_metaclass: Parent metaclass for AutoRegistrar.
        mro_scope: Scope for MRO parameter collection.  Accepts
            ``"local"``, ``"third_party"``, ``"all"``, or a
            ``list[str]`` of package names (local + listed packages).
        mro_depth: Max MRO levels to traverse for parameter collection.
            ``None`` means unlimited.

    Returns:
        A LayerRegistrar subclass with all class-level attributes populated.

    Raises:
        InvalidProtocolError: If protocol is not @runtime_checkable.
    """
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
    registry = LayerRegistry(name, protocol)

    # 3. Create AutoRegistrar metaclass
    meta = create_auto_registrar(registry, kt, base_metaclass=base_metaclass)

    # 4. Dynamically create LayerRegistrar subclass
    registrar_cls = type(
        f"{name.title()}Registrar",
        (LayerRegistrar,),
        {
            "protocol": protocol,
            "_registry": registry,
            "Meta": meta,
            "_key_transform": staticmethod(kt),
            "discriminator_field": discriminator_field,
            "_mro_scope": mro_scope,
            "_mro_depth": mro_depth,
        },
    )
    return registrar_cls
