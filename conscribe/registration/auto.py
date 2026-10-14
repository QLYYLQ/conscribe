"""AutoRegistrar metaclass factory.

Creates metaclasses that automatically register concrete subclasses
into a LayerRegistry at class definition time.
This is the shared mechanism for Path A (inheritance) and Path B (bridge).

Every registration path ends in the same tail, :func:`_register_class`:
grammar-check the merged ``__wiring__`` → add every key → stamp the class →
install its ``WiredField`` placeholders.  Undoing a failure is the caller's
job, because the paths differ in scope (the metaclass and the propagation
hook purge the class from every registry; ``register()`` undoes only its
own binding).

Four rules hold for every class the metaclass creates:

- **Keys.** A registry key is a non-empty ``str``; ``__registry_key__`` may
  also be a non-empty list of them.  Anything else (an Enum member, a tuple,
  an int, an empty value) raises :class:`~conscribe.exceptions.InvalidRegistryKeyError`
  before the class touches any registry.
- **Wiring.** Every class gets its ``WiredField`` injection placeholders
  (``conscribe.wiring.inject_wired_descriptors``), registered or not, since
  subclasses inherit them.  A class that registers must also have a
  well-formed merged ``__wiring__``: ``check_wiring_declaration`` runs after
  the skip filters and before any ``registry.add``, so a malformed
  declaration fails the class statement and registers nothing.  (Classes the
  filters skip are not checked; their entries are checked on every
  registering subclass, which inherits them.)
- **Atomic registration.** If anything fails after the class object exists
  (a multi-key collision, the second registry of a cross-registry diamond
  ``A.Meta | B.Meta``, an invalid ``__config_schema__``), the class is purged
  from every live registry before the error propagates, so no registry keeps
  a class whose definition failed.
- **Re-creation.** Every registered class is stamped with
  ``__registry_origin__ = id(cls)``.  ``@dataclass(slots=True)``,
  ``attrs.define`` and pydantic's ``dataclass(slots=True)`` rebuild a class
  from a copy of its ``__dict__``; the copy carries the stamp, so the
  metaclass recognises the rebuilt class, rebinds every key of the original
  in every registry to it and skips normal registration (including the
  wiring check, which the original already passed).  A class body that
  simply runs twice carries no stamp and still raises ``DuplicateKeyError``.
  Unsupported: a Path C ``@register`` applied *below* ``@dataclass(slots=True)``
  (the registration stays on the replaced class; put ``@register`` on top),
  and ``@pydantic.dataclasses.dataclass`` stacked on a stdlib ``@dataclass``
  (pydantic wraps the class in a same-named subclass, which collides with the
  original's key).
"""
from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Any, Callable

from conscribe.exceptions import InvalidConfigSchemaError, InvalidRegistryKeyError
from conscribe.registration.filters import (
    RegistrationContext,
    build_filter_chain,
    should_skip_registration,
)
from conscribe.registration.key_transform import KeyTransform
from conscribe.registration.meta_base import AutoRegistrarBase, MetaRegistrarType
from conscribe.registration.registry import _purge_class, _rebind_recreated

if TYPE_CHECKING:
    from collections.abc import Mapping

    from conscribe.registration.registry import LayerRegistry

#: Class attribute stamped on every registered class: ``id()`` of the class
#: object that was registered.  A class re-created from a copy of a registered
#: class's namespace carries the copy source's id in its own namespace.
REGISTRY_ORIGIN_ATTR = "__registry_origin__"


def _register_class(
    registry: LayerRegistry,
    cls: type,
    keys: list[str],
    *,
    protocol_check: bool,
) -> None:
    """The registration tail every path shares: check, add, stamp, inject.

    1. Grammar-check the merged ``__wiring__``
       (``conscribe.wiring.check_wiring_declaration``) -- a malformed
       declaration raises before the class touches the registry.
    2. Add *cls* under every key (the Protocol check, when asked for, runs
       on the first add, before anything is stored).
    3. Stamp ``__registry_key__`` (the first key), ``__registry_keys__``
       (all keys, multi-key registration only) and ``__registry_origin__``.
    4. Install the ``WiredField`` placeholders
       (``conscribe.wiring.inject_wired_descriptors``).

    Undoing a failure after the first add is the caller's job: the
    registration metaclass and the propagation hook purge the class from
    every live registry, ``register()`` removes only its own binding.

    Args:
        registry: The target registry.
        cls: The class to register.
        keys: Validated keys (see :func:`_registry_keys`).
        protocol_check: Run the registry's Protocol check (Path C).
    """
    from conscribe.wiring import check_wiring_declaration, inject_wired_descriptors

    check_wiring_declaration(cls)
    for index, key in enumerate(keys):
        registry.add(key, cls, protocol_check=protocol_check and index == 0)
    cls.__registry_key__ = keys[0]  # type: ignore[attr-defined]
    if len(keys) > 1:
        cls.__registry_keys__ = list(keys)  # type: ignore[attr-defined]
    setattr(cls, REGISTRY_ORIGIN_ATTR, id(cls))
    inject_wired_descriptors(cls)


def _registry_keys(value: object, cls: type, *, source: str) -> list[str]:
    """Validate a registration key (or key list) and return the key list.

    A key is a non-empty ``str`` that is not an Enum member; ``value`` may
    also be a non-empty ``list`` of such keys (multi-key registration).

    Args:
        value: The key(s), as written.
        cls: The class being registered (named in the error).
        source: Where *value* came from, for the error message
            (``"__registry_key__"``, ``"register(key=...)"``,
            ``"key_transform"``).

    Returns:
        The keys as a new list.

    Raises:
        InvalidRegistryKeyError: Naming the class, the value and the fix.
    """
    if isinstance(value, list):
        if not value:
            raise InvalidRegistryKeyError(
                cls.__qualname__, value, source,
                "an empty list registers the class under no key; list at least one "
                "key, or set __abstract__ = True to keep the class out of the registry",
            )
        for item in value:
            detail = _key_problem(item, in_list=True)
            if detail is not None:
                raise InvalidRegistryKeyError(cls.__qualname__, value, source, detail)
        return list(value)
    detail = _key_problem(value, in_list=False)
    if detail is not None:
        raise InvalidRegistryKeyError(cls.__qualname__, value, source, detail)
    return [value]  # type: ignore[list-item]


def _key_problem(key: object, *, in_list: bool) -> str | None:
    """Why *key* is not a registry key, or ``None`` when it is one."""
    where = "every key in the list" if in_list else "a registry key"
    if isinstance(key, enum.Enum):
        plain = getattr(key, "value", None)
        fix = f"write the plain string {plain!r}" if isinstance(plain, str) else (
            "write a plain string"
        )
        return (
            f"{where} must be a plain str, not the Enum member {key!r} "
            f"(str() of it is {str(key)!r}, so the class would be registered under "
            f"a key nobody looks up); {fix}"
        )
    if isinstance(key, str):
        if not key:
            return f"{where} must be a non-empty str"
        return None
    if isinstance(key, tuple) and not in_list:
        return (
            "a tuple is not a key; list several keys in a list "
            "(__registry_key__ = ['a', 'b']) to register the class under each"
        )
    return f"{where} must be a non-empty str, got {type(key).__name__} {key!r}"


def create_auto_registrar(
    registry: LayerRegistry,
    key_transform: KeyTransform,
    *,
    base_metaclass: type = type,
    skip_pydantic_generic: bool = True,
    skip_filter: Callable[[type], bool] | None = None,
    key_separator: str = "",
    inline_only: bool = False,
) -> type[AutoRegistrarBase]:
    """Create an AutoRegistrar metaclass via closure.

    Args:
        registry: Target registry instance. All subclasses register here.
        key_transform: Key inference function (class name -> registry key).
        base_metaclass: Parent metaclass for AutoRegistrar.
            Default is ``type``. Pass ``ABCMeta`` etc. to resolve conflicts.
        skip_pydantic_generic: If True (default), skip classes whose name
            contains ``[`` — these are Pydantic Generic specialization
            intermediates (e.g. ``BaseEvent[str]``) and should not be
            registered.
        skip_filter: Optional callable ``(cls) -> bool``. If it returns
            True for a class, that class is skipped (not registered).
        key_separator: Separator for hierarchical keys (e.g. ``"."``).
            Empty string means flat keys (backward compatible).
        inline_only: Declares the layer inline-only: its instances may only
            be written inside the wiring receptors of other layers, never as
            a top-level section of a composed config.  Recorded on the
            metaclass and readable as ``Meta.inline_only``; registration
            itself does not change.  ``create_registrar(inline_only=...)``
            passes its flag through here and exposes it as
            ``Registrar.inline_only``.

    Returns:
        An AutoRegistrar metaclass (subclass of base_metaclass).

    Raises:
        TypeError: If ``inline_only`` is not a bool.
    """
    if not isinstance(inline_only, bool):
        raise TypeError(
            f"inline_only must be a bool, got {type(inline_only).__name__} "
            f"({inline_only!r})."
        )

    # Ensure AutoRegistrarBase is in the metaclass chain
    if not issubclass(base_metaclass, AutoRegistrarBase):
        if base_metaclass is type:
            effective_base = AutoRegistrarBase
        else:
            # Combined metaclass: base_metaclass + AutoRegistrarBase
            # Use MetaRegistrarType so the | operator works
            effective_base = MetaRegistrarType(
                f"_{base_metaclass.__name__}AutoBase",
                (AutoRegistrarBase, base_metaclass),
                {},
            )
    else:
        effective_base = base_metaclass

    filters = build_filter_chain(
        skip_pydantic_generic=skip_pydantic_generic,
        skip_filter=skip_filter,
        registry_name=registry.name,
    )

    def _register(cls: type, name: str, bases: tuple, namespace: Mapping[str, Any]) -> None:
        """Steps 2-6 of class creation (the class object already exists)."""
        # Step 2: Tag with layer name
        cls.__registry_name__ = registry.name  # type: ignore[attr-defined]

        # Step 3: Re-creation from a registered class's namespace
        # (dataclass/attrs slots).  Read the namespace, not cls.__dict__,
        # so every metaclass of a cross-registry diamond sees the copy
        # source's id.  The copied namespace carries the placeholders too.
        origin = namespace.get(REGISTRY_ORIGIN_ATTR)
        if (
            type(origin) is int
            and origin != id(cls)
            and _rebind_recreated(origin, cls, namespace)
        ):
            setattr(cls, REGISTRY_ORIGIN_ATTR, id(cls))
            return

        # Step 4: Skip conditions via filter chain.  A skipped class still
        # gets its WiredField placeholders: subclasses inherit them.
        ctx = RegistrationContext(
            cls=cls,
            name=name,
            bases=bases,
            namespace=namespace,  # type: ignore[arg-type]
            registry_name=registry.name,
        )
        if should_skip_registration(filters, ctx):
            from conscribe.wiring import inject_wired_descriptors

            inject_wired_descriptors(cls)
            return

        # Step 5: Keys (explicit, hierarchical, multi-key; validated) and
        # __config_schema__, before touching the registry
        keys = _resolve_keys(
            namespace.get("__registry_key__"), name, cls, registry, key_transform,
            key_separator,
        )
        config_schema = namespace.get("__config_schema__")
        if config_schema is not None:
            _validate_config_schema(name, config_schema)

        # Step 6: The shared tail: wiring grammar (a malformed declaration
        # fails the class statement before any registry.add), add, stamp,
        # placeholders.
        _register_class(registry, cls, keys, protocol_check=False)

    class AutoRegistrar(effective_base):  # type: ignore[misc]
        """Metaclass that auto-registers concrete subclasses."""

        def __new__(
            mcs,
            name: str,
            bases: tuple,
            namespace: dict,
            **kwargs: object,
        ) -> type:
            # Step 1: Create the class — pass **kwargs through!
            cls = super().__new__(mcs, name, bases, namespace, **kwargs)
            try:
                _register(cls, name, bases, namespace)
            except BaseException:
                # Atomic registration: no registry keeps a class whose
                # definition failed (covers keys this metaclass added and
                # keys added by inner metaclasses or __init_subclass__ hooks).
                _purge_class(cls)
                raise
            return cls

    # Debug-friendly qualname
    AutoRegistrar.__qualname__ = f"AutoRegistrar[{registry.name}]"
    AutoRegistrar.__name__ = f"AutoRegistrar[{registry.name}]"
    # Read through MetaRegistrarType.inline_only (never via the classes the
    # metaclass creates).
    AutoRegistrar._conscribe_inline_only = inline_only

    return AutoRegistrar


def _resolve_keys(
    explicit_key: object,
    name: str,
    cls: type,
    registry: LayerRegistry,
    key_transform: KeyTransform,
    key_separator: str,
) -> list[str]:
    """Resolve registration keys from explicit key, name, or hierarchy.

    Supports:
    - ``__registry_key__ = "foo"`` — single explicit key
    - ``__registry_key__ = ["foo", "bar"]`` — multi-key registration
    - Hierarchical key derivation when ``key_separator`` is set
    - Default: ``key_transform(name)``

    Raises:
        InvalidRegistryKeyError: If the explicit key, or what the key
            transform returned, is not a valid key (see :func:`_registry_keys`).
    """
    if explicit_key is not None:
        return _registry_keys(explicit_key, cls, source="__registry_key__")

    if key_separator:
        derived = _derive_hierarchical_key(
            cls, name, registry, key_transform, key_separator,
        )
        if derived is not None:
            return [derived]

    return [_transformed_key(key_transform, name, cls)]


def _transformed_key(key_transform: KeyTransform, name: str, cls: type) -> str:
    """``key_transform(name)``, validated as a registry key."""
    return _registry_keys(key_transform(name), cls, source="key_transform")[0]


def _derive_hierarchical_key(
    cls: type,
    name: str,
    registry: LayerRegistry,
    key_transform: KeyTransform,
    separator: str,
) -> str | None:
    """Derive a hierarchical key from the first parent with a key in MRO.

    Finds the nearest parent that has ``__registry_key__`` set (whether
    the parent is registered or abstract). This allows abstract parents
    with explicit keys to serve as hierarchy roots.
    """
    for base in cls.__mro__[1:]:
        if base is object:
            continue
        parent_key = base.__dict__.get("__registry_key__", None)
        if parent_key and isinstance(parent_key, str):
            return f"{parent_key}{separator}{_transformed_key(key_transform, name, cls)}"
    return None


def _validate_config_schema(name: str, config_schema: object) -> None:
    """Validate that __config_schema__ is a BaseModel subclass."""
    from pydantic import BaseModel

    if not (isinstance(config_schema, type) and issubclass(config_schema, BaseModel)):
        raise InvalidConfigSchemaError(name, config_schema)
