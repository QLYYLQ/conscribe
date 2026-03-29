"""Custom exception classes for the conscribe package.

All registration-related exceptions inherit from RegistryError,
allowing consumers to catch all registry exceptions with a single handler.
Each exception carries structured context (layer name, key, class qualname, etc.)
to provide actionable error messages.
"""
from __future__ import annotations


class RegistryError(Exception):
    """Base class for all registry exceptions. Consumers can catch this uniformly."""


class DuplicateKeyError(RegistryError):
    """Raised when a duplicate key is registered in the same layer.

    Attributes:
        layer_name: Which layer (e.g. "agent").
        key: The conflicting key.
        existing_cls: The already-registered class.
        new_cls: The class that attempted to register.
    """

    def __init__(
        self,
        layer_name: str,
        key: str,
        existing_cls: type,
        new_cls: type,
    ) -> None:
        self.layer_name = layer_name
        self.key = key
        self.existing_cls = existing_cls
        self.new_cls = new_cls
        message = (
            f"Duplicate key '{key}' in '{layer_name}' registry: "
            f"'{new_cls.__qualname__}' conflicts with existing "
            f"'{existing_cls.__qualname__}'. "
            f"To fix: add __registry_key__ = '<unique_key>' to {new_cls.__name__}."
        )
        super().__init__(message)


class KeyNotFoundError(RegistryError, KeyError):
    """Raised when querying a key that does not exist in the registry.

    Inherits from KeyError for dict-like semantics.

    Attributes:
        layer_name: Which layer.
        key: The missing key.
        available_keys: All currently registered keys.
    """

    def __init__(
        self,
        layer_name: str,
        key: str,
        available_keys: list[str],
    ) -> None:
        self.layer_name = layer_name
        self.key = key
        self.available_keys = available_keys
        message = (
            f"Key '{key}' not found in '{layer_name}' registry. "
            f"Available keys: {', '.join(sorted(available_keys))}. "
            f"Did you forget to call discover() or import the module "
            f"containing this implementation?"
        )
        # KeyError uses args[0] for repr, so pass the full message there
        super().__init__(message)


class ProtocolViolationError(RegistryError, TypeError):
    """Raised when a class does not satisfy the layer's Protocol interface.

    Inherits from TypeError for semantic consistency.

    Attributes:
        layer_name: Which layer.
        cls: The offending class.
        missing_methods: Methods required by the protocol but not found on cls.
        protocol: The Protocol type that was violated.
    """

    def __init__(
        self,
        layer_name: str,
        cls: type,
        missing_methods: list[str],
        protocol: type,
    ) -> None:
        self.layer_name = layer_name
        self.cls = cls
        self.missing_methods = missing_methods
        self.protocol = protocol
        message = (
            f"'{cls.__qualname__}' cannot be registered in '{layer_name}' layer: "
            f"missing methods required by {protocol.__name__}: "
            f"{', '.join(sorted(missing_methods))}. "
            f"Implement these methods or inherit from the layer's base class."
        )
        super().__init__(message)


class InvalidConfigSchemaError(RegistryError, TypeError):
    """Raised when __config_schema__ is not a BaseModel subclass.

    Attributes:
        cls_name: Name of the class with the bad schema.
        actual: The actual value of __config_schema__.
    """

    def __init__(self, cls_name: str, actual: object) -> None:
        self.cls_name = cls_name
        self.actual = actual
        message = (
            f"{cls_name}.__config_schema__ must be a pydantic.BaseModel subclass, "
            f"got {type(actual).__name__}."
        )
        super().__init__(message)


class InvalidProtocolError(RegistryError, TypeError):
    """Raised when the provided protocol is not a @runtime_checkable Protocol.

    Attributes:
        protocol: The invalid protocol type.
    """

    def __init__(self, protocol: type) -> None:
        self.protocol = protocol
        message = (
            f"Protocol '{protocol.__name__}' must be decorated with "
            f"@runtime_checkable. "
            f"Add @runtime_checkable above your Protocol class definition."
        )
        super().__init__(message)


class WiringResolutionError(RegistryError):
    """Raised when ``__wiring__`` cannot be resolved at config build time.

    Possible causes:
    - Referenced registry not found (not yet created or typo in name).
    - Explicit key subset contains keys not present in the referenced registry.
    - Referenced registry is empty (forgot to call ``discover()``).

    Attributes:
        cls_name: The class that declared the wiring.
        param_name: The ``__init__`` parameter or injected field name.
        registry_name: The referenced registry name (if any).
        detail: Human-readable description of the problem.
    """

    def __init__(
        self,
        cls_name: str,
        param_name: str,
        registry_name: str,
        detail: str,
    ) -> None:
        self.cls_name = cls_name
        self.param_name = param_name
        self.registry_name = registry_name
        self.detail = detail
        message = (
            f"Wiring resolution failed for '{cls_name}.{param_name}' "
            f"(registry: '{registry_name}'): {detail}"
        )
        super().__init__(message)
