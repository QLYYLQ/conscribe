"""Three-artifact parity matrix: in-process models, generated source, JSON schema.

conscribe derives three artifacts from one set of registered classes: the
in-process pydantic models, the generated Python source (``<layer>_config.py``
/ ``_composed.py``) and the published JSON schema.  This matrix builds small
registries for every

* **generation path** -- per-layer flat, per-layer nested, composed with
  inline wiring (in each ``top_level_extra`` mode) and composed with key
  selectors (``inline_wiring=False``) -- and every
* **owner path** -- the registrar metaclass (Path A), ``register()`` (Path C),
  ``bridge()``, the ``register(propagate=True)`` hook, BaseModel subclasses,
  stdlib dataclasses and pydantic dataclasses --

and asserts the three artifacts agree: equal JSON schemas (the composed
schema *is* the top-level model's own schema), and identical outcomes --
accepted with equal dumps, or rejected with the same errors (type, location,
message, context) -- for valid payloads and for payloads derived from them
(every field omitted, nulled, given values of other types; wrong
discriminator tags; constraint violations; inline-only refusal; default
instances; ``null`` on non-Optional receptors), in python and JSON mode, with
the generated source loaded as a module *and* exec'd into a plain dict (every
schema there, and the top-level samples).

The fixtures exercise F3 (strict top level, inline-only layers), F4 (default
instances, the Optional receptor's tri-state), F5 (keyword defaults,
constraints, tuples, sets, pure and impure factories), D9 (descriptions on
inline receptors) and D6 (imported user types; the old-shape guard of a class
1.x would have flattened).  Everything is compared
artifact against artifact -- no expected shapes are hard-coded -- so the
matrix stays valid when extraction details change; a few spec'd outcomes
(F3/F4) are asserted on top.

The fixture classes are written once as data (:data:`OWNER_KNOBS`,
:data:`RECEPTORS`, ...) and rendered into each owner path's class syntax
(:func:`_class_source`).
"""
from __future__ import annotations

import dataclasses
import enum
import os
import shutil
import subprocess
import sys
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, Optional, Protocol, runtime_checkable

import pydantic  # noqa: F401 -- the pydantic-dataclass fixture sources use it
import pytest
from pydantic import BaseModel, ConfigDict, Field, model_validator

from conscribe import create_registrar
from conscribe.config.builder import build_layer_config
from conscribe.config.codegen import (
    _union_alias_name,
    generate_composed_config_source,
    generate_layer_config_source,
)
from conscribe.config.composed import INLINE_ONLY_ERROR_TYPE, build_composed_config
from conscribe.config.json_schema import (
    generate_composed_json_schema,
    generate_layer_json_schema,
)
from conscribe.testing import isolated_registries

from tests.config.artifact_parity import (
    assert_artifacts_agree,
    composed_expectations,
    layer_expectations,
    perturbations,
    validation_outcome,
)

# ---------------------------------------------------------------------------
# Importable fixture types (module level: generated sources import them)
# ---------------------------------------------------------------------------


@runtime_checkable
class TargetP(Protocol):
    def run(self) -> None: ...


@runtime_checkable
class LlmP(Protocol):
    def run(self) -> None: ...


@runtime_checkable
class SinkP(Protocol):
    def run(self) -> None: ...


class Shade(enum.Enum):
    LIGHT = "light"
    DARK = "dark"


class UserKnob(BaseModel):
    """A user BaseModel knob (frozen: dataclasses accept it as a default)."""

    model_config = ConfigDict(frozen=True)
    depth: Annotated[int, Field(ge=0)] = 1
    mode: Literal["fast", "slow"] = "fast"


@dataclasses.dataclass(frozen=True)
class UserPoint:
    x: int = 0
    y: int = 0


class Tier3Schema(BaseModel):
    """A Tier-3 ``__config_schema__`` with its own validator (inherited, K)."""

    label: Annotated[str, Field(min_length=1)] = "t"
    target: Optional[str] = None

    @model_validator(mode="after")
    def _no_bad_label(self) -> Tier3Schema:
        if self.label == "bad":
            raise ValueError("bad label")
        return self


_GLOBS = ("*.py", "*.md")


def fresh_globs() -> list[str]:
    """An impure default factory (reads a module global): rendered by reference."""
    return list(_GLOBS)


TARGET, LLM, SINK, HOOK, OWNER, NESTED = (
    "mx_target", "mx_llm", "mx_sink", "mx_hook", "mx_owner", "mx_nested",
)

# ---------------------------------------------------------------------------
# Fixture classes as data
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Knob:
    """One constructor parameter / field of a fixture class.

    Attributes:
        name: Parameter / field name.
        annotation: Its annotation (source).  Receptors use the protocol type.
        default: Default expression (source); ``None`` with no ``factory``
            means required.
        factory: Default factory expression (source).
        key_annotation: The annotation on owner paths whose receptors are
            key selectors (BaseModel and pydantic dataclasses validate their
            fields, so they cannot hold protocol types).
    """

    name: str
    annotation: str
    default: str | None = None
    factory: str | None = None
    key_annotation: str | None = None

    def ann(self, receptor_style: str) -> str:
        if receptor_style == "key" and self.key_annotation is not None:
            return self.key_annotation
        return self.annotation


#: Knobs of the owner class under test (F5 / D6 vocabulary).
OWNER_KNOBS: tuple[Knob, ...] = (
    Knob("label", "str", "'x'"),
    Knob("count", "Annotated[int, Field(ge=0, le=10, description='how many')]", "3"),
    Knob("ratio", "Annotated[float, Field(gt=0, lt=1)]", "0.5"),
    Knob("tags", "tuple[str, ...]", "('a', 'b')"),
    Knob("pair", "tuple[int, int]", "(1, 2)"),
    Knob("shade", "Shade", "Shade.LIGHT"),
    Knob("knob", "UserKnob", "UserKnob()"),
    Knob("point", "Optional[UserPoint]", "None"),
    Knob("code", "Annotated[str, Field(pattern='^[a-z]+$', max_length=8)]", "'abc'"),
    Knob("limits", "frozenset[str]", "frozenset({'b', 'a', 'c'})"),
    # Default factories: impure (rendered by reference) and pure (a builtin).
    Knob("globs", "list[str]", factory="fresh_globs"),
    Knob("extras", "list[int]", factory="list"),
)

#: Wiring receptors of the owner class under test (F4 / D9), typed with the
#: target protocols (key types where the owner validates its own fields).
RECEPTORS: tuple[Knob, ...] = (
    Knob(
        "primary",
        "Annotated[TargetP, Field(description='the primary target')]",
        key_annotation="Annotated[str, Field(description='the primary target')]",
    ),
    Knob("backup", "Optional[TargetP]", "None", key_annotation="Optional[str]"),
    Knob("pick", "Optional[TargetP]", "None", key_annotation="Optional[str]"),
    Knob(
        "llm",
        "Annotated[Optional[LlmP], Field(description='which model')]",
        "None",
        key_annotation="Annotated[Optional[str], Field(description='which model')]",
    ),
    Knob("sinks", "Optional[list[SinkP]]", "None", key_annotation="Optional[list[str]]"),
)

MAIN_WIRING: dict[str, Any] = {
    "primary": {"registry": TARGET, "default": "alpha"},  # default instance
    "backup": {"registry": TARGET, "default": "alpha"},  # Optional: tri-state
    "pick": (TARGET, ["beta", "gamma"]),  # subset of a flat layer
    "llm": (LLM, ["openai.azure", "anthropic.direct"]),  # subset of a nested layer
    "sinks": SINK,  # list receptor of the inline-only layer
}

#: A second owner class: one required knob.
PLAIN_KNOBS: tuple[Knob, ...] = (
    Knob("name", "str"),
    Knob("retries", "Annotated[int, Field(ge=0)]", "0"),
    Knob("note", "Optional[str]", "None"),
)

#: A third owner class: one BaseModel parameter, which 1.x flattened into
#: the config (D6) -- 2.0 nests it and refuses the old flat keys.
LEGACY_KNOBS: tuple[Knob, ...] = (Knob("settings", "UserKnob", "UserKnob()"),)

#: Nested-layer levels: level 0 on the family classes, level 1 on the leaves.
FAMILY_KNOBS: dict[str, tuple[Knob, ...]] = {
    "openai": (
        Knob(
            "temperature",
            "Annotated[float, Field(ge=0, le=2, description='sampling temperature')]",
            "0.7",
        ),
    ),
    "anthropic": (Knob("model", "str", "'claude'"),),
}
LEAF_KNOBS: dict[str, tuple[Knob, ...]] = {
    "openai.azure": (Knob("deployment", "str", "'dep'"), Knob("version", "tuple[int, int]", "(2024, 1)")),
    "openai.official": (
        Knob("endpoint", "str", "'https://api'"),
        Knob("retries", "Annotated[int, Field(ge=0)]", "2"),
        # A factory makes the whole provider block default via
        # Field(default_factory=<segment model>).
        Knob("headers", "dict[str, str]", factory="dict"),
    ),
    "anthropic.direct": (Knob("org", "str"), Knob("shade", "Shade", "Shade.DARK")),
}

# ---------------------------------------------------------------------------
# Owner paths
# ---------------------------------------------------------------------------

#: owner path -> (class syntax, receptor style)
OWNER_PATHS: dict[str, tuple[str, str]] = {
    "metaclass": ("init", "protocol"),
    "register": ("init", "protocol"),
    "bridge": ("init", "protocol"),
    "propagate": ("init", "protocol"),
    "basemodel": ("fields", "key"),
    "dataclass": ("fields", "protocol"),
    "pydantic_dataclass": ("fields", "key"),
}


class External:
    """A plain class bridged into a registry (``bridge()`` owner path)."""

    def __init__(self, ext_flag: bool = False):
        self.ext_flag = ext_flag

    def run(self) -> None: ...


def _ordered(knobs: Iterable[Knob]) -> list[Knob]:
    """Required knobs first, then defaults, then factories."""
    knobs = list(knobs)
    return (
        [k for k in knobs if k.default is None and k.factory is None]
        + [k for k in knobs if k.default is not None]
        + [k for k in knobs if k.factory is not None]
    )


def _init_source(knobs: Sequence[Knob], style: str, *, forward: bool) -> list[str]:
    """``def __init__`` for the init-syntax paths (factories keyword-only)."""
    params = ["self"]
    factories = []
    for knob in _ordered(knobs):
        ann = knob.ann(style)
        if knob.factory is not None:
            factories.append(f"{knob.name}: Annotated[{ann}, Field(default_factory={knob.factory})]")
        elif knob.default is None:
            params.append(f"{knob.name}: {ann}")
        else:
            params.append(f"{knob.name}: {ann} = {knob.default}")
    if factories:
        params.append("*")
        params.extend(factories)
    if forward:
        params.append("**kwargs")
    body = "super().__init__(**kwargs)" if forward else "..."
    return [f"def __init__({', '.join(params)}):", f"    {body}"]


def _field_source(knobs: Sequence[Knob], style: str, factory_call: str) -> list[str]:
    """Class-level field declarations for the field-syntax paths."""
    lines = []
    for knob in _ordered(knobs):
        ann = knob.ann(style)
        if knob.factory is not None:
            lines.append(f"{knob.name}: {ann} = {factory_call}(default_factory={knob.factory})")
        elif knob.default is None:
            lines.append(f"{knob.name}: {ann}")
        else:
            lines.append(f"{knob.name}: {ann} = {knob.default}")
    return lines


def _class_source(
    path: str,
    *,
    name: str,
    key: str,
    knobs: Sequence[Knob],
    base: str | None = None,
    wiring: Mapping[str, Any] | None = None,
    registered: bool = True,
) -> str:
    """Source of one fixture class, in the class syntax of ``path``.

    ``registered=False`` declares an unregistered keyed parent (nested
    layers' family level): ``__abstract__`` on the metaclass paths, a plain
    ``__registry_key__`` attribute elsewhere.
    """
    syntax, style = OWNER_PATHS[path]
    decorators: list[str] = []
    body: list[str] = []
    if path in ("metaclass", "bridge", "propagate"):
        bases = base or {"metaclass": "BASE", "bridge": "BRIDGE", "propagate": "ROOT"}[path]
        body.append(f"__registry_key__ = {key!r}")
        if not registered:
            body.append("__abstract__ = True")
    else:
        bases = base or ("BaseModel" if path == "basemodel" else "")
        if registered:
            decorators.append(f"@REG.register({key!r})")
        else:
            body.append(f"__registry_key__ = {key!r}")
        if path == "dataclass":
            decorators.append("@dataclasses.dataclass(kw_only=True)")
        elif path == "pydantic_dataclass":
            decorators.append("@pydantic.dataclasses.dataclass(kw_only=True)")
    if wiring:
        body.append(f"__wiring__ = {dict(wiring)!r}")
    if syntax == "init":
        # Nested leaves (given a ``base``) forward to their family: a nested
        # layer's config follows the leaf's real constructor.
        body.extend(_init_source(knobs, style, forward=path == "bridge" or base is not None))
    else:
        factory_call = "dataclasses.field" if path == "dataclass" else "Field"
        body.extend(_field_source(knobs, style, factory_call))
    body.append("def run(self) -> None: ...")
    header = f"class {name}({bases}):" if bases else f"class {name}:"
    return "\n".join([*decorators, header, *("    " + line for line in body)]) + "\n"


def _define(path: str, registrar: Any, sources: Iterable[str]) -> dict[str, Any]:
    """Execute class sources for ``path`` against ``registrar``; return the namespace."""
    namespace: dict[str, Any] = {**globals(), "REG": registrar}
    if path == "metaclass":
        namespace["BASE"] = _abstract_base(registrar)
    elif path == "bridge":
        namespace["BRIDGE"] = registrar.bridge(External)
    elif path == "propagate":
        root_key = f"root_{registrar._registry.name}"
        if registrar.discriminator_fields:  # nested-mode keys: one segment per level
            root_key = registrar._key_separator.join([root_key] * len(registrar.discriminator_fields))

        @registrar.register(root_key, propagate=True)
        class Root:
            def __init__(self, root_flag: bool = False): ...

            def run(self) -> None: ...

        namespace["ROOT"] = Root
    for source in sources:
        exec(compile(source, f"<{path} fixture>", "exec"), namespace)  # noqa: S102
    return namespace


def _abstract_base(registrar: Any) -> type:
    class _Base(metaclass=registrar.Meta):
        __abstract__ = True

        def run(self) -> None: ...

    return _Base


# ---------------------------------------------------------------------------
# Layers
# ---------------------------------------------------------------------------


def _shared_layers(*, sink_inline_only: bool) -> dict[str, Any]:
    """The wiring targets (Path A): a flat target, a nested LLM layer, the sink
    layer (inline-only in inline builds) and a hook that always gives it a home."""
    target = create_registrar(TARGET, TargetP, discriminator_field="kind")
    target_base = _abstract_base(target)

    class Alpha(target_base):
        __registry_key__ = "alpha"

        def __init__(
            self,
            level: Annotated[int, Field(ge=0, description="alpha level")] = 1,
            tags: tuple[str, ...] = (),
        ): ...

    class Beta(target_base):
        __registry_key__ = "beta"

        def __init__(self, name: str, weight: float = 1.0): ...

    class Gamma(target_base):
        __registry_key__ = "gamma"

        def __init__(self, mode: Literal["a", "b"] = "a", shade: Shade = Shade.DARK): ...

    llm = create_registrar(LLM, LlmP, discriminator_fields=["family", "provider"], key_separator=".")

    class _OpenAI(metaclass=llm.Meta):
        __registry_key__ = "openai"
        __abstract__ = True

        def run(self) -> None: ...

        def __init__(self, temperature: Annotated[float, Field(ge=0, le=2)] = 0.7): ...

    class Azure(_OpenAI):
        __registry_key__ = "openai.azure"

        def __init__(self, deployment: str = "dep", **kwargs: Any): ...

    class Official(_OpenAI):
        __registry_key__ = "openai.official"

        def __init__(self, endpoint: str = "https://api", retries: Annotated[int, Field(ge=0)] = 2, **kwargs: Any): ...

    class _Anthropic(metaclass=llm.Meta):
        __registry_key__ = "anthropic"
        __abstract__ = True

        def run(self) -> None: ...

    class Direct(_Anthropic):
        __registry_key__ = "anthropic.direct"

        def __init__(self, org: str = "acme"): ...

    sink = create_registrar(SINK, SinkP, discriminator_field="sink_id", inline_only=sink_inline_only)
    sink_base = _abstract_base(sink)

    class Disk(sink_base):
        __registry_key__ = "disk"

        def __init__(self, path: str = "/tmp", rotate: bool = False): ...

    class Net(sink_base):
        __registry_key__ = "net"

        def __init__(self, host: str = "localhost", port: Annotated[int, Field(gt=0, lt=65536)] = 80): ...

    hook = create_registrar(HOOK, SinkP, discriminator_field="hook_id")
    hook_base = _abstract_base(hook)

    class TraceWriter(hook_base):
        __registry_key__ = "trace_writer"
        __wiring__ = {"archive": SINK}

        def __init__(
            self,
            trace_dir: str = "traces",
            archive: Annotated[Optional[list[SinkP]], Field(description="where traces go")] = None,
        ): ...

    return {TARGET: target, LLM: llm, SINK: sink, HOOK: hook}


def _owner_layer(path: str) -> Any:
    """The owner layer, its classes registered through ``path``."""
    owner = create_registrar(OWNER, TargetP, discriminator_field="owner_id")
    _define(path, owner, [
        _class_source(path, name="Main", key="main", knobs=RECEPTORS + OWNER_KNOBS, wiring=MAIN_WIRING),
        _class_source(path, name="Plain", key="plain", knobs=PLAIN_KNOBS),
        _class_source(path, name="Legacy", key="legacy", knobs=LEGACY_KNOBS),
    ])
    return owner


def _nested_layer(path: str) -> Any:
    """A nested-mode layer whose classes are declared through ``path``."""
    nested = create_registrar(NESTED, LlmP, discriminator_fields=["family", "provider"], key_separator=".")
    families = {
        family: _class_source(path, name=f"_{family.title()}", key=family, knobs=knobs, registered=False)
        for family, knobs in FAMILY_KNOBS.items()
    }
    leaves = [
        _class_source(
            path,
            name=key.replace(".", "_").title().replace("_", ""),
            key=key,
            knobs=knobs,
            base=f"_{key.split('.')[0].title()}",
        )
        for key, knobs in LEAF_KNOBS.items()
    ]
    _define(path, nested, [*families.values(), *leaves])
    return nested


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------

#: Receptor values in both shapes (inline configs and key selectors), valid
#: and invalid; :func:`perturbations` puts each into every field.
RECEPTOR_VALUES: tuple[Any, ...] = (
    "alpha",
    "beta",
    "openai.azure",
    ["disk"],
    ["disk", "zz"],
    {"kind": "alpha"},
    {"kind": "beta"},  # Beta's ``name`` is required
    {"kind": "beta", "name": "b"},
    {"kind": "gamma", "mode": "z"},
    {"kind": "nope"},
    {"family": "openai", "provider": {"name": "azure"}},
    {"family": "openai", "provider": {"name": "official", "retries": -1}},
    {"family": "anthropic"},
    [{"sink_id": "disk"}, {"sink_id": "net", "port": 0}],
    [{"sink_id": "zz"}],
    [{"path": "/x"}],
)

MAIN_INLINE = {
    "owner_id": "main",
    "primary": {"kind": "beta", "name": "b"},
    "backup": {"kind": "gamma"},
    "pick": {"kind": "gamma", "mode": "b"},
    "llm": {"family": "anthropic", "provider": {"name": "direct", "org": "o"}},
    "sinks": [{"sink_id": "net", "port": 8080}, {"sink_id": "disk"}],
    "knob": {"depth": 2, "mode": "slow"},
    "point": {"x": 1, "y": 2},
    "tags": ["q"],
    "pair": [3, 4],
    "shade": "dark",
    "limits": ["z"],
    "globs": ["a"],
    "extras": [1],
}
MAIN_KEYS = {
    **{k: v for k, v in MAIN_INLINE.items() if k not in {"primary", "backup", "pick", "llm", "sinks"}},
    "primary": "beta",
    "backup": "gamma",
    "pick": "beta",
    "llm": "openai.azure",
    "sinks": ["disk", "net"],
}
PLAIN = {"owner_id": "plain", "name": "n"}
LEGACY = {"owner_id": "legacy", "settings": {"depth": 2, "mode": "slow"}}
#: The 1.x (flattened) spellings of ``LEGACY``.
LEGACY_FLAT: tuple[Any, ...] = (
    {"owner_id": "legacy", "depth": 2},
    {"owner_id": "legacy", "depth": 2, "mode": "slow"},
    {"owner_id": "legacy", "settings": {"depth": 2}, "mode": "slow"},
    {"owner_id": "legacy", "depth": -1, "zz_unknown": 1},
)


def _model_samples(
    model: type[BaseModel],
    bases: Sequence[Mapping[str, Any]],
    disc: str,
    *,
    receptors: Iterable[str] = (),
) -> list[Any]:
    """Discriminator / shape errors, then each base and its perturbations.

    The first base is perturbed in every field; receptor fields (of every
    base) also get :data:`RECEPTOR_VALUES`.
    """
    tag = model.model_fields[disc].default
    fields = [name for name in model.model_fields if name != disc]
    receptor_fields = [name for name in receptors if name in model.model_fields]
    out: list[Any] = [{disc: tag}, {}, {disc: "zz_nope"}, None, [], "text"]
    for i, base in enumerate(bases):
        out.append(dict(base))
        if i == 0:
            out.extend(perturbations(base, fields))
        out.extend(perturbations(base, receptor_fields, values=RECEPTOR_VALUES))
    return out


def _fit(payload: Mapping[str, Any], model: type[BaseModel]) -> dict[str, Any]:
    """``payload`` restricted to the fields ``model`` declares."""
    return {k: v for k, v in payload.items() if k in model.model_fields}


def _owner_model_samples(result: Any) -> dict[str, list[Any]]:
    """``{per-key model name: samples}`` for the owner layer's models."""
    main = result.per_key_models["main"]
    plain = result.per_key_models["plain"]
    legacy = result.per_key_models["legacy"]
    bases = [_fit(MAIN_INLINE, main), _fit(MAIN_KEYS, main), {"owner_id": "main"}]
    return {
        main.__name__: _model_samples(main, bases, "owner_id", receptors=MAIN_WIRING),
        plain.__name__: _model_samples(plain, [PLAIN], "owner_id"),
        legacy.__name__: [*_model_samples(legacy, [LEGACY], "owner_id"), *LEGACY_FLAT],
    }


NESTED_PAYLOADS: tuple[Any, ...] = (
    {"family": "openai"},
    {"family": "openai", "provider": {"name": "azure"}},
    {"family": "openai", "provider": {"name": "azure", "deployment": "d", "version": [2025, 2]}},
    {"family": "openai", "temperature": 1.5, "provider": {"name": "official"}},
    {"family": "openai", "temperature": 3, "provider": {"name": "official", "retries": -1}},
    {"family": "openai", "provider": {"name": "nope"}},
    {"family": "openai", "provider": "azure"},
    {"family": "anthropic", "provider": {"name": "direct", "org": "o", "shade": "light"}},
    {"family": "anthropic", "provider": {"name": "direct"}},
    {"family": "anthropic"},
    {"family": "nope"},
    {},
    {"provider": {"name": "azure"}},
    {"family": "openai", "provider": {"name": "azure", "zz": 1}},
    {"family": "openai", "zz": 1},
)


def _composed_samples(result: Any) -> list[Any]:
    """Top-level payloads: sections, typos, inline-only names, aggregated errors."""
    main = result.layer_results[OWNER].per_key_models["main"]
    return [
        {},
        {OWNER: [_fit(MAIN_INLINE, main), PLAIN]},
        {OWNER: [_fit(MAIN_KEYS, main)]},
        {OWNER: [{"owner_id": "main"}]},
        {OWNER: [{"owner_id": "main", "primary": None}]},
        {OWNER: [{"owner_id": "main", "backup": None}]},
        {OWNER: [{"owner_id": "nope"}]},
        {OWNER: [{}]},
        {OWNER: {"owner_id": "main"}},
        {"zz_unknown_section": []},
        {SINK: []},
        {SINK: None},
        {SINK: {}},
        {SINK: [{"sink_id": "disk"}]},
        {SINK: [], "zz_unknown_section": 1, OWNER: [{"owner_id": "main", "count": -1}]},
        {HOOK: [{"hook_id": "trace_writer", "archive": [{"sink_id": "net", "port": 0}]}]},
        {HOOK: [{"hook_id": "trace_writer", "archive": ["disk"]}]},
        {TARGET: [{"kind": "beta"}, {"kind": "alpha", "level": -1}], LLM: [{"family": "openai"}]},
        {LLM: [{"family": "openai", "provider": {"name": "official", "retries": -1}}]},
        {LLM: [{"family": "anthropic", "provider": {"name": "direct"}}], TARGET: None},
    ]


# ---------------------------------------------------------------------------
# Generation paths
# ---------------------------------------------------------------------------

GENERATION_PATHS = (
    "layer_flat",
    "layer_nested",
    "composed_inline_forbid",
    "composed_inline_ignore",
    "composed_inline_allow",
    "composed_keys",
)


def _check_layer_flat(path: str) -> None:
    _shared_layers(sink_inline_only=False)
    result = build_layer_config(_owner_layer(path))
    samples = _owner_model_samples(result)
    main = result.per_key_models["main"]
    alias = _union_alias_name(result.layer_name)
    samples[alias] = [
        _fit(MAIN_INLINE, main), _fit(MAIN_KEYS, main), PLAIN, {"owner_id": "main"}, {"owner_id": "zz"}, {},
    ]
    assert_artifacts_agree(
        generate_layer_config_source(result),
        layer_expectations(result),
        samples=samples,
        published=(generate_layer_json_schema(result), result.union_type),
        plain_namespace=[alias],
        min_accepted=1,
    )


def _check_layer_nested(path: str) -> None:
    result = build_layer_config(_nested_layer(path))
    assert result.discriminator_fields == ["family", "provider"]
    alias = _union_alias_name(result.layer_name)
    samples: dict[str, list[Any]] = {alias: list(NESTED_PAYLOADS)}
    for model in result.per_key_models.values():
        samples[model.__name__] = list(NESTED_PAYLOADS)
    assert_artifacts_agree(
        generate_layer_config_source(result),
        layer_expectations(result),
        samples=samples,
        published=(generate_layer_json_schema(result), result.union_type),
        plain_namespace=[alias],
        min_accepted=1,
    )


def _check_composed(path: str, *, inline: bool, mode: str) -> None:
    layers = _shared_layers(sink_inline_only=inline)
    layers[OWNER] = _owner_layer(path)
    result = build_composed_config(layers, inline_wiring=inline, top_level_extra=mode)
    top = result.top_level_type
    samples = _owner_model_samples(result.layer_results[OWNER])
    samples[top.__name__] = _composed_samples(result)
    module = assert_artifacts_agree(
        generate_composed_config_source(result),
        composed_expectations(result),
        samples=samples,
        exact_published=(generate_composed_json_schema(result), top),
        plain_namespace=[top.__name__],
        min_accepted=1,
    )

    # Spec'd outcomes on top of parity (both sides agree, so checking the
    # in-process side checks all three).
    assert top.model_config["extra"] == mode
    typo = validation_outcome(top, {"zz_unknown_section": []})
    if mode == "forbid":
        assert typo[0] == "error" and [e[0] for e in typo[1]] == ["extra_forbidden"]
    else:
        assert typo[0] == "ok"
    for value in ([], None, {}, [{"sink_id": "disk"}]):
        outcome = validation_outcome(top, {SINK: value})
        if inline:  # F3: an inline-only layer's name is refused in every mode
            assert outcome[0] == "error"
            assert [(e[0], e[1]) for e in outcome[1]] == [(INLINE_ONLY_ERROR_TYPE, (SINK,))]
    if inline:
        assert result.inline_only_layers and SINK in result.inline_only_layers
        assert set(generate_composed_json_schema(result)["properties"]) == set(result.top_level_layers)
        assert getattr(module, top.__name__).model_fields[SINK].exclude is True


@pytest.fixture(autouse=True)
def _isolated(conscribe_isolated):
    yield conscribe_isolated


#: Cells that cannot pass yet because of builder gaps outside the renderer:
#: ``(owner path, generation) -> (exception, reason)``.  Strict: when the gap
#: is closed the cell passes and the marker must go.
KNOWN_GAPS: dict[tuple[str, str], tuple[type[BaseException], str]] = {}


def _cells() -> list[Any]:
    cells = []
    for owner_path in OWNER_PATHS:
        for generation in GENERATION_PATHS:
            marks = []
            gap = KNOWN_GAPS.get((owner_path, generation))
            if gap is not None:
                marks.append(pytest.mark.xfail(strict=True, raises=gap[0], reason=gap[1]))
            cells.append(pytest.param(owner_path, generation, id=f"{owner_path}-{generation}", marks=marks))
    return cells


@pytest.mark.parametrize(("owner_path", "generation"), _cells())
def test_artifacts_agree(owner_path: str, generation: str) -> None:
    if generation == "layer_flat":
        _check_layer_flat(owner_path)
    elif generation == "layer_nested":
        _check_layer_nested(owner_path)
    elif generation == "composed_keys":
        _check_composed(owner_path, inline=False, mode="forbid")
    else:
        _check_composed(owner_path, inline=True, mode=generation.rsplit("_", 1)[1])


# ---------------------------------------------------------------------------
# Focused receptor shapes
# ---------------------------------------------------------------------------


def test_optional_protocol_receptor() -> None:
    """``backup: Optional[TargetP] = None`` -- af's receptor spelling -- inlines
    with a tri-state default instance, identically in all three artifacts."""
    layers = _shared_layers(sink_inline_only=True)
    owner = create_registrar(OWNER, TargetP, discriminator_field="owner_id")
    _define("metaclass", owner, [
        _class_source(
            "metaclass", name="Keeper", key="keeper",
            knobs=(Knob("backup", "Optional[TargetP]", "None"), Knob("sinks", "Optional[list[SinkP]]", "None")),
            wiring={"backup": {"registry": TARGET, "default": "alpha"}, "sinks": SINK},
        ),
    ])
    result = build_composed_config({**layers, OWNER: owner})
    keeper = result.layer_results[OWNER].per_key_models["keeper"]
    assert keeper.__inline_wired__["backup"]["optional"] is True
    assert_artifacts_agree(
        generate_composed_config_source(result),
        composed_expectations(result),
        samples={keeper.__name__: [{}, {"backup": None}, {"backup": {"kind": "gamma"}}, {"backup": "alpha"}]},
        exact_published=(generate_composed_json_schema(result), result.top_level_type),
        plain_namespace=True,
        min_accepted=3,
    )


@pytest.mark.parametrize("inline", [True, False])
def test_tier3_schema_owner(inline: bool) -> None:
    """A ``__config_schema__`` owner: the emitted per-key class subclasses the
    imported schema, so its validator and constraints apply in the generated
    file exactly as in-process (K)."""
    layers = _shared_layers(sink_inline_only=inline)
    owner = create_registrar(OWNER, TargetP, discriminator_field="owner_id")

    @owner.register("tier3")
    class Tier3:
        __config_schema__ = Tier3Schema
        __wiring__ = {"target": {"registry": TARGET, "keys": ["alpha", "gamma"]}}

        def run(self) -> None: ...

    result = build_composed_config({**layers, OWNER: owner}, inline_wiring=inline)
    model = result.layer_results[OWNER].per_key_models["tier3"]
    source = generate_composed_config_source(result)
    assert f"class {model.__name__}(Tier3Schema):" in source
    assert_artifacts_agree(
        source,
        composed_expectations(result),
        samples={model.__name__: [
            {}, {"label": "bad"}, {"label": ""}, {"target": "alpha"}, {"target": {"kind": "gamma"}},
            {"owner_id": "tier3", "zz": 1},
        ]},
        exact_published=(generate_composed_json_schema(result), result.top_level_type),
        plain_namespace=True,
        min_accepted=2,
    )


def test_described_optional_receptors() -> None:
    """D9: described ``Annotated[Optional[...], Field(description=...)] = None``
    receptors keep their descriptions inline, identically in all three
    artifacts (Python 3.10's ``get_type_hints`` wraps them in a second
    ``Optional``, which extraction normalizes)."""
    layers = _shared_layers(sink_inline_only=True)
    owner = create_registrar(OWNER, TargetP, discriminator_field="owner_id")
    _define("metaclass", owner, [
        _class_source(
            "metaclass", name="Described", key="described",
            knobs=(
                Knob("backup", "Annotated[Optional[str], Field(description='fallback target')]", "None"),
                Knob("archive", "Annotated[Optional[list[SinkP]], Field(description='where it goes')]", "None"),
            ),
            wiring={"backup": {"registry": TARGET, "default": "alpha"}, "archive": SINK},
        ),
    ])
    result = build_composed_config({**layers, OWNER: owner})
    described = result.layer_results[OWNER].per_key_models["described"]
    assert described.model_fields["backup"].description == "fallback target"
    assert described.model_fields["archive"].description == "where it goes"
    properties = generate_composed_json_schema(result)["$defs"][described.__name__]["properties"]
    assert properties["backup"]["description"] == "fallback target"
    assert properties["archive"]["description"] == "where it goes"
    assert_artifacts_agree(
        generate_composed_config_source(result),
        composed_expectations(result),
        samples={described.__name__: [
            {}, {"backup": None}, {"backup": {"kind": "beta", "name": "b"}}, {"archive": [{"sink_id": "net"}]},
            {"archive": ["net"]}, {"backup": "alpha"},
        ]},
        exact_published=(generate_composed_json_schema(result), result.top_level_type),
        plain_namespace=True,
        min_accepted=3,
    )


def test_nested_basemodel_leaves_keep_their_fields() -> None:
    result = build_layer_config(_nested_layer("basemodel"))
    azure = result.per_segment_models["provider"]["azure"]
    assert "deployment" in azure.model_fields
    assert "temperature" in result.per_key_models["openai.azure"].model_fields


# ---------------------------------------------------------------------------
# Determinism and type checking of the generated files
# ---------------------------------------------------------------------------

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_DETERMINISM_SCRIPT = textwrap.dedent(
    """
    import json, sys
    from conscribe.config.builder import build_layer_config
    from conscribe.config.codegen import generate_composed_config_source, generate_layer_config_source
    from conscribe.config.composed import build_composed_config
    from conscribe.config.json_schema import generate_composed_json_schema, generate_layer_json_schema
    from conscribe.testing import isolated_registries
    import tests.config.test_parity_matrix as matrix

    for inline in (True, False):
        with isolated_registries():
            layers = matrix._shared_layers(sink_inline_only=inline)
            layers[matrix.OWNER] = matrix._owner_layer("metaclass")
            result = build_composed_config(layers, inline_wiring=inline)
            sys.stdout.write(generate_composed_config_source(result))
            sys.stdout.write(json.dumps(generate_composed_json_schema(result), indent=1))
    with isolated_registries():
        result = build_layer_config(matrix._nested_layer("metaclass"))
        sys.stdout.write(generate_layer_config_source(result))
        sys.stdout.write(json.dumps(generate_layer_json_schema(result), indent=1))
    """
)


def _generate_with_hash_seed(seed: str) -> str:
    proc = subprocess.run(
        [sys.executable, "-c", _DETERMINISM_SCRIPT],
        cwd=_ROOT,
        env={**os.environ, "PYTHONHASHSEED": seed},
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_generated_artifacts_do_not_depend_on_the_hash_seed() -> None:
    """Composed source + schema (inline and key-selector builds) and a nested
    layer's source + schema are byte-identical across ``PYTHONHASHSEED``
    values (the fixtures hold set-valued defaults)."""
    first, second = (_generate_with_hash_seed(seed) for seed in ("1", "2"))
    assert "frozenset({'a', 'b', 'c'})" in first
    assert first == second


def _generated_files() -> dict[str, str]:
    """A representative set of generated files (all constructs of the matrix)."""
    files: dict[str, str] = {}
    with isolated_registries():
        layers = _shared_layers(sink_inline_only=True)
        layers[OWNER] = _owner_layer("metaclass")
        for mode in ("forbid", "allow"):
            result = build_composed_config(layers, top_level_extra=mode)
            files[f"composed_{mode}.py"] = generate_composed_config_source(result)
        files["owner_layer.py"] = generate_layer_config_source(build_layer_config(layers[OWNER]))
    with isolated_registries():
        files["nested_layer.py"] = generate_layer_config_source(build_layer_config(_nested_layer("metaclass")))
    with isolated_registries():
        layers = _shared_layers(sink_inline_only=False)
        layers[OWNER] = _owner_layer("basemodel")
        files["composed_keys.py"] = generate_composed_config_source(
            build_composed_config(layers, inline_wiring=False),
        )
    return files


@pytest.mark.skipif(shutil.which("pyright") is None, reason="pyright is not on PATH")
def test_generated_files_type_check_with_pyright(tmp_path: Any) -> None:
    """Optional: the matrix's generated files are pyright-clean."""
    paths = []
    for name, text in _generated_files().items():
        path = tmp_path / name
        path.write_text(text)
        paths.append(str(path))
    proc = subprocess.run(
        ["pyright", "--pythonpath", sys.executable, *paths],
        cwd=_ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
