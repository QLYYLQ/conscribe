"""Artifact-parity checks for generated config sources (test helper).

conscribe derives three artifacts from one declaration: the in-process
pydantic models, the generated Python source (``<layer>_config.py`` /
``_composed.py``) and the published JSON schema.  This helper checks that
they agree -- by comparing the artifacts with each other, never with
hard-coded shapes, so the checks stay valid when extraction details change:

* :func:`load_generated_module` executes generated source as a real module
  (registered in ``sys.modules`` under a unique name, so pydantic can
  resolve deferred annotations and ``model_rebuild()`` works);
  :func:`exec_generated_namespace` executes it into a plain ``dict`` (no
  module at all), which generated files must support too.
* :func:`assert_artifacts_agree` compares, for every ``name -> in-process
  object`` pair, the generated object with the in-process one: their JSON
  schemas (``$ref``s inlined so codegen's class renames do not matter,
  single-item ``allOf`` wrappers collapsed, set defaults sorted) and the
  outcome of validating each sample payload, in python and JSON mode
  (accepted -> equal dumps; rejected -> the same errors: type, location,
  message and context, with class names read as schema titles so a renamed
  class compares equal).  Optionally it also compares a published schema
  (``generate_*_json_schema``) with the in-process schema -- ``x-``
  extensions ignored, or exactly (the composed top level publishes its own
  schema).
* :func:`perturbations` derives invalid / edge-case payloads from a valid
  one (every field omitted, nulled, given values of other types), so
  samples need not be hand-written per field.
* :func:`layer_expectations` / :func:`composed_expectations` build the
  ``name -> in-process object`` mapping for the standard outputs.
* :func:`type_to_source` / :func:`value_to_source` render one type
  expression / value on its own (the renderer's output, names materialized).

Used by ``tests/config/test_codegen.py``, ``tests/config/test_composed_v2.py``
and the parity matrix (``tests/config/test_parity_matrix.py``).
"""
from __future__ import annotations

import decimal
import difflib
import enum
import itertools
import json
import math
import re
import sys
import types
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, get_args, get_origin

from pydantic import BaseModel, TypeAdapter, ValidationError
from pydantic.fields import FieldInfo

from conscribe.config.json_schema import _DeterministicJsonSchema

__all__ = [
    "PERTURBATION_VALUES",
    "assert_artifacts_agree",
    "composed_expectations",
    "exec_generated_namespace",
    "layer_expectations",
    "load_generated_module",
    "normalize_schema",
    "perturbations",
    "same_value",
    "schema_of",
    "strip_degraded_hints",
    "type_to_source",
    "validation_outcome",
    "value_to_source",
]

_counter = itertools.count()


def load_generated_module(source: str, *, name: str | None = None) -> types.ModuleType:
    """Execute generated ``source`` as a fresh module and return it.

    The module stays registered in ``sys.modules`` (under a unique name) so
    its models keep resolving their annotations after this call.
    """
    module_name = name or f"_conscribe_generated_{next(_counter)}"
    module = types.ModuleType(module_name)
    module.__file__ = f"<{module_name}>"
    sys.modules[module_name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)  # noqa: S102
    return module


def exec_generated_namespace(source: str) -> dict[str, Any]:
    """Execute generated ``source`` into a plain ``dict`` (no module) and return it.

    Nothing is registered in ``sys.modules`` and the namespace has no
    ``__name__``: the file's own ``model_rebuild()`` calls must resolve every
    deferred annotation.
    """
    namespace: dict[str, Any] = {}
    exec(compile(source, "<generated, plain namespace>", "exec"), namespace)  # noqa: S102
    return namespace


def schema_of(obj: Any) -> dict[str, Any]:
    """JSON schema of a model class or a type (union alias), deterministic."""
    return TypeAdapter(obj).json_schema(schema_generator=_DeterministicJsonSchema)


def normalize_schema(schema: Mapping[str, Any]) -> Any:
    """Inline ``$ref``s, drop ``$defs``, collapse single-item ``allOf``.

    Discriminator mappings point at ``$defs`` entries by name; they are
    replaced by the referenced definition's ``title`` so renamed classes
    compare equal when their titles (pinned by codegen) do.
    """
    defs: Mapping[str, Any] = schema.get("$defs", {})

    def ref_name(ref: str) -> str:
        return ref.rsplit("/", 1)[-1]

    def resolve(node: Any, stack: frozenset[str]) -> Any:
        if isinstance(node, list):
            return [resolve(item, stack) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            name = ref_name(node["$ref"])
            if name in stack:
                return {"$recursive": defs.get(name, {}).get("title", name)}
            target = resolve(defs[name], stack | {name})
            rest = {k: resolve(v, stack) for k, v in node.items() if k != "$ref"}
            return {**target, **rest}
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key == "$defs":
                continue
            if key == "discriminator" and isinstance(value, dict) and "mapping" in value:
                mapping = {
                    tag: defs.get(ref_name(ref), {}).get("title", ref_name(ref))
                    for tag, ref in value["mapping"].items()
                }
                out[key] = {**value, "mapping": mapping}
                continue
            out[key] = resolve(value, stack)
        all_of = out.get("allOf")
        if isinstance(all_of, list) and len(all_of) == 1 and isinstance(all_of[0], dict):
            inner = out.pop("allOf")[0]
            out = {**inner, **out}
        return out

    return resolve(dict(schema), frozenset())


def _strip_extensions(node: Any) -> Any:
    if isinstance(node, list):
        return [_strip_extensions(item) for item in node]
    if isinstance(node, dict):
        return {k: _strip_extensions(v) for k, v in node.items() if not k.startswith("x-")}
    return node


_DEGRADED_HINT = re.compile(r"\s*\[degraded\] Type was: .* \(not serializable in config\)$")


def strip_degraded_hints(node: Any) -> Any:
    """Remove the per-layer schema's ``[degraded] ...`` description hints.

    ``generate_layer_json_schema`` appends them (with ``x-degraded-from``)
    to degraded properties -- the one documented non-``x-`` addition of a
    published layer schema.  A description that was only the hint goes away.
    """
    if isinstance(node, list):
        return [strip_degraded_hints(item) for item in node]
    if not isinstance(node, dict):
        return node
    out = {k: strip_degraded_hints(v) for k, v in node.items()}
    description = out.get("description")
    if "x-degraded-from" in node and isinstance(description, str):
        stripped = _DEGRADED_HINT.sub("", description)
        if stripped:
            out["description"] = stripped
        else:
            out.pop("description")
    return out


# ---------------------------------------------------------------------------
# Validation outcomes
# ---------------------------------------------------------------------------


def _is_class(tp: Any) -> bool:
    """A real class (on Python 3.10 ``isinstance(list[int], type)`` is true)."""
    return isinstance(tp, type) and not isinstance(tp, types.GenericAlias)


def _reachable_models(obj: Any) -> Iterator[type[BaseModel]]:
    """Every pydantic model class reachable from a type (fields, args, metadata)."""
    seen: set[int] = set()
    stack: list[Any] = [obj]
    while stack:
        tp = stack.pop()
        if id(tp) in seen:
            continue
        seen.add(id(tp))
        if isinstance(tp, FieldInfo):
            stack.append(tp.annotation)
            stack.extend(tp.metadata)
            continue
        if _is_class(tp) and issubclass(tp, BaseModel):
            yield tp
            stack.extend(tp.model_fields.values())
            stack.extend(tp.__mro__[1:])
            continue
        if _is_class(tp):
            continue
        stack.extend(get_args(tp))
        origin = get_origin(tp)
        if origin is not None:
            stack.append(origin)


def _titles(obj: Any) -> dict[str, str]:
    """``{class name: schema title}`` of the models reachable from ``obj``.

    Codegen renames a generated class only on a name collision and then
    pins its in-process name as ``title``; reading every class name as its
    title (on both sides) makes renamed classes compare equal in error
    locations and messages.
    """
    titles: dict[str, str] = {}
    for model in _reachable_models(obj):
        title = model.model_config.get("title")
        titles.setdefault(model.__name__, title if isinstance(title, str) else model.__name__)
    return titles


def _jsonable(value: Any) -> Any:
    if isinstance(value, BaseException):
        return f"{type(value).__name__}: {value}"
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


class _Validator:
    """A cached ``TypeAdapter`` plus the title map of one object."""

    _cache: dict[int, tuple[Any, _Validator]] = {}

    def __init__(self, obj: Any) -> None:
        self.adapter = TypeAdapter(obj)
        self.titles = _titles(obj)
        self.renamed = {name: title for name, title in self.titles.items() if name != title}
        names = sorted(self.renamed, key=len, reverse=True)
        self.pattern = re.compile("|".join(rf"\b{re.escape(n)}\b" for n in names)) if names else None

    def normalize(self, error: Mapping[str, Any]) -> tuple[Any, ...]:
        """``(type, loc, msg, ctx)`` with class names read as schema titles."""
        titles = self.titles
        loc = tuple(titles.get(part, part) if isinstance(part, str) else part for part in error["loc"])
        msg = error["msg"]
        if self.pattern is not None:
            msg = self.pattern.sub(lambda m: self.renamed[m.group(0)], msg)
        ctx = dict(error.get("ctx") or {})
        if isinstance(ctx.get("class_name"), str):
            ctx["class_name"] = titles.get(ctx["class_name"], ctx["class_name"])
        return (error["type"], loc, msg, json.dumps(_jsonable(ctx), sort_keys=True))

    @classmethod
    def of(cls, obj: Any) -> _Validator:
        entry = cls._cache.get(id(obj))
        if entry is None or entry[0] is not obj:
            if len(cls._cache) > 256:  # keep the cache from pinning a whole session's models
                cls._cache.clear()
            entry = (obj, cls(obj))
            cls._cache[id(obj)] = entry
        return entry[1]


def validation_outcome(obj: Any, payload: Any, *, mode: str = "python") -> tuple[str, Any]:
    """``("ok", python-mode dump)`` or ``("error", sorted normalized errors)``.

    Errors are ``(type, loc, msg, ctx)`` tuples; class names in ``loc`` /
    ``msg`` / ``ctx["class_name"]`` are read as schema titles.  The
    python-mode dump keeps non-JSON values (sentinels, enums, paths) as
    objects; both sides import the same objects, so the dumps compare equal.

    Args:
        obj: Model class or type.
        payload: The input.
        mode: ``"python"`` (``validate_python``) or ``"json"``
            (``validate_json`` of the payload's JSON text).
    """
    validator = _Validator.of(obj)
    try:
        if mode == "json":
            value = validator.adapter.validate_json(json.dumps(payload))
        else:
            value = validator.adapter.validate_python(payload)
    except ValidationError as exc:
        errors = [validator.normalize(err) for err in exc.errors()]
        return "error", sorted(errors, key=repr)
    except NotImplementedError as exc:
        # pydantic < 2.8 cannot validate some types from JSON (``type[int]``):
        # an outcome both artifacts must share like any other.
        return "unsupported", f"{type(exc).__name__}: {exc}"
    return "ok", validator.adapter.dump_python(value)


def _json_safe(payload: Any) -> bool:
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError):
        return False
    return True


def same_value(a: Any, b: Any) -> bool:
    """Deep equality that treats ``NaN`` (float or ``Decimal``) as equal to ``NaN``.

    Sequences must also have the same type (a tuple never equals a list).
    """
    try:
        if a == b:  # fast path; ``==`` already tells tuples from lists
            return True
    except Exception:  # noqa: BLE001 -- incomparable values: compare structurally
        pass
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    if isinstance(a, decimal.Decimal) and isinstance(b, decimal.Decimal) and a.is_nan() and b.is_nan():
        return str(a) == str(b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_value(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(
            same_value(x, y) for x, y in zip(a, b)
        )
    return bool(a == b)


def _diff(a: Any, b: Any) -> str:
    left = json.dumps(a, indent=1, sort_keys=True, default=repr).splitlines()
    right = json.dumps(b, indent=1, sort_keys=True, default=repr).splitlines()
    return "\n".join(difflib.unified_diff(left, right, "in-process", "generated", lineterm="", n=2))


# ---------------------------------------------------------------------------
# Samples
# ---------------------------------------------------------------------------

#: Values :func:`perturbations` puts into every field: other types, nulls,
#: empty and non-empty containers, boundary-ish numbers, long strings.
PERTURBATION_VALUES: tuple[Any, ...] = (
    None,
    -1,
    0,
    11,
    2.5,
    True,
    "",
    "zz",
    "x" * 40,
    [],
    ["a", "b"],
    [1, 2, 3],
    {},
    {"zz_unknown": 1},
)


def perturbations(
    base: Mapping[str, Any],
    fields: Iterable[str],
    *,
    values: Iterable[Any] = PERTURBATION_VALUES,
    keep: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Payloads derived from the valid ``base``: one change each.

    For every field in ``fields`` (except ``keep``): ``base`` without the
    field, and ``base`` with the field set to each of ``values``.  Plus
    ``base`` with an unknown key.  The outcomes are not predicted -- the
    artifacts must merely agree on each one.
    """
    kept = set(keep)
    values = list(values)
    out: list[dict[str, Any]] = [{**base, "zz_unknown_key": 1}]
    for name in fields:
        if name in kept:
            continue
        out.append({k: v for k, v in base.items() if k != name})
        out.extend({**base, name: value} for value in values)
    return out


# ---------------------------------------------------------------------------
# The comparison
# ---------------------------------------------------------------------------


def assert_artifacts_agree(
    source: str,
    expected: Mapping[str, Any],
    *,
    samples: Mapping[str, Iterable[Any]] | None = None,
    published: tuple[Mapping[str, Any], Any] | None = None,
    exact_published: tuple[Mapping[str, Any], type[BaseModel]] | None = None,
    plain_namespace: bool | Iterable[str] = False,
    min_accepted: int = 0,
) -> types.ModuleType:
    """Assert the generated ``source`` agrees with the in-process objects.

    Args:
        source: Generated Python source.
        expected: ``{name in the generated module: in-process model or type}``.
        samples: ``{name: payloads}``; each payload must be accepted (with an
            equal dump) or rejected (with the same errors) by both, in python
            mode and -- for JSON-serializable payloads -- in JSON mode.
        published: ``(published_schema, in_process_object)``: the published
            JSON schema must equal the in-process object's schema, ``x-``
            extensions (and the per-layer ``[degraded]`` hints) ignored.
        exact_published: ``(published_schema, in_process_model)``: the
            published schema must equal
            ``in_process_model.model_json_schema()`` exactly.
        plain_namespace: Also execute ``source`` into a plain ``dict`` and
            check that namespace: every schema, and the samples of every
            name (``True``) or of the given names only.
        min_accepted: Fail unless at least this many samples (per name) are
            accepted by the in-process side -- guards against sample sets
            that only exercise rejections.

    Returns:
        The loaded generated module.
    """
    module = load_generated_module(source)
    # (label, namespace, names whose samples run there; None = all)
    namespaces: list[tuple[str, Mapping[str, Any], set[str] | None]] = [
        ("module", vars(module), None),
    ]
    if plain_namespace:
        sampled = None if plain_namespace is True else set(plain_namespace)  # type: ignore[arg-type]
        namespaces.append(("plain dict", exec_generated_namespace(source), sampled))
    problems: list[str] = []
    for name, in_process in expected.items():
        reference = normalize_schema(schema_of(in_process))
        payloads = list((samples or {}).get(name, ()))
        accepted = 0
        outcomes = []
        for payload in payloads:
            per_mode = {"python": validation_outcome(in_process, payload)}
            if _json_safe(payload):
                per_mode["json"] = validation_outcome(in_process, payload, mode="json")
            accepted += per_mode["python"][0] == "ok"
            outcomes.append((payload, per_mode))
        if payloads and accepted < min_accepted:
            problems.append(
                f"{name}: only {accepted} of {len(payloads)} samples are accepted "
                f"in-process (expected at least {min_accepted})"
            )
        for label, namespace, sampled in namespaces:
            if name not in namespace:
                problems.append(f"{name}: not defined by the generated source ({label})")
                continue
            generated = namespace[name]
            other = normalize_schema(schema_of(generated))
            if not same_value(reference, other):
                problems.append(f"{name} ({label}): JSON schemas differ\n{_diff(reference, other)}")
            if sampled is not None and name not in sampled:
                continue
            for payload, per_mode in outcomes:
                for mode, left in per_mode.items():
                    right = validation_outcome(generated, payload, mode=mode)
                    if not same_value(left, right):
                        problems.append(
                            f"{name} ({label}, {mode} mode): payload {payload!r}:\n"
                            f"  in-process {left!r}\n  generated  {right!r}"
                        )
    if published is not None:
        published_schema, in_process = published
        # The composed top-level model carries its x- extensions itself
        # (model_config json_schema_extra); ignore them on both sides.
        a = normalize_schema(_strip_extensions(schema_of(in_process)))
        b = normalize_schema(_strip_extensions(strip_degraded_hints(dict(published_schema))))
        if not same_value(a, b):
            problems.append(f"published schema differs from in-process\n{_diff(a, b)}")
    if exact_published is not None:
        published_schema, model = exact_published
        own = model.model_json_schema(schema_generator=_DeterministicJsonSchema)
        if published_schema != own:
            problems.append(
                f"published schema is not the model's own schema\n{_diff(own, dict(published_schema))}"
            )
    if problems:
        raise AssertionError("artifacts disagree:\n" + "\n\n".join(problems))
    return module


def type_to_source(tp: Any) -> str:
    """Render one type expression on its own (imports implied)."""
    from conscribe.config._render import RenderContext, render_type

    ctx = RenderContext()
    return ctx.ns.materialize(render_type(tp, ctx))


def value_to_source(value: Any) -> str:
    """Render one value on its own (imports implied)."""
    from conscribe.config._render import RenderContext, render_value

    ctx = RenderContext()
    return ctx.ns.materialize(render_value(value, ctx))


def layer_expectations(result: Any) -> dict[str, Any]:
    """``{emitted name: in-process object}`` for ``generate_layer_config_source``."""
    from conscribe.config.codegen import _union_alias_name

    expected: dict[str, Any] = {m.__name__: m for m in result.per_key_models.values()}
    if len(result.per_key_models) > 1:
        expected[_union_alias_name(result.layer_name)] = result.union_type
    return expected


def composed_expectations(result: Any) -> dict[str, Any]:
    """``{emitted name: in-process object}`` for ``generate_composed_config_source``."""
    from conscribe.config.codegen import _composed_alias_names

    expected: dict[str, Any] = {}
    aliases = _composed_alias_names(result)
    for layer in result.dependency_order:
        lr = result.layer_results[layer]
        for model in lr.per_key_models.values():
            expected[model.__name__] = model
        if layer in aliases:
            expected[aliases[layer]] = lr.union_type
    expected[result.top_level_type.__name__] = result.top_level_type
    return expected
