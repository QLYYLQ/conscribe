# Changelog

All notable changes to conscribe are documented here.

The format is loosely based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.3.0]

Two fixes to `__wiring__`, both of which remove a way for a declaration to be
silently lost or over-constrained. Neither changes the output for any wiring
declaration that was already legal, so upgrading regenerates byte-identical
config for existing consumers.

### Added

- **Capability-relative wiring keys.** When the target registry was created
  with a `key_separator` (e.g. `create_registrar(..., key_separator=".")`),
  the explicit key lists of the tuple modes may now name a bare **trailing
  segment** instead of a fully-qualified key:

  ```python
  # Registered: "browser.click", "desktop.click", "browser.scroll"
  __wiring__ = {"action": ("action", ["click"], ["scroll"])}
  # -> Literal["browser.click", "desktop.click", "browser.scroll"]
  ```

  Previously only `"browser.click"` was accepted and `"click"` was rejected as
  a missing key, which type-welded every declaring class to exactly one
  provider. A key that already contains the separator is passed through
  **unchanged**, so pinning a specific provider still works and remains the
  escape hatch. Registries without a `key_separator` are completely unaffected.

  This is **not** a new `__wiring__` grammar mode — the three existing modes
  are unchanged; only the matching of the key strings inside them is relaxed.

  If a short name matches several providers, *all* of them are included in the
  generated `Literal[...]`. Conscribe expands, it does not arbitrate: at config
  generation time it cannot know which providers a given assembled runtime will
  hold, so every candidate is legitimately possible. Narrowing to one belongs to
  the consuming framework, at the point where it knows the concrete provider set.

- `conscribe.wiring.expand_relative_keys(declared, registry_keys, separator)` —
  the expansion rule, exposed so consumers can project their own key spaces the
  same way.

### Fixed

- **`__wiring__` receptors are no longer dropped by a parameter-less
  `__init__`.** `extract_config_schema()` returned `None` as soon as
  `inspect.signature(cls.__init__)` yielded no named parameters — before
  `__wiring__` was applied. A class that declared no config knobs of its own
  therefore produced a discriminator-only model, and every receptor it declared
  or inherited through `__wiring__` disappeared silently: the dependency became
  unselectable from config and bound to a default instead. The only cure was to
  widen the signature with a throw-away parameter.

  Wiring is now applied first, and the "nothing to extract" bail fires only when
  the class contributes neither named parameters *nor* wiring receptors. Classes
  with no wiring are unaffected and still yield `None`.

### Changed

- The project tagline "`__init__` signature is config schema" was accurate but
  incomplete, and the incompleteness was load-bearing for the bug above. It now
  reads "`__init__` signature *plus* `__wiring__` is config schema"
  (`README.md`, `conscribe/llms.txt`).

### Known limitation

- The sibling bail at `config/extractor.py` — `find_init_definer(cls) is None`,
  i.e. *no* class in the MRO defines `__init__` at all — still returns `None`
  before wiring is applied, so such a class also loses its receptors. It was
  left alone deliberately: it is a different code path from the one fixed here
  and was not covered by the measurement that validated this release. Give the
  class an `__init__` (even an empty one) as a workaround.

## [1.2.0]

- Fixed the shape of wired fields in generated configs and made `.pyi` output
  valid. Container-annotated receptors become `list[Literal[...]]`; slot-less
  receptors become `Optional[Literal[...]] = None`.

## [1.1.2]

- Emit nested submodels in composed / nested config source.

## [1.1.1]

- Alias-faithful imports and class-level attributes in `.pyi` stubs.

## [1.1.0]

- Resolve string annotations and `TYPE_CHECKING` imports in `.pyi` stubs.

## [1.0.0]

- `WiredField` descriptor for wired attribute injection.

## [0.9.0]

- CLI `scan` and `list` commands for registry discovery.

## [0.8.0]

- Composed config with inline wiring for multi-layer YAML schema.

## [0.7.0]

- Generate `.pyi` stubs for wired class attributes.

## [0.6.1]

- 3-element `__wiring__` tuple for required / optional key subsets.

## [0.6.0]

- Cross-registry wiring via `__wiring__` for config generation.

## [0.5.0]

- Hierarchical keys, cross-registry diamond inheritance, predicate filters,
  nested config mode.
