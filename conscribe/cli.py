"""Command-line interface: ``conscribe <command> [options]``.

Commands::

    generate-config            one layer -> Python config source (--json-schema adds the schema)
    generate-composed-config   several layers -> composed JSON Schema or Python source
    generate-stubs             one layer -> .pyi stubs for injected wired attributes
    inspect                    one layer -> registered keys and their config fields
    scan                       static AST scan for create_registrar() calls (imports nothing)
    list                       import packages, then list what the runtime registries hold

Loading layers (``generate-*`` and ``inspect``)::

    --registrar MODULE:ATTRIBUTE   import MODULE and use the registrar ATTRIBUTE (repeatable)
    --discover PACKAGE             conscribe.discover(PACKAGE) first (repeatable)
    --allow-missing MODULE         forwarded to discover(allow_missing=...) (repeatable)
    --exclude PREFIX               forwarded to discover(exclude=...) (repeatable)
    --layer NAME / --layers NAME   choose layers by registry name (--layers is repeatable)

The registrars named by ``--registrar`` are imported first, then the
``--discover`` packages, in the order given.  ``generate-config``,
``generate-stubs`` and ``inspect`` work on one layer: the one ``--layer``
names, or else the only ``--registrar`` given.  ``generate-composed-config``
composes the layers ``--layers`` names, or else every ``--registrar`` given.
A layer named by ``--layer`` / ``--layers`` is looked up among the
``--registrar`` values first, then among every registrar loaded by the
imports (created by ``create_registrar()`` and still current).  ``list``
also takes ``--discover`` / ``--allow-missing`` / ``--exclude``.

The current directory is importable while a command runs (as with
``python -m``).  ``discover()`` is strict: a module that fails to import
stops the command with an error naming it; skip it with ``--exclude`` (or
``--allow-missing`` for a missing optional dependency).

Output files: everything is generated before anything is written, and each
file is first written in full next to its target under a temporary name;
only then are the files moved into place (the JSON Schema of
``generate-config --json-schema`` before the source).  A failure while
writing (an unwritable directory, a target that is a directory, a full disk)
leaves every target untouched; the source goes to stdout only after that.
``generate-stubs`` reports conscribe-generated stubs that the run did not
rewrite as ``Warning: ...`` lines on stderr (``--prune`` deletes them).

Exit status: 0 on success; 1 on an error, reported as ``Error: <message>`` on
stderr without a traceback (bad ``MODULE:ATTRIBUTE``, import failures,
unknown or empty layers, registry / wiring / annotation / declaration /
render errors, unwritable output), and when stdout is closed early (e.g.
piped into ``head``; nothing is reported then); 2 on a usage error, a missing
command included (reported by argparse).
"""
from __future__ import annotations

import argparse
import contextlib
import difflib
import importlib
import inspect
import json
import os
import re
import sys
import sysconfig
import traceback
import uuid
import warnings
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from pydantic.errors import PydanticUserError

from conscribe.exceptions import (
    ConscribeWarning,
    EmptyLayerError,
    ExtractionWarning,
    RegistryError,
    StaleStubWarning,
)

#: Errors raised while generating that describe a user mistake: conscribe's
#: own taxonomy, ``ValueError`` (e.g. a registrar without a discriminator),
#: pydantic's verdict on user types, file-system errors on output, and
#: conscribe warnings that the warning filters turned into errors
#: (``-W error::conscribe.ConscribeWarning``).
_GENERATION_ERRORS: tuple[type[BaseException], ...] = (
    RegistryError,
    ConscribeWarning,
    ValueError,
    PydanticUserError,
    OSError,
)

_DISCOVER_NOTE = re.compile(r"importing module '([^']+)'")


class _CLIError(Exception):
    """A user-facing error: printed as ``Error: <message>``, exit status 1."""


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns the exit status (0 = success, 1 = error).

    Usage errors (no command, unknown options, bad choices) and ``--help``
    exit through argparse's ``SystemExit`` (status 2 and 0).  When stdout is
    closed before the output is complete (``conscribe inspect ... | head``),
    the command stops quietly with status 1.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        status = _COMMANDS[args.command](args)
        # Flush here: a closed pipe must surface now, not at interpreter exit.
        sys.stdout.flush()
    except _CLIError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except (RegistryError, ConscribeWarning) as exc:  # wherever a command lets one through
        print(f"Error: {_exception_message(exc)}", file=sys.stderr)
        return 1
    except BrokenPipeError:
        _discard_stdout()
        return 1
    return status


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="conscribe",
        description="Config typing tools for conscribe registries.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>", required=True)

    # generate-config
    gen_parser = subparsers.add_parser(
        "generate-config",
        help="Generate a layer's Python config source file (and optionally its JSON Schema).",
        description="Generate the Python config source (and optionally the JSON Schema) of one layer.",
    )
    _add_loading_options(gen_parser)
    _add_layer_option(gen_parser)
    gen_parser.add_argument(
        "--output", default=None, help="Output file path. Prints to stdout if omitted."
    )
    gen_parser.add_argument(
        "--json-schema",
        dest="json_schema",
        default=None,
        metavar="PATH",
        help=(
            "Also write the layer's JSON Schema to PATH. Both files are written in "
            "full before either replaces its target, the schema first."
        ),
    )

    # generate-stubs
    stub_parser = subparsers.add_parser(
        "generate-stubs",
        help="Generate .pyi stubs for classes with wired attributes.",
        description="Generate .pyi stubs for the classes of one layer that have injected wired attributes.",
    )
    _add_loading_options(stub_parser)
    _add_layer_option(stub_parser)
    stub_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. If omitted, writes .pyi alongside source files.",
    )
    stub_parser.add_argument(
        "--partial-class-fallback",
        action="store_true",
        default=False,
        help=(
            "Emit class-level __getattr__ returning Incomplete to tolerate "
            "dynamic instance attributes (loosens strict type checking)."
        ),
    )
    stub_parser.add_argument(
        "--prune",
        action="store_true",
        default=False,
        help=(
            "Delete the conscribe-generated stubs (recognised by their "
            "'# Generated by conscribe' header) in the affected directories "
            "that this run did not rewrite. Without it they are only reported."
        ),
    )

    # generate-composed-config
    composed_parser = subparsers.add_parser(
        "generate-composed-config",
        help="Generate composed config schema from multiple registries.",
        description="Compose several layers into one JSON Schema or Python config source.",
    )
    _add_loading_options(composed_parser)
    composed_parser.add_argument(
        "--layers",
        action="extend",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "Layers (registry names) to compose. Repeatable. Default: every "
            "layer given with --registrar."
        ),
    )
    composed_parser.add_argument(
        "--inline-wiring",
        action="store_true",
        default=True,
        help="Inline wired fields as target layer union types (default).",
    )
    composed_parser.add_argument(
        "--no-inline-wiring",
        dest="inline_wiring",
        action="store_false",
        help="Keep wired fields as Literal[...] strings.",
    )
    composed_parser.add_argument(
        "--top-level-extra",
        choices=["forbid", "ignore", "allow"],
        default="forbid",
        help=(
            "How the composed top-level model treats keys that are not layer "
            "names: forbid (default), ignore or allow. Inline-only layer names "
            "are refused in every mode."
        ),
    )
    composed_parser.add_argument(
        "--format",
        choices=["python", "json-schema"],
        default="json-schema",
        help="Output format (default: json-schema).",
    )
    composed_parser.add_argument(
        "--output",
        default=None,
        help="Output file path. Prints to stdout if omitted.",
    )

    # inspect
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect a registry's keys and config fields.",
        description="Print the registered keys of one layer and the config fields of each.",
    )
    _add_loading_options(inspect_parser)
    _add_layer_option(inspect_parser)

    # scan
    scan_parser = subparsers.add_parser(
        "scan",
        help="Scan Python files for registrar definitions (static analysis).",
    )
    scan_parser.add_argument(
        "--path",
        default=".",
        help="Root directory to scan (default: current directory).",
    )

    # list
    list_parser = subparsers.add_parser(
        "list",
        help="List registered content from runtime registries.",
    )
    list_parser.add_argument(
        "--discover",
        action="extend",
        nargs="+",
        dest="discover_packages",
        default=None,
        metavar="PACKAGE",
        help="Packages to import for registration (auto-detected if omitted).",
    )
    _add_discover_filters(list_parser)
    list_parser.add_argument(
        "--layer",
        default=None,
        help="Show only this registry (by name).",
    )
    list_parser.add_argument(
        "--path",
        default=None,
        help="Filter entries by source file path prefix.",
    )

    return parser


def _add_loading_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--registrar",
        action="extend",
        nargs="+",
        default=None,
        metavar="MODULE:ATTRIBUTE",
        help=(
            "Registrar to use: import MODULE and take its attribute ATTRIBUTE "
            "(a class returned by create_registrar()). Repeatable."
        ),
    )
    parser.add_argument(
        "--discover",
        action="extend",
        nargs="+",
        default=None,
        metavar="PACKAGE",
        help=(
            "Import every module of PACKAGE first (conscribe.discover), so that "
            "implementations register. Repeatable."
        ),
    )
    _add_discover_filters(parser)


def _add_discover_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--allow-missing",
        action="extend",
        nargs="+",
        default=None,
        metavar="MODULE",
        help=(
            "Top-level name of an optional dependency: modules that fail to "
            "import because it is missing are skipped "
            "(discover(allow_missing=...)). Repeatable."
        ),
    )
    parser.add_argument(
        "--exclude",
        action="extend",
        nargs="+",
        default=None,
        metavar="PREFIX",
        help="Module or package (and everything below it) never imported (discover(exclude=...)). Repeatable.",
    )


def _add_layer_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--layer",
        default=None,
        metavar="NAME",
        help="Layer (registry name) to use. Optional when exactly one --registrar is given.",
    )


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _cmd_generate_config(args: argparse.Namespace) -> int:
    """Handle the generate-config subcommand."""
    from conscribe.config.builder import build_layer_config
    from conscribe.config.codegen import generate_layer_config_source
    from conscribe.config.json_schema import generate_layer_json_schema

    if (
        args.output is not None
        and args.json_schema is not None
        and Path(args.output).resolve() == Path(args.json_schema).resolve()
    ):
        raise _CLIError(
            f"--output and --json-schema name the same file ({args.output}); "
            "give the JSON Schema its own path"
        )

    with _loaded_layers(args) as given:
        registrar = _select_layer(given, args.layer)
        with _user_errors():
            result = build_layer_config(registrar)
            source = generate_layer_config_source(result)
            outputs: list[tuple[str, str | None]] = []
            if args.json_schema is not None:
                outputs.append((_json_text(generate_layer_json_schema(result)), args.json_schema))
            outputs.append((source, args.output))
            # Everything is generated before anything is written; the schema
            # is moved into place first.
            _write_outputs(outputs)
    return 0


def _cmd_generate_composed_config(args: argparse.Namespace) -> int:
    """Handle the generate-composed-config subcommand."""
    from conscribe.config.codegen import generate_composed_config_source
    from conscribe.config.composed import build_composed_config
    from conscribe.config.json_schema import generate_composed_json_schema

    with _loaded_layers(args) as given:
        registrars = _select_layers(given, args.layers)
        with _user_errors():
            result = build_composed_config(
                registrars,
                inline_wiring=args.inline_wiring,
                top_level_extra=args.top_level_extra,
            )
            if args.format == "json-schema":
                content = _json_text(generate_composed_json_schema(result))
            else:
                content = generate_composed_config_source(result)
            _write_outputs([(content, args.output)])
    return 0


def _cmd_generate_stubs(args: argparse.Namespace) -> int:
    """Handle the generate-stubs subcommand.

    Stale conscribe-generated stubs that the writer reports
    (:class:`~conscribe.exceptions.StaleStubWarning`) are printed as
    ``Warning: ...`` lines on stderr; ``--prune`` asks the writer to delete
    them.
    """
    from conscribe.stubs.writer import write_layer_stubs

    options: dict[str, Any] = {"partial_class_fallback": args.partial_class_fallback}
    if args.prune:
        options["prune"] = True

    with _loaded_layers(args) as given:
        registrar = _select_layer(given, args.layer)
        with _user_errors(), _stale_stub_warnings_as_lines():
            if not registrar.get_all():
                raise EmptyLayerError([_layer_name(registrar)])
            written = write_layer_stubs(registrar, args.output_dir, **options)

    if not written:
        print("No classes with injected wired attributes found.", file=sys.stderr)
        return 0

    for path in written:
        print(f"  {path}")
    print(f"Generated {len(written)} stub file(s).")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    """Handle the inspect subcommand.

    Every key is shown; a key whose config cannot be built is reported as an
    ``Error:`` line on stderr (after the listing) and the exit status is 1.
    """
    with _loaded_layers(args) as given:
        registrar = _select_layer(given, args.layer)
        all_classes = registrar.get_all()
        print(f"Layer: {_layer_name(registrar)}")
        nested = getattr(registrar, "discriminator_fields", None)
        if nested:
            separator = getattr(registrar, "_key_separator", "")
            print(f"Discriminator: {', '.join(nested)} (nested, key separator {separator!r})")
        else:
            print(f"Discriminator: {getattr(registrar, 'discriminator_field', '')}")
        if getattr(registrar, "inline_only", False):
            print("Inline-only: yes")
        print(f"Registered keys ({len(all_classes)}):")

        errors: list[str] = []
        for key in sorted(all_classes, key=str):
            cls = all_classes[key]
            print(f"\n  {key} ({cls.__name__}):")
            try:
                with _user_errors():
                    schema = registrar.get_config_schema(key)
            except _CLIError as exc:
                errors.append(f"{key} ({cls.__qualname__}): {exc}")
                print("    (config cannot be built, see the error below)")
                continue
            fields = getattr(schema, "model_fields", None) or {}
            if not fields:
                print("    (no config parameters)")
                continue
            wired = getattr(schema, "__wired_fields__", None) or {}
            for field_name, field_info in fields.items():
                print(f"    {_describe_field(field_name, field_info, wired.get(field_name))}")

    for message in errors:
        print(f"Error: {message}", file=sys.stderr)
    return 1 if errors else 0


def _cmd_scan(args: argparse.Namespace) -> int:
    """Handle the scan subcommand."""
    from conscribe.scanner import scan_registrar_definitions

    root = Path(args.path).resolve()

    if not root.is_dir():
        raise _CLIError(f"{root} is not a directory.")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ExtractionWarning)
        definitions = scan_registrar_definitions(root)
    for warning in caught:
        print(f"Warning: {warning.message}", file=sys.stderr)

    if not definitions:
        print(f"No registrar definitions found under {root}.")
        return 0

    print(f"Found {len(definitions)} registrar(s):\n")
    for defn in definitions:
        rel_path = defn.file_path
        try:
            rel_path = str(Path(defn.file_path).relative_to(root))
        except ValueError:
            pass
        var_part = f"  (var: {defn.variable_name})" if defn.variable_name else ""
        print(
            f"  {defn.name:<16}"
            f"{defn.protocol_name:<20}"
            f"{rel_path}:{defn.line_number}"
            f"{var_part}"
        )

    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    """Handle the list subcommand."""
    from conscribe.scanner import find_packages, summarize_registries

    root = Path.cwd()
    packages = args.discover_packages

    if packages is None:
        packages = find_packages(root)
        if not packages:
            raise _CLIError(
                "no Python packages found under current directory.\n"
                "Use --discover to specify packages explicitly."
            )

    path_filter = None if args.path is None else _path_prefix(args.path, root)
    with _cwd_importable():
        _discover(packages, allow_missing=args.allow_missing, exclude=args.exclude)
        summaries = summarize_registries(root, path_filter=path_filter)

    if args.layer is not None:
        selected = [s for s in summaries if s.name == args.layer]
        if not selected:
            available = ", ".join(s.name for s in summaries) or "none"
            raise _CLIError(f"registry {args.layer!r} not found. Available: {available}.")
        summaries = selected

    if not summaries:
        print("No registries found.")
        return 0

    if path_filter is not None:
        # Hide empty registries when filtering by path
        summaries = [s for s in summaries if s.entry_count]
        if not summaries:
            where = "" if args.layer is None else f" in registry {args.layer!r}"
            print(
                f"No registered classes{where} are defined under {path_filter!r} "
                f"(--path is a prefix of source paths relative to {root})."
            )
            return 0

    for summary in summaries:
        print(
            f"Registry: {summary.name} ({summary.protocol_name})"
            f" — {summary.entry_count} entries"
        )
        if not summary.entries:
            print("  (empty)")
        else:
            for entry in summary.entries:
                loc = ""
                if entry.file_path is not None:
                    loc = entry.file_path
                    if entry.line_number is not None:
                        loc += f":{entry.line_number}"
                print(
                    f"  {entry.key:<20}"
                    f"{entry.class_name:<24}"
                    f"{loc}"
                )
        print()

    return 0


_COMMANDS = {
    "generate-config": _cmd_generate_config,
    "generate-composed-config": _cmd_generate_composed_config,
    "generate-stubs": _cmd_generate_stubs,
    "inspect": _cmd_inspect,
    "scan": _cmd_scan,
    "list": _cmd_list,
}


# ---------------------------------------------------------------------------
# Loading registrars
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _loaded_layers(args: argparse.Namespace) -> Iterator[list[type]]:
    """Import ``--registrar`` modules and ``--discover`` packages.

    Yields the registrars named by ``--registrar`` (in order, without
    duplicates).  The current directory stays importable until the block
    ends.
    """
    specs = [(spec, *_parse_registrar_spec(spec)) for spec in args.registrar or ()]
    with _cwd_importable():
        given: list[type] = []
        for spec, module_name, attribute in specs:
            registrar = _load_registrar(spec, module_name, attribute)
            if all(registrar is not seen for seen in given):
                given.append(registrar)
        _discover(args.discover or (), allow_missing=args.allow_missing, exclude=args.exclude)
        yield given


def _parse_registrar_spec(spec: str) -> tuple[str, str]:
    """Split ``MODULE:ATTRIBUTE``; reject anything else."""
    module_name, colon, attribute = spec.partition(":")
    module_name, attribute = module_name.strip(), attribute.strip()
    if not colon or not module_name or not attribute or ":" in attribute:
        raise _CLIError(
            "--registrar expects MODULE:ATTRIBUTE (for example "
            f"my_app.layers:AgentRegistrar), got {spec!r}"
        )
    return module_name, attribute


def _load_registrar(spec: str, module_name: str, attribute: str) -> type:
    """Import *module_name* and return the registrar at *attribute* (dotted)."""
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - user code: any failure is reported, without a traceback
        raise _CLIError(
            f"--registrar {spec!r}: could not import module {module_name!r}: "
            f"{_describe_failure(exc)}"
        ) from None

    obj: Any = module
    walked: list[str] = []
    for part in attribute.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError:
            owner = module_name if not walked else f"{module_name}:{'.'.join(walked)}"
            hint = _registrars_in(module) if not walked else ""
            raise _CLIError(
                f"--registrar {spec!r}: {owner!r} has no attribute {part!r}{hint}"
            ) from None
        walked.append(part)

    if not _is_registrar(obj):
        if isinstance(obj, type):
            got = f"class {obj.__module__}.{obj.__qualname__}"
        else:
            got = f"{type(obj).__name__} {_short_repr(obj)}"
        raise _CLIError(
            f"--registrar {spec!r}: expected a registrar returned by create_registrar(), got {got}"
        )
    return obj


def _registrars_in(module: Any) -> str:
    """`` (registrars defined there: A, B)`` for an error message, or ``""``."""
    names = sorted(
        name for name, value in vars(module).items() if _is_registrar(value)
    )
    return f" (registrars defined there: {', '.join(names)})" if names else ""


def _is_registrar(obj: Any) -> bool:
    from conscribe.registration.registrar import LayerRegistrar

    return (
        isinstance(obj, type)
        and issubclass(obj, LayerRegistrar)
        and obj is not LayerRegistrar
        and getattr(obj, "_registry", None) is not None
    )


def _discover(
    packages: Iterable[str],
    *,
    allow_missing: Iterable[str] | None,
    exclude: Iterable[str] | None,
) -> None:
    """``discover()`` each package (strict); import failures become ``_CLIError``."""
    from conscribe.discover import discover

    allowed = tuple(allow_missing or ())
    excluded = tuple(exclude or ())
    for package in packages:
        try:
            discover(package, allow_missing=allowed, exclude=excluded)
        except Exception as exc:  # noqa: BLE001 - user code: any failure is reported, without a traceback
            raise _CLIError(_discover_failure(package, exc)) from None


def _discover_failure(package: str, exc: BaseException) -> str:
    """Message for a module that failed to import during ``discover(package)``."""
    module_name = package
    cause: BaseException = exc
    for note in getattr(exc, "__notes__", None) or ():
        match = _DISCOVER_NOTE.search(note) if isinstance(note, str) else None
        if match:
            module_name = match.group(1)
    if (
        isinstance(exc, ImportError)
        and str(exc).startswith("conscribe.discover() could not import module")
        and exc.__cause__ is not None
    ):
        # Python 3.10 (no exception notes): discover() wraps a failing submodule.
        module_name = exc.name or module_name
        cause = exc.__cause__

    message = f"could not import module {module_name!r} (--discover {package}): {_describe_failure(cause)}"
    if module_name == package:
        return message
    missing = getattr(cause, "name", None) if isinstance(cause, ModuleNotFoundError) else None
    dependency = missing.split(".")[0] if isinstance(missing, str) and missing else None
    if dependency is not None and dependency != package.split(".")[0]:
        return (
            f"{message}\nIf {dependency!r} is an optional dependency, skip the modules that "
            f"need it with --allow-missing {dependency}; or skip this module with "
            f"--exclude {module_name}."
        )
    return f"{message}\nFix the module, or skip it with --exclude {module_name}."


def _select_layer(given: list[type], layer: str | None) -> type:
    """The single registrar a one-layer command works on."""
    if layer is not None:
        return _find_layer(layer, given)
    if len(given) == 1:
        return given[0]
    if not given:
        raise _CLIError(
            "no layer selected: pass --registrar MODULE:ATTRIBUTE "
            "(or --layer NAME together with --discover PACKAGE)"
        )
    names = ", ".join(_layer_name(r) for r in given)
    raise _CLIError(f"{len(given)} registrars given ({names}); choose one with --layer NAME")


def _select_layers(given: list[type], layers: list[str] | None) -> dict[str, type]:
    """``{layer name: registrar}`` for generate-composed-config."""
    selected: dict[str, type] = {}
    if layers:
        for name in layers:
            if name not in selected:
                selected[name] = _find_layer(name, given)
        return selected
    if not given:
        raise _CLIError(
            "no layers selected: pass --registrar MODULE:ATTRIBUTE for each layer "
            "(or --layers NAME ... together with --discover PACKAGE)"
        )
    for registrar in given:
        name = _layer_name(registrar)
        if name in selected:
            raise _CLIError(
                f"two different registrars of layer {name!r} were given with --registrar"
            )
        selected[name] = registrar
    return selected


def _find_layer(name: str, given: list[type]) -> type:
    for registrar in given:
        if _layer_name(registrar) == name:
            return registrar
    try:
        return _get_registrar(name)
    except KeyError as exc:
        raise _CLIError(_exception_message(exc)) from None


def _get_registrar(layer_name: str) -> type:
    """Look up a loaded registrar by layer (registry) name.

    A registrar is loaded when the module that called ``create_registrar()``
    has been imported and its registry is still the current one of that
    name (a later ``create_registrar()`` with the same name replaces it).

    Raises:
        KeyError: If no loaded registrar has that name; the message lists
            the loaded layers.
    """
    loaded = _loaded_registrars()
    for registrar in loaded:
        if _layer_name(registrar) == layer_name:
            return registrar
    names = sorted({_layer_name(r) for r in loaded})
    close = difflib.get_close_matches(layer_name, names, n=1, cutoff=0.6)
    hint = f" (did you mean {close[0]!r}?)" if close else ""
    listed = ", ".join(names) if names else "none"
    raise KeyError(
        f"no registrar for layer {layer_name!r} is loaded{hint}; loaded layers: {listed}. "
        "Pass --registrar MODULE:ATTRIBUTE, or --discover the package that creates it."
    )


def _loaded_registrars() -> list[type]:
    """Every registrar created by ``create_registrar()`` whose registry is current."""
    from conscribe.registration.registrar import LayerRegistrar
    from conscribe.registration.registry import get_registry

    loaded: list[type] = []
    for registrar in LayerRegistrar.__subclasses__():
        registry = getattr(registrar, "_registry", None)
        name = getattr(registry, "name", None)
        if isinstance(name, str) and get_registry(name) is registry:
            loaded.append(registrar)
    return loaded


def _layer_name(registrar: type) -> str:
    return registrar._registry.name  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _cwd_importable() -> Iterator[None]:
    """Put the current directory on ``sys.path`` for the duration of a command."""
    importlib.invalidate_caches()  # files may have appeared since the last import
    cwd = os.getcwd()
    added = cwd not in sys.path and "" not in sys.path
    if added:
        sys.path.insert(0, cwd)
    try:
        yield
    finally:
        if added:
            with contextlib.suppress(ValueError):
                sys.path.remove(cwd)


@contextlib.contextmanager
def _user_errors() -> Iterator[None]:
    """Turn the user-facing errors of a generation step into ``_CLIError``."""
    try:
        yield
    except BrokenPipeError:
        raise  # stdout was closed early: main() stops quietly
    except _GENERATION_ERRORS as exc:
        raise _CLIError(_exception_message(exc)) from None


@contextlib.contextmanager
def _stale_stub_warnings_as_lines() -> Iterator[None]:
    """Print each ``StaleStubWarning`` as a ``Warning: ...`` line on stderr.

    Always shown (whatever the warning filters say); every other warning is
    left to the active filters and the original ``showwarning``.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("always", StaleStubWarning)
        show = warnings.showwarning

        def show_stale_as_line(
            message: Warning | str,
            category: type[Warning],
            filename: str,
            lineno: int,
            file: Any = None,
            line: str | None = None,
        ) -> None:
            if issubclass(category, StaleStubWarning):
                print(f"Warning: {message}", file=sys.stderr)
            else:
                show(message, category, filename, lineno, file, line)

        warnings.showwarning = show_stale_as_line
        yield


def _write_outputs(outputs: Sequence[tuple[str, str | None]]) -> None:
    """Write each ``(content, path)`` in order; a ``None`` path means stdout.

    Every file is first written in full to a temporary file next to its
    target (parent directories created), then all of them are moved into
    place in the order given, and only then is anything written to stdout.
    A failure while writing (an unwritable directory, a target that is a
    directory, a full disk) therefore leaves every target untouched; the
    temporary files are removed.
    """
    staged: list[tuple[Path, Path]] = []
    try:
        for content, output in outputs:
            if output is None:
                continue
            target = Path(output)
            if target.is_symlink():
                target = target.resolve()  # write through the link, keep the link
            if target.is_dir():
                raise IsADirectoryError(f"{output}: is a directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex[:12]}.tmp")
            with open(temporary, "x", encoding="utf-8") as handle:
                staged.append((temporary, target))
                handle.write(content)
            if target.exists():
                with contextlib.suppress(OSError):
                    os.chmod(temporary, target.stat().st_mode & 0o7777)
        for temporary, target in staged:
            os.replace(temporary, target)
    finally:
        for temporary, _ in staged:
            with contextlib.suppress(OSError):  # moved into place already, or not removable
                temporary.unlink()
    for content, output in outputs:
        if output is None:
            sys.stdout.write(content if content.endswith("\n") else content + "\n")


def _discard_stdout() -> None:
    """Point stdout at the null device after a ``BrokenPipeError``.

    Python flushes stdout once more at exit; without this, that flush fails
    again on the closed pipe and prints an ``Exception ignored`` message
    (see "Note on SIGPIPE" in the :mod:`signal` documentation).
    """
    try:
        descriptor = sys.stdout.fileno()
    except (AttributeError, OSError, ValueError):  # replaced stdout (tests, embedding)
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, descriptor)
    finally:
        os.close(devnull)


def _path_prefix(path: str, root: Path) -> str:
    """``list --path`` as a prefix of the root-relative source paths it filters.

    ``./pkg`` and an absolute path inside *root* select the same entries as
    ``pkg``; a trailing separator is kept (``pkg/sub/`` does not match
    ``pkg/sub_other/``).
    """
    trailing = path.endswith(("/", os.sep))
    prefix = os.path.normpath(path)
    if os.path.isabs(prefix):
        try:
            relative = os.path.relpath(os.path.realpath(prefix), root.resolve())
        except ValueError:  # another drive (Windows)
            relative = os.pardir
        if relative == os.pardir or relative.startswith(os.pardir + os.sep):
            return prefix  # outside root: matches nothing, and the message says why
        prefix = relative
    if prefix == os.curdir:
        return ""
    return prefix + os.sep if trailing else prefix


def _json_text(data: Any) -> str:
    return json.dumps(data, indent=2) + "\n"


def _describe_field(name: str, info: Any, wired_from: str | None) -> str:
    """``name: type = default  # description; wired from: registry``."""
    text = f"{name}: {_format_type(info.annotation)}"
    if getattr(info, "default_factory", None) is not None:
        text += " = <factory>"
    elif not info.is_required():
        text += f" = {info.default!r}"
    notes = []
    if info.description:
        notes.append(info.description)
    if wired_from:
        notes.append(f"wired from: {wired_from}")
    if notes:
        text += f"  # {'; '.join(notes)}"
    return text


def _format_type(annotation: Any) -> str:
    try:
        return inspect.formatannotation(annotation)
    except Exception:  # noqa: BLE001 - display only
        return repr(annotation)


def _exception_message(exc: BaseException) -> str:
    """``str(exc)``, without the quotes ``KeyError`` adds around its message."""
    if isinstance(exc, KeyError) and len(exc.args) == 1 and isinstance(exc.args[0], str):
        return exc.args[0]
    try:
        return str(exc)
    except Exception:  # noqa: BLE001 - a broken __str__ in user code; the type still helps
        return f"<{type(exc).__name__} whose message cannot be shown>"


def _describe_failure(exc: BaseException) -> str:
    """``Type: message (file, line N)`` plus any exception notes, for import failures."""
    message = _exception_message(exc)
    text = f"{type(exc).__name__}: {message}" if message else type(exc).__name__
    if not isinstance(exc, SyntaxError):  # its message already names file and line
        location = _user_location(exc)
        if location:
            text += f" ({location})"
    for note in getattr(exc, "__notes__", None) or ():
        if isinstance(note, str) and not note.startswith("(raised while conscribe.discover()"):
            text += f"\n  {note}"
    return text


def _user_location(exc: BaseException) -> str | None:
    """``path, line N`` of the innermost traceback frame in user code."""
    roots = _library_roots()
    for frame in reversed(traceback.extract_tb(exc.__traceback__)):
        filename = frame.filename
        if filename.startswith("<"):
            continue  # <frozen importlib._bootstrap>, <string>
        path = os.path.abspath(filename)
        if any(path == root or path.startswith(root + os.sep) for root in roots):
            continue
        try:
            shown = os.path.relpath(path)
        except ValueError:  # another drive (Windows)
            shown = path
        if shown.startswith(".."):
            shown = path
        return f"{shown}, line {frame.lineno}"
    return None


def _library_roots() -> list[str]:
    """Directories whose frames are not the user's: conscribe, stdlib, site-packages."""
    import conscribe

    roots = {os.path.dirname(os.path.abspath(conscribe.__file__))}
    paths = sysconfig.get_paths()
    for key in ("stdlib", "platstdlib", "purelib", "platlib"):
        value = paths.get(key)
        if value:
            roots.add(os.path.abspath(value))
    return sorted(roots)


def _short_repr(obj: Any, limit: int = 80) -> str:
    try:
        text = repr(obj)
    except Exception:  # noqa: BLE001 - display only
        text = f"<{type(obj).__name__} object>"
    return text if len(text) <= limit else text[: limit - 3] + "..."
