"""CLI entry point for conscribe.

Provides commands for generating config stubs and inspecting registries.

Commands:
    generate-config --layer <name> [--output <path>]
    inspect --layer <name>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Union


def _get_registrar(layer_name: str) -> type:
    """Look up a registrar by layer name.

    This is the default implementation — meant to be overridden
    or patched in tests. In production, this would look up from
    a global registry or config file.

    Raises:
        KeyError: If the layer name is not found.
    """
    raise KeyError(f"No registrar found for layer {layer_name!r}")


def main(argv: Union[list[str], None] = None) -> int:
    """CLI entry point. Returns exit code (0=success, 1=error)."""
    parser = argparse.ArgumentParser(
        prog="conscribe",
        description="Config typing tools for conscribe registries.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # generate-config
    gen_parser = subparsers.add_parser(
        "generate-config",
        help="Generate a Python config stub file from a registry.",
    )
    gen_parser.add_argument(
        "--layer", required=True, help="Layer name (e.g. 'llm', 'agent')."
    )
    gen_parser.add_argument(
        "--output", default=None, help="Output file path. Prints to stdout if omitted."
    )

    # generate-stubs
    stub_parser = subparsers.add_parser(
        "generate-stubs",
        help="Generate .pyi stubs for classes with wired attributes.",
    )
    stub_parser.add_argument(
        "--layer", required=True, help="Layer name (e.g. 'llm', 'agent')."
    )
    stub_parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. If omitted, writes .pyi alongside source files.",
    )

    # inspect
    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect a registry's keys and config fields.",
    )
    inspect_parser.add_argument(
        "--layer", required=True, help="Layer name (e.g. 'llm', 'agent')."
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_usage(sys.stderr)
        return 1

    if args.command == "generate-config":
        return _cmd_generate_config(args.layer, args.output)
    elif args.command == "generate-stubs":
        return _cmd_generate_stubs(args.layer, args.output_dir)
    elif args.command == "inspect":
        return _cmd_inspect(args.layer)

    return 1


def _cmd_generate_config(layer_name: str, output: Union[str, None]) -> int:
    """Handle the generate-config subcommand."""
    from conscribe.config.builder import build_layer_config
    from conscribe.config.codegen import generate_layer_config_source

    try:
        registrar = _get_registrar(layer_name)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    result = build_layer_config(registrar)
    source = generate_layer_config_source(result)

    if output is not None:
        Path(output).write_text(source, encoding="utf-8")
    else:
        print(source)

    return 0


def _cmd_generate_stubs(layer_name: str, output_dir: Union[str, None]) -> int:
    """Handle the generate-stubs subcommand."""
    from conscribe.stubs.writer import write_layer_stubs

    try:
        registrar = _get_registrar(layer_name)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    written = write_layer_stubs(registrar, output_dir)

    if not written:
        print("No classes with injected wired attributes found.", file=sys.stderr)
        return 0

    for path in written:
        print(f"  {path}")
    print(f"Generated {len(written)} stub file(s).")
    return 0


def _cmd_inspect(layer_name: str) -> int:
    """Handle the inspect subcommand."""
    from conscribe.config.extractor import extract_config_schema

    try:
        registrar = _get_registrar(layer_name)
    except KeyError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    all_classes = registrar.get_all()
    print(f"Layer: {layer_name}")
    print(f"Discriminator: {registrar.discriminator_field}")
    print(f"Registered keys ({len(all_classes)}):")

    for key in sorted(all_classes.keys()):
        cls = all_classes[key]
        schema = extract_config_schema(cls)
        print(f"\n  {key} ({cls.__name__}):")
        if schema is not None:
            for field_name, field_info in schema.model_fields.items():
                type_name = getattr(field_info.annotation, "__name__", str(field_info.annotation))
                desc = f"  # {field_info.description}" if field_info.description else ""
                default_str = ""
                if not field_info.is_required():
                    default_str = f" = {field_info.default!r}"
                print(f"    {field_name}: {type_name}{default_str}{desc}")
        else:
            print("    (no config parameters)")

    return 0
