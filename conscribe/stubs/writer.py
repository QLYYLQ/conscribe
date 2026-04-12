"""Write .pyi stub files grouped by source module."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Union

from conscribe.stubs.collector import ClassStubInfo, collect_class_stub_info
from conscribe.stubs.generator import generate_module_stub


def write_layer_stubs(
    registrar: type,
    output_dir: Union[str, Path, None] = None,
) -> list[Path]:
    """Generate and write ``.pyi`` stubs for all wired classes in a layer.

    Args:
        registrar: A ``LayerRegistrar`` subclass (from ``create_registrar``).
        output_dir: If given, mirror module paths under this directory.
            Otherwise write each ``.pyi`` alongside its source ``.py``.

    Returns:
        List of paths to the written ``.pyi`` files.
    """
    all_classes = registrar.get_all()

    # Collect stub info (only classes with injected wired attrs)
    stub_infos: list[ClassStubInfo] = []
    for _key, cls in all_classes.items():
        info = collect_class_stub_info(cls)
        if info is not None:
            stub_infos.append(info)

    if not stub_infos:
        return []

    # Group by source file
    by_module: dict[str, list[ClassStubInfo]] = defaultdict(list)
    for info in stub_infos:
        by_module[info.module].append(info)

    # Generate and write
    out_dir = Path(output_dir) if output_dir is not None else None
    written: list[Path] = []

    for module_name, classes in sorted(by_module.items()):
        source = generate_module_stub(module_name, classes)
        if not source:
            continue

        pyi_path = _resolve_output_path(classes[0].source_file, module_name, out_dir)
        pyi_path.parent.mkdir(parents=True, exist_ok=True)
        pyi_path.write_text(source, encoding="utf-8")
        written.append(pyi_path)

    return written


def _resolve_output_path(
    source_file: str,
    module_name: str,
    output_dir: Path | None,
) -> Path:
    """Determine where to write the .pyi file."""
    if output_dir is None:
        # Write alongside source .py
        return Path(source_file).with_suffix(".pyi")

    # Mirror module structure under output_dir
    parts = module_name.split(".")
    return output_dir / Path(*parts).with_suffix(".pyi")
