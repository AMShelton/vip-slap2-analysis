"""Utilities for generating simple filesystem manifests.

This module is intended to live in ``vip_slap2_analysis/utils`` and provides a
small reusable function plus a command-line interface for writing directory tree
manifests as TSV files.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable, Literal, Optional

ManifestKind = Literal["file", "dir"]


def generate_directory_manifest(
    root: str | Path,
    out_path: str | Path | None = None,
    *,
    include_root: bool = False,
    include_hidden: bool = True,
    relative: bool = True,
    sort: bool = True,
    encoding: str = "utf-8",
) -> list[tuple[str, ManifestKind]]:
    """Generate and optionally save a directory manifest for an arbitrary path.

    Parameters
    ----------
    root
        Root directory to scan recursively.
    out_path
        Optional TSV output path. If provided, the manifest is written with two
        columns: ``path`` and ``kind``.
    include_root
        Include the root directory itself as ``.`` in the manifest.
    include_hidden
        Include hidden files/directories such as ``.DS_Store`` and ``.git``.
    relative
        Store paths relative to ``root``. If false, store absolute paths.
    sort
        Sort rows lexicographically by path before returning/writing.
    encoding
        Text encoding used when writing the TSV.

    Returns
    -------
    list[tuple[str, Literal["file", "dir"]]]
        Manifest rows as ``(path, kind)`` tuples.
    """
    root = Path(root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root path does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root}")

    rows: list[tuple[str, ManifestKind]] = []

    def keep(path: Path) -> bool:
        if include_hidden:
            return True
        rel_parts = path.relative_to(root).parts
        return not any(part.startswith(".") for part in rel_parts)

    if include_root:
        rows.append(("." if relative else str(root), "dir"))

    for path in root.rglob("*"):
        if not keep(path):
            continue
        label = str(path.relative_to(root)) if relative else str(path)
        kind: ManifestKind = "dir" if path.is_dir() else "file"
        rows.append((label, kind))

    if sort:
        rows = sorted(rows, key=lambda x: (x[0].lower(), x[1]))

    if out_path is not None:
        out_path = Path(out_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_manifest_tsv(rows, out_path, encoding=encoding)

    return rows


def write_manifest_tsv(
    rows: Iterable[tuple[str, ManifestKind]],
    out_path: str | Path,
    *,
    encoding: str = "utf-8",
) -> Path:
    """Write manifest rows to a two-column TSV file.

    Parameters
    ----------
    rows
        Iterable of ``(path, kind)`` tuples.
    out_path
        Destination TSV path.
    encoding
        Text encoding for the output file.

    Returns
    -------
    pathlib.Path
        Resolved output path.
    """
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding=encoding) as f:
        writer = csv.writer(f, delimiter="\t")
        for path, kind in rows:
            writer.writerow([path, kind])
    return out_path


def build_argparser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Generate a simple two-column TSV manifest for a directory tree."
    )
    parser.add_argument("root", type=Path, help="Root directory to scan.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output TSV path. Defaults to <root>/<root_name>_tree_manifest.tsv.",
    )
    parser.add_argument("--include-root", action="store_true", help="Include root as '.'")
    parser.add_argument("--exclude-hidden", action="store_true", help="Skip hidden files/directories.")
    parser.add_argument("--absolute", action="store_true", help="Write absolute paths instead of relative paths.")
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_argparser().parse_args()
    root = args.root.expanduser().resolve()
    out = args.out if args.out is not None else root / f"{root.name}_tree_manifest.tsv"
    rows = generate_directory_manifest(
        root,
        out_path=out,
        include_root=args.include_root,
        include_hidden=not args.exclude_hidden,
        relative=not args.absolute,
    )
    print(f"Wrote {len(rows)} rows to: {out}")


if __name__ == "__main__":
    main()
