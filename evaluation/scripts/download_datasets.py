"""Download, inspect and manifest every registered dataset (Phase 2).

Usage (from the project root or anywhere):

    python evaluation/scripts/download_datasets.py
    python evaluation/scripts/download_datasets.py --only D1 D3
    python evaluation/scripts/download_datasets.py --force

Writes: data/raw/... and data/processed/dataset_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.dataloaders import (  # noqa: E402
    REGISTRY,
    build_manifest,
    inspect,
    load,
    write_manifest,
)


def _print_summary(summary: dict) -> None:
    print(f"\n[{summary['key']}] {summary['identifier']}"
          + (f" ({summary['config']})" if summary.get("config") else ""))
    print(f"    task      : {summary['task']}")
    print(f"    role      : {summary['role']}")
    print(f"    license   : {summary['license']}")
    print(f"    total rows: {summary['total_rows']}")
    if summary.get("extra"):
        print(f"    extra     : {json.dumps(summary['extra'], ensure_ascii=False)}")
    for split, entry in summary["splits"].items():
        flag = ""
        if "matches_expected" in entry:
            flag = "  [OK]" if entry["matches_expected"] else (
                f"  [MISMATCH expected={entry['expected_rows']}]"
            )
        print(f"    - {split:<12} rows={entry['rows']:<7} "
              f"text={entry['text_field']} label={entry['label_field']}{flag}")
        if "label_distribution" in entry:
            dist = entry["label_distribution"]
            preview = ", ".join(f"{k}:{v}" for k, v in list(dist.items())[:8])
            more = "" if len(dist) <= 8 else f", ... (+{len(dist) - 8} classes)"
            print(f"        classes={entry['num_classes']}  {preview}{more}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and inspect datasets.")
    parser.add_argument("--only", nargs="*", metavar="KEY",
                        help="Subset of registry keys, e.g. D1 D3")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if cached")
    parser.add_argument("--no-manifest", action="store_true",
                        help="Skip writing dataset_manifest.json")
    args = parser.parse_args()

    keys = args.only or list(REGISTRY)
    unknown = [k for k in keys if k not in REGISTRY]
    if unknown:
        parser.error(f"Unknown keys {unknown}; known: {sorted(REGISTRY)}")

    loaded = {}
    failures: list[tuple[str, str]] = []
    for key in keys:
        try:
            loaded[key] = load(key, force=args.force)
        except Exception as exc:  # noqa: BLE001 - report and continue
            failures.append((key, f"{type(exc).__name__}: {exc}"))
            print(f"[{key}] FAILED -> {type(exc).__name__}: {exc}")

    for key, ds in loaded.items():
        _print_summary(inspect(ds))

    if not args.no_manifest:
        manifest = build_manifest(loaded)
        path = write_manifest(manifest)
        print(f"\nManifest: {path}")

    print(f"\nDone: {len(loaded)} ok, {len(failures)} failed.")
    for key, msg in failures:
        print(f"  ! {key}: {msg}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
