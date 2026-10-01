"""Dataset loading and inspection utilities (Phase 2)."""

from app.dataloaders.loaders import (
    LoadedDataset,
    build_manifest,
    inspect,
    label_to_index,
    load,
    load_all,
    manifest_path,
    to_xy,
    write_manifest,
)
from app.dataloaders.registry import REGISTRY, DatasetSpec, by_role, get, spec_as_dict

__all__ = [
    "REGISTRY",
    "DatasetSpec",
    "LoadedDataset",
    "build_manifest",
    "by_role",
    "get",
    "inspect",
    "label_to_index",
    "load",
    "load_all",
    "manifest_path",
    "spec_as_dict",
    "to_xy",
    "write_manifest",
]
