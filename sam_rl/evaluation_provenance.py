"""Stable provenance helpers for distributed evaluation campaigns."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_SOURCE_PATTERNS = (
    "sam_rl/**/*.py",
    "pyproject.toml",
    "uv.lock",
    "requirements*.txt",
    "environment*.yml",
    "environment*.yaml",
)


def evaluation_config_sha256(config: Mapping[str, Any]) -> str:
    """Hash the resolved evaluation config using canonical JSON serialization."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def evaluation_source_fingerprint(repo_root: str | Path) -> dict[str, Any]:
    """Fingerprint Python source and dependency manifests without absolute paths.

    The source list intentionally covers all project modules so evaluator behavior
    cannot change through an indirect helper import without changing this digest.
    """
    root = Path(repo_root).resolve()
    files: dict[str, str] = {}
    for pattern in _SOURCE_PATTERNS:
        for path in root.glob(pattern):
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not files:
        raise ValueError(f"No evaluation source files found under repository root {root}.")
    _add_segment_anything_sources(files)
    digest = hashlib.sha256()
    for relative, file_hash in sorted(files.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
    return {
        "evaluation_source_sha256": digest.hexdigest(),
        "evaluation_source_files": files,
    }


def _add_segment_anything_sources(files: dict[str, str]) -> None:
    """Hash installed SAM Python files under stable paths, without importing SAM."""
    spec = importlib.util.find_spec("segment_anything")
    locations = (
        sorted(str(Path(item).resolve()) for item in spec.submodule_search_locations or [])
        if spec
        else []
    )
    dependency_files: dict[str, str] = {}
    for index, location in enumerate(locations):
        package_root = Path(location)
        prefix = "external/segment_anything"
        if len(locations) > 1:
            prefix = f"{prefix}/location-{index:03d}"
        for path in package_root.rglob("*.py"):
            if path.is_file():
                relative = path.relative_to(package_root.parent).as_posix()
                dependency_files[f"{prefix}/{relative}"] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    if dependency_files:
        files.update(dependency_files)
    else:
        marker = hashlib.sha256(b"segment_anything source unavailable").hexdigest()
        files["external/segment_anything/MISSING"] = marker
