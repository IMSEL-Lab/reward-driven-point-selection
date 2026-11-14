"""Content-based provenance and atomic cache publication helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_HASH_MEMO: dict[tuple[str, int, int, int], str] = {}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_fingerprint(path: str | Path, *, use_memo: bool = True) -> dict[str, Any]:
    """Hash file content; the memo avoids rehashing unchanged large checkpoints."""
    target = Path(path)
    stat = target.stat()
    key = (str(target.resolve()), int(stat.st_size), int(stat.st_mtime_ns), int(stat.st_ctime_ns))
    digest = _HASH_MEMO.get(key) if use_memo else None
    if digest is None:
        hasher = hashlib.sha256()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                hasher.update(chunk)
        digest = hasher.hexdigest()
        _HASH_MEMO[key] = digest
    return {"sha256": digest, "size": int(stat.st_size)}


def path_fingerprint(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    return file_fingerprint(target) if target.is_file() else {"missing": True}


def episode_provenance(episode: Any, *, include_support: bool = True) -> dict[str, Any]:
    """Capture logical episode identity and source assets without absolute-root dependence."""
    result: dict[str, Any] = {
        "dataset": str(episode.dataset),
        "split": str(episode.split),
        "class_name": str(episode.class_name),
        "support_id": str(episode.support_id),
        "query_id": str(episode.query_id),
        "image_id": str(episode.image_id),
        "query_image": file_fingerprint(episode.query_image_path),
        "query_mask": file_fingerprint(episode.query_mask_path),
    }
    if include_support:
        result["support_image"] = file_fingerprint(episode.support_image_path)
        result["support_mask"] = file_fingerprint(episode.support_mask_path)
    return result


def cache_metadata(
    *, kind: str, episode: Any, config: Mapping[str, Any], schema_version: int = 1
) -> dict[str, Any]:
    payload = {
        "kind": str(kind),
        "schema_version": int(schema_version),
        "episode": episode_provenance(episode),
        "config": dict(config),
    }
    return {**payload, "cache_key": stable_hash(payload)}


def cache_path(cache_dir: str | Path, kind: str, episode: Any, metadata: Mapping[str, Any]) -> Path:
    stem = f"{safe_component(episode.split)}_{safe_component(episode.image_id)}"
    return Path(cache_dir) / str(kind) / str(metadata["cache_key"][:24]) / f"{stem}.npz"


def safe_component(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "-_." else "_" for char in str(value))
    return cleaned.strip("_.") or "item"


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, default=_json_default)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def atomic_write_npz(path: str | Path, **arrays: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    try:
        with open(temp_name, "wb") as handle:
            import numpy as np

            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot encode {type(value).__name__} as JSON.")
