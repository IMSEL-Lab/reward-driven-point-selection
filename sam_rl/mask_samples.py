"""Deterministic, lightweight final-mask examples for evaluation outputs."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from PIL import Image

from sam_rl.episode_manifest import EpisodeSpec, episode_identity
from sam_rl.provenance import atomic_write_json

SAMPLE_RULE = "first_n_in_ordered_manifest_v1"
SAMPLE_COUNT = 10


def mask_sample_dir_for_csv(csv_path: str | Path) -> Path:
    return Path(csv_path).with_suffix(".mask_samples")


def write_final_mask_samples(
    *,
    output_dir: str | Path,
    episodes: Sequence[EpisodeSpec],
    mask_entries: Sequence[tuple[EpisodeSpec, np.ndarray]],
    sample_count: int,
    result_csv: str | Path,
    metadata_path: str | Path | None = None,
) -> Path | None:
    """Save masks for the first N ordered episodes without any additional decode."""
    requested = int(sample_count)
    if requested < 0:
        raise ValueError("sample_count must be nonnegative.")
    if requested == 0:
        return None
    if not episodes:
        raise ValueError("Cannot sample final masks from an empty episode sequence.")
    selected_count = min(requested, len(episodes))
    selected = list(mask_entries)
    expected_keys = [episode_identity(episode) for episode in episodes]
    selected_keys = [episode_identity(episode) for episode, _mask in selected]
    if len(set(expected_keys)) != len(expected_keys):
        raise ValueError("Episode manifest contains duplicate stable identities.")
    if selected_keys != expected_keys[:selected_count]:
        raise ValueError("Mask examples must match the first N episodes in manifest order.")
    if len(selected) != selected_count:
        raise ValueError(f"Expected {selected_count} sampled masks, received {len(selected)}.")

    csv_file = Path(result_csv)
    sidecar = (
        Path(metadata_path) if metadata_path is not None else csv_file.with_suffix(".metadata.json")
    )
    if not sidecar.is_file():
        raise FileNotFoundError(f"Result metadata sidecar is required for mask samples: {sidecar}.")
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    if metadata.get("expected_episode_keys") != expected_keys:
        raise ValueError("Result metadata episode identities do not match the evaluated manifest.")
    required = (
        "dataset",
        "split",
        "condition",
        "training_seed",
        "eval_seed",
        "decoder",
        "protocol",
        "manifest_hash",
        "model_config_hash",
        "model_hash",
        "checkpoint_sha256",
        "evaluation_config_sha256",
        "evaluation_source_sha256",
    )
    missing = [
        field
        for field in required
        if metadata.get(field) is None
        or (isinstance(metadata.get(field), str) and not metadata[field])
    ]
    if missing:
        raise ValueError(f"Result metadata is missing mask-sample linkage fields {missing}.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    samples = []
    for index, (episode, raw_mask) in enumerate(selected, start=1):
        array = np.asarray(raw_mask)
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.ndim != 2:
            raise ValueError(f"Expected a 2-D final mask, got shape {array.shape}.")
        if not np.isfinite(array).all():
            raise ValueError("Final mask contains nonfinite values.")
        binary = (array > 0).astype(np.uint8)
        filename = f"mask_{index:03d}.png"
        png_path = destination / filename
        Image.fromarray(binary * 255).save(png_path, format="PNG")
        samples.append(
            {
                "sample_index": index,
                "episode_key": episode_identity(episode),
                "image_id": episode.image_id,
                "class_name": episode.class_name,
                "support_id": episode.support_id,
                "query_id": episode.query_id,
                "mask_png": filename,
                "width": int(binary.shape[1]),
                "height": int(binary.shape[0]),
                "foreground_pixels": int(binary.sum()),
                "mask_sha256": _sha256(png_path),
            }
        )

    manifest_path = destination / "sample_manifest.json"
    manifest = {
        "schema_version": 1,
        "sampling_rule": SAMPLE_RULE,
        "requested_sample_count": requested,
        "sample_count": len(samples),
        "expected_episode_count": len(episodes),
        "sampled_episode_keys": [sample["episode_key"] for sample in samples],
        "result_csv": os.path.relpath(csv_file, destination),
        "result_metadata": os.path.relpath(sidecar, destination),
        "result_metadata_sha256": _sha256(sidecar),
        "dataset": str(metadata["dataset"]),
        "split": str(metadata["split"]),
        "condition": str(metadata["condition"]),
        "training_seed": int(metadata["training_seed"]),
        "eval_seed": int(metadata["eval_seed"]),
        "decoder": str(metadata["decoder"]),
        "protocol": str(metadata["protocol"]),
        "budget": int(metadata.get("budget", 0)),
        "manifest_hash": str(metadata["manifest_hash"]),
        "model_config_hash": str(metadata["model_config_hash"]),
        "model_hash": str(metadata["model_hash"]),
        "checkpoint_sha256": str(metadata["checkpoint_sha256"]),
        "evaluation_config_sha256": str(metadata["evaluation_config_sha256"]),
        "evaluation_source_sha256": str(metadata["evaluation_source_sha256"]),
        "samples": samples,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def validate_final_mask_samples(
    manifest_path: str | Path,
    *,
    metadata_path: str | Path,
    expected_episode_keys: list[str] | tuple[str, ...],
    sample_count: int = SAMPLE_COUNT,
) -> dict[str, Any]:
    """Validate deterministic sample identities, sidecar linkage, and binary PNGs."""
    manifest_file = Path(manifest_path)
    payload = json.loads(manifest_file.read_text(encoding="utf-8"))
    expected = list(map(str, expected_episode_keys))
    expected_count = min(int(sample_count), len(expected))
    if payload.get("sampling_rule") != SAMPLE_RULE:
        raise ValueError("Mask sample manifest uses an unsupported sampling rule.")
    if int(payload.get("sample_count", -1)) != expected_count:
        raise ValueError("Mask sample manifest has the wrong sample count.")
    selected_keys = expected[:expected_count]
    if payload.get("sampled_episode_keys") != selected_keys:
        raise ValueError("Mask sample manifest does not contain the first ordered episodes.")
    metadata_file = Path(metadata_path)
    if payload.get("result_metadata_sha256") != _sha256(metadata_file):
        raise ValueError("Mask sample manifest is not linked to the current result metadata.")
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    for field in (
        "dataset",
        "split",
        "condition",
        "training_seed",
        "eval_seed",
        "decoder",
        "protocol",
        "manifest_hash",
        "model_config_hash",
        "model_hash",
        "checkpoint_sha256",
        "evaluation_config_sha256",
        "evaluation_source_sha256",
    ):
        if payload.get(field) != metadata.get(field):
            raise ValueError(f"Mask sample manifest linkage differs for {field}.")
    samples_value = payload.get("samples")
    if not isinstance(samples_value, list) or len(samples_value) != expected_count:
        raise ValueError("Mask sample manifest has an invalid samples array.")
    samples = cast(list[dict[str, Any]], samples_value)
    for index, sample in enumerate(samples, start=1):
        if (
            sample.get("sample_index") != index
            or sample.get("episode_key") != selected_keys[index - 1]
        ):
            raise ValueError("Mask sample entries are out of deterministic manifest order.")
        relative = Path(str(sample.get("mask_png", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Mask sample path must stay inside its sample directory.")
        path = manifest_file.parent / relative
        if not path.is_file() or sample.get("mask_sha256") != _sha256(path):
            raise ValueError(f"Mask PNG is missing or its digest is incorrect: {path}.")
        with Image.open(path) as image:
            pixels = np.asarray(image)
        if pixels.ndim != 2 or not set(np.unique(pixels)).issubset({0, 255}):
            raise ValueError(f"Mask PNG is not binary grayscale: {path}.")
        if pixels.shape != (int(sample["height"]), int(sample["width"])):
            raise ValueError(f"Mask PNG shape disagrees with manifest: {path}.")
    return payload


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
