"""Read-only validation of completed greedy-oracle label caches."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from sam_rl.baseline_eval import DINO_REVISION
from sam_rl.episode_manifest import (
    EpisodeSpec,
    episode_identity,
    load_episode_manifest,
    safe_slug,
)
from sam_rl.provenance import cache_metadata, file_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate oracle label files and shard coverage without loading models."
    )
    parser.add_argument("--config", required=True, type=Path, help="BC experiment YAML config.")
    parser.add_argument("--split", required=True, choices=("train", "val"))
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--oracle-dir", type=Path, help="Override the configured split cache directory."
    )
    parser.add_argument(
        "--max-episodes", type=int, help="Validate the first N manifest episodes (pilot cap)."
    )
    parser.add_argument(
        "--shard-dir",
        type=Path,
        help="Shard manifest directory; defaults to oracle-dir/shards when present.",
    )
    parser.add_argument(
        "--shard-manifests", type=Path, nargs="*", help="Explicit shard manifest files."
    )
    return parser.parse_args()


def load_bc_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in BC config {path}")
    return config


def load_expected_episodes(
    config: dict[str, Any],
    split: str,
    *,
    repo_root: str | Path,
    max_episodes: int | None = None,
) -> list[EpisodeSpec]:
    """Read expected identities from frozen manifests, or the config's fixed FSS pair schedule."""
    root = Path(repo_root)
    dataset = config["dataset"]
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive when supplied")
    data_root = resolve_config_path(dataset.get("data_root", dataset.get("root", ".")), root)
    manifests = dataset.get("episode_manifests")
    if manifests:
        path = resolve_config_path(manifests[split], root)
        return load_episode_manifest(
            path,
            max_episodes=max_episodes,
            validate_paths=False,
            data_root=data_root,
        )
    manifest = dataset.get("episode_manifest")
    if manifest:
        return load_episode_manifest(
            resolve_config_path(manifest, root),
            max_episodes=max_episodes,
            validate_paths=False,
            data_root=data_root,
        )
    if not dataset.get("split_file") or not dataset.get("pair_schedule"):
        raise ValueError("Config must provide episode manifests or split_file and pair_schedule")
    split_path = resolve_config_path(dataset["split_file"], root)
    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    if split not in split_payload:
        raise KeyError(f"Split {split!r} is absent from {split_path}")
    episodes: list[EpisodeSpec] = []
    for class_name in split_payload[split]:
        class_dir = data_root / str(class_name)
        for support_id, query_id in dataset["pair_schedule"]:
            sid, qid = str(support_id), str(query_id)
            episodes.append(
                EpisodeSpec(
                    dataset="fss1000",
                    split=split,
                    class_name=str(class_name),
                    support_id=sid,
                    query_id=qid,
                    support_image_path=class_dir / f"{sid}.jpg",
                    support_mask_path=class_dir / f"{sid}.png",
                    query_image_path=class_dir / f"{qid}.jpg",
                    query_mask_path=class_dir / f"{qid}.png",
                )
            )
            if max_episodes is not None and len(episodes) >= max_episodes:
                return episodes
    return episodes


def expected_oracle_provenance(
    episode: EpisodeSpec, config: dict[str, Any], *, repo_root: str | Path | None = None
) -> dict[str, Any]:
    models = config["models"]
    oracle = config["oracle"]
    model = config["model"]
    protocol = config.get("protocol", {})
    grid = int(model.get("grid", 37))
    budget = int(oracle["max_budget"])
    image_size = int(models["image_size"])
    precision = str(oracle.get("precision", protocol.get("precision", "fp32"))).lower()
    decoder_policy = str(
        oracle.get("decoder_policy", protocol.get("decoder_policy", "single"))
    ).lower()
    if grid != 37:
        raise ValueError(f"Oracle config must use the common 37x37 grid, got {grid}")
    if budget != 5 or int(config.get("train", {}).get("budget", budget)) != 5:
        raise ValueError("Oracle and BC train budgets must both be exactly 5")
    if (
        decoder_policy != "single"
        or str(protocol.get("decoder_policy", decoder_policy)) != "single"
    ):
        raise ValueError("Oracle config must use single-mask decoding for every prefix")
    model_precision = str(models.get("precision", "fp32")).lower()
    if precision != "fp32" or model_precision != "fp32" or bool(oracle.get("use_bfloat16", False)):
        raise ValueError("Oracle labels must use the FP32 protocol")

    root = Path(repo_root) if repo_root is not None else Path.cwd()
    sam_checkpoint = resolve_config_path(
        models.get("oracle_sam_checkpoint", models.get("sam_checkpoint")), root
    )
    dino_checkpoint = resolve_config_path(models["dino_checkpoint"], root)
    sam_type = str(models.get("oracle_sam_model_type", models.get("sam_model_type", "")))
    sim_cache_metadata = cache_metadata(
        kind="dino_similarity",
        episode=episode,
        config={
            "encoder_checkpoint": file_fingerprint(dino_checkpoint),
            "encoder": "dinov2_vitl14",
            "image_size": image_size,
            "preprocessing": "rgb_imagenet_normalize_resize_bicubic",
            "precision": "fp32",
            "encoder_revision": DINO_REVISION,
        },
        schema_version=2,
    )
    cache_config = {
        "grid": grid,
        "max_budget": budget,
        "candidate_pool_size": oracle.get("candidate_pool_size"),
        "decoder_policy": decoder_policy,
        "coordinate_policy": "rounded_pixel_centers_v1",
        "mask_resize": "bilinear_align_corners_false",
        "sam_model_type": sam_type,
        "sam_checkpoint": file_fingerprint(sam_checkpoint),
        "similarity_cache": sim_cache_metadata,
        "image_size": image_size,
        "precision": precision,
        "candidate_domain": "all_unused_grid_cells"
        if oracle.get("candidate_pool_size") is None
        else "top_similarity_pool",
    }
    return cache_metadata(kind="greedy_oracle", episode=episode, config=cache_config)


def validate_oracle_payload(
    payload: dict[str, Any],
    *,
    episode: EpisodeSpec,
    expected_provenance: dict[str, Any],
    config: dict[str, Any],
    budget: int = 5,
    grid: int = 37,
) -> None:
    identity = episode_identity(episode)
    if str(payload.get("episode_key", "")) != identity:
        raise ValueError(f"{episode.image_id}: episode_key mismatch")
    if str(payload.get("image_id", "")) != episode.image_id:
        raise ValueError(f"{episode.image_id}: image_id mismatch")
    for field, expected in (
        ("dataset", episode.dataset),
        ("split", episode.split),
        ("class_name", episode.class_name),
        ("grid", grid),
        ("max_budget", budget),
        ("decoder_policy", "single"),
    ):
        if payload.get(field) != expected:
            raise ValueError(f"{episode.image_id}: {field} mismatch; expected {expected!r}")
    if payload.get("provenance") != expected_provenance:
        raise ValueError(
            f"{episode.image_id}: checkpoint, source asset, or cache provenance mismatch"
        )
    config_provenance = payload["provenance"].get("config", {})
    if config_provenance.get("precision") != "fp32":
        raise ValueError(f"{episode.image_id}: provenance is not FP32")
    if config_provenance.get("decoder_policy") != "single":
        raise ValueError(f"{episode.image_id}: provenance is not single-mask")
    expected_cache_config = expected_provenance["config"]
    if payload.get("candidate_pool_size") != expected_cache_config["candidate_pool_size"]:
        raise ValueError(f"{episode.image_id}: candidate pool provenance mismatch")
    models = config["models"]
    expected_checkpoint = models.get("oracle_sam_checkpoint", models.get("sam_checkpoint"))
    expected_model_type = models.get("oracle_sam_model_type", models.get("sam_model_type"))
    if str(payload.get("oracle_sam_checkpoint", "")) != str(expected_checkpoint or ""):
        raise ValueError(f"{episode.image_id}: SAM checkpoint path mismatch")
    if str(payload.get("oracle_sam_model_type", "")) != str(expected_model_type or ""):
        raise ValueError(f"{episode.image_id}: SAM model type mismatch")
    if payload.get("sam_prompt_mode") != "one_shot_positive_points":
        raise ValueError(f"{episode.image_id}: unexpected SAM prompt mode")

    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list) or len(raw_steps) != budget:
        raise ValueError(f"{episode.image_id}: expected exactly {budget} oracle steps")
    steps: list[dict[str, Any]] = []
    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            raise ValueError(f"{episode.image_id}: every oracle step must be an object")
        steps.append(raw_step)
    image_size = payload.get("image_size")
    expected_image_size = int(config["models"]["image_size"])
    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or image_size != [expected_image_size, expected_image_size]
    ):
        raise ValueError(
            f"{episode.image_id}: processed image_size must be "
            f"[{expected_image_size}, {expected_image_size}]"
        )
    height, width = (int(value) for value in image_size)
    used_locations: set[int] = set()
    used_points: set[tuple[int, int]] = set()
    metric_names = ("iou", "dice", "boundary_iou", "sim_value")
    for expected_step, step in enumerate(steps, start=1):
        if int(step.get("step_index", -1)) != expected_step:
            raise ValueError(f"{episode.image_id}: step indices must be ordered 1..{budget}")
        location = step.get("location_idx")
        if (
            isinstance(location, bool)
            or not isinstance(location, int)
            or not 0 <= location < grid**2
        ):
            raise ValueError(f"{episode.image_id}: invalid grid location index {location!r}")
        if location in used_locations:
            raise ValueError(f"{episode.image_id}: duplicate selected grid cell {location}")
        used_locations.add(location)
        grid_x, grid_y = step.get("grid_x"), step.get("grid_y")
        if isinstance(grid_x, bool) or not isinstance(grid_x, int):
            raise ValueError(f"{episode.image_id}: grid coordinates must be integers")
        if isinstance(grid_y, bool) or not isinstance(grid_y, int):
            raise ValueError(f"{episode.image_id}: grid coordinates must be integers")
        if grid_x != location % grid or grid_y != location // grid:
            raise ValueError(f"{episode.image_id}: grid coordinates do not match location_idx")
        point: list[int] = []
        for axis in ("point_x", "point_y"):
            value = step.get(axis)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{episode.image_id}: {axis} must be a finite rounded coordinate")
            if value <= 0 or not float(value).is_integer():
                raise ValueError(f"{episode.image_id}: {axis} must be a positive integer")
            point.append(int(value))
        x, y = point
        if x >= width or y >= height:
            raise ValueError(f"{episode.image_id}: rounded point is outside the query image")
        expected_x = round(((int(grid_x) + 0.5) / grid) * width)
        expected_y = round(((int(grid_y) + 0.5) / grid) * height)
        expected_x = min(max(expected_x, 0), width - 1)
        expected_y = min(max(expected_y, 0), height - 1)
        if (x, y) != (expected_x, expected_y):
            raise ValueError(
                f"{episode.image_id}: rounded point {(x, y)} does not match "
                f"grid-cell center {(expected_x, expected_y)}"
            )
        rounded_point = (x, y)
        if rounded_point in used_points:
            raise ValueError(f"{episode.image_id}: duplicate rounded prompt point {rounded_point}")
        used_points.add(rounded_point)
        for metric_name in metric_names:
            value = step.get(metric_name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{episode.image_id}: {metric_name} must be numeric")
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{episode.image_id}: {metric_name} must be finite and in [0, 1]")
        wallclock = step.get("wallclock_ms")
        if not isinstance(wallclock, (int, float)) or not math.isfinite(wallclock) or wallclock < 0:
            raise ValueError(f"{episode.image_id}: wallclock_ms must be finite and nonnegative")


def validate_oracle_split(
    episodes: list[EpisodeSpec],
    *,
    oracle_dir: str | Path,
    config: dict[str, Any],
    repo_root: str | Path | None = None,
    max_episodes: int | None = None,
    shard_manifests: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    oracle_path = Path(oracle_dir)
    if max_episodes is not None:
        if max_episodes <= 0:
            raise ValueError("max_episodes must be positive when supplied")
        episodes = episodes[:max_episodes]
    expected_ids = [episode_identity(episode) for episode in episodes]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("Episode manifest contains duplicate stable identities")
    expected_files: set[Path] = set()
    for episode in episodes:
        path = oracle_path / f"{episode.split}_{safe_slug(episode.image_id)}.json"
        expected_files.add(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_oracle_payload(
            payload,
            episode=episode,
            expected_provenance=expected_oracle_provenance(episode, config, repo_root=repo_root),
            config=config,
            budget=5,
            grid=37,
        )
    extras: list[Path] = []
    if max_episodes is None:
        actual_files = set(oracle_path.glob("*.json"))
        missing = sorted(expected_files - actual_files)
        extras = sorted(actual_files - expected_files)
        if missing or extras:
            raise ValueError(
                f"Oracle label file set mismatch; missing={missing[:5]}, extra={extras[:5]}"
            )
    shard_report = None
    if shard_manifests:
        shard_report = validate_shard_manifests(shard_manifests, episodes)
    return {
        "split": episodes[0].split if episodes else "",
        "episode_count": len(episodes),
        "validated_label_files": len(expected_files),
        "unexpected_files": [str(path) for path in extras],
        "shards": shard_report,
    }


def validate_shard_manifests(
    paths: Sequence[str | Path], episodes: list[EpisodeSpec]
) -> dict[str, Any]:
    expected_order = [episode.image_id for episode in episodes]
    stable_order = [episode_identity(episode) for episode in episodes]
    expected_split = episodes[0].split if episodes else ""
    if not paths:
        raise ValueError("No shard manifests were supplied")
    shards: list[dict[str, Any]] = [
        json.loads(Path(path).read_text(encoding="utf-8")) for path in paths
    ]
    expected_count = len(shards)
    seen_indices: set[int] = set()
    seen_ids: list[str] = []
    for shard in shards:
        index = int(shard.get("shard_index", -1))
        if not 0 <= index < expected_count:
            raise ValueError(f"Invalid shard index {index} for {expected_count} shard manifests")
        if index in seen_indices:
            raise ValueError(f"Duplicate shard index {index}")
        seen_indices.add(index)
        if str(shard.get("split", "")) != expected_split:
            raise ValueError(f"Shard {index}: split name mismatch")
        if int(shard.get("num_shards", -1)) != expected_count:
            raise ValueError("Shard manifests disagree on num_shards or are incomplete")
        stored_order = shard.get("ordered_episode_ids", [])
        identity_order = stable_order if stored_order == stable_order else expected_order
        if stored_order != identity_order:
            raise ValueError(f"Shard {index}: ordered episode identities mismatch")
        assigned = shard.get("expected_episode_ids", [])
        complete = shard.get("completed_episode_ids", [])
        expected_assigned = [
            episode_id
            for row_index, episode_id in enumerate(identity_order)
            if row_index % expected_count == index
        ]
        if assigned != expected_assigned or complete != assigned:
            raise ValueError(f"Shard {index}: missing, extra, or incomplete episode identities")
        if shard.get("status") != "complete":
            raise ValueError(f"Shard {index}: status is not complete")
        seen_ids.extend(assigned)
    if len(seen_ids) != len(set(seen_ids)):
        raise ValueError("Shard union contains duplicate episode identities")
    if set(seen_ids) not in (set(expected_order), set(stable_order)):
        raise ValueError("Shard union is missing or contains extra episode identities")
    return {"shard_count": expected_count, "episode_count": len(seen_ids)}


def resolve_config_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    config = load_bc_config(resolve_config_path(args.config, root))
    episodes = load_expected_episodes(
        config,
        args.split,
        repo_root=root,
        max_episodes=args.max_episodes,
    )
    if not episodes:
        raise ValueError("No expected episodes were selected")
    configured_oracle_dir = (
        resolve_config_path(args.oracle_dir, root)
        if args.oracle_dir
        else resolve_config_path(config["oracle"]["cache_root"], root) / args.split
    )
    shard_paths = [resolve_config_path(path, root) for path in (args.shard_manifests or [])]
    if args.shard_dir:
        shard_dir = resolve_config_path(args.shard_dir, root)
        shard_paths.extend(sorted(shard_dir.glob("shard-*.json")))
    elif not shard_paths:
        default_shard_dir = configured_oracle_dir / "shards"
        if default_shard_dir.is_dir():
            shard_paths = sorted(default_shard_dir.glob("shard-*.json"))
    print(
        json.dumps(
            validate_oracle_split(
                episodes,
                oracle_dir=configured_oracle_dir,
                config=config,
                repo_root=root,
                max_episodes=args.max_episodes,
                shard_manifests=shard_paths or None,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
