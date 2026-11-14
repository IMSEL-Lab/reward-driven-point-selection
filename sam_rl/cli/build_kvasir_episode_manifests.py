from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sam_rl.episode_manifest import (
    EpisodeSpec,
    resolve_episode_paths,
    validate_episode_paths,
    write_episode_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "kvasir_manifest_builder_v2.json"
SUPPORTED_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


@dataclass(frozen=True)
class KvasirAsset:
    image_id: str
    image_path: Path
    mask_path: Path
    width: int
    height: int
    mask_area_pixels: int
    mask_area_fraction: float
    quantile_bin: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic Kvasir-SEG train/val/test splits and explicit "
            "support-query episode manifests."
        )
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="JSON config path. Defaults to configs/kvasir_manifest_builder_v2.json.",
    )
    parser.add_argument(
        "--skip-path-validation",
        action="store_true",
        help="Skip existence validation for manifest episode paths before writing outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    outputs = build_kvasir_manifests(
        config,
        repo_root=REPO_ROOT,
        validate_paths=not bool(args.skip_path_validation),
    )
    print(json.dumps(outputs, indent=2, sort_keys=True))


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = (
        "dataset_name",
        "class_name",
        "dataset_root",
        "images_dir",
        "masks_dir",
        "seed",
        "stratify_quantile_bins",
        "split_sizes",
        "support_bank_size",
        "support_assignments_per_query",
        "output_split_path",
        "output_manifest_index_path",
        "output_manifest_paths",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Missing required config keys in {path}: {missing}")
    return payload


def build_kvasir_manifests(
    config: dict[str, Any],
    *,
    repo_root: Path,
    validate_paths: bool = True,
) -> dict[str, Any]:
    dataset_root = resolve_repo_path(config["dataset_root"], repo_root)
    images_dir = resolve_repo_path(config["images_dir"], repo_root)
    masks_dir = resolve_repo_path(config["masks_dir"], repo_root)
    split_sizes = normalize_split_sizes(config["split_sizes"])
    seed = int(config["seed"])
    episode_seed = int(config.get("episode_seed", seed))
    quantile_bins = int(config["stratify_quantile_bins"])
    support_bank_size = int(config["support_bank_size"])
    support_assignments_per_query = int(config["support_assignments_per_query"])
    dataset_name = str(config["dataset_name"])
    class_name = str(config["class_name"])

    raw_assets = discover_assets(images_dir=images_dir, masks_dir=masks_dir)
    assets = assign_quantile_bins(raw_assets, quantile_bins=quantile_bins)
    split_assets = stratified_split(assets, split_sizes=split_sizes, seed=seed)
    train_assets = split_assets["train"]
    support_bank, support_bank_segments = select_support_bank(
        train_assets,
        bank_size=support_bank_size,
        seed=seed,
    )

    manifest_paths = {
        split: resolve_repo_path(path, repo_root)
        for split, path in config["output_manifest_paths"].items()
    }
    validate_split_mapping_keys("output_manifest_paths", manifest_paths)
    split_path = resolve_repo_path(config["output_split_path"], repo_root)
    manifest_index_path = resolve_repo_path(config["output_manifest_index_path"], repo_root)

    support_bank_metadata = {
        "selection_strategy": "seeded_mask_area_segments_over_train_split",
        "seed": seed,
        "size": len(support_bank),
        "selected_image_ids": [asset.image_id for asset in support_bank],
        "segments": support_bank_segments,
    }

    split_json_payload = build_split_payload(
        dataset_name=dataset_name,
        dataset_root=dataset_root,
        seed=seed,
        episode_seed=episode_seed,
        quantile_bins=quantile_bins,
        split_sizes=split_sizes,
        split_assets=split_assets,
        support_bank_metadata=support_bank_metadata,
        repo_root=repo_root,
    )
    write_json(split_path, split_json_payload)

    manifest_index_payload = {
        "dataset": dataset_name,
        "dataset_root": format_output_path(dataset_root, repo_root),
        "class_name": class_name,
        "seed": seed,
        "split_seed": seed,
        "split_sizes": split_sizes,
        "split_path": format_output_path(split_path, repo_root),
        "manifest_paths": {
            split: format_output_path(path, repo_root) for split, path in manifest_paths.items()
        },
        "support_bank": support_bank_metadata,
        "support_assignment_count_per_query": support_assignments_per_query,
        "support_assignment_strategy": "seeded_query_independent_random_from_fixed_train_bank",
        "episode_seed": episode_seed,
        "support_assignments": {},
    }

    for split_name, assets_for_split in split_assets.items():
        ordered_queries = seeded_asset_order(
            assets_for_split,
            seed=episode_seed,
            salt=f"manifest-query-order:{split_name}",
        )
        assignments = {
            query.image_id: [
                support.image_id
                for support in assign_supports(
                    query,
                    support_bank=support_bank,
                    count=support_assignments_per_query,
                    seed=episode_seed,
                )
            ]
            for query in ordered_queries
        }
        manifest_index_payload["support_assignments"][split_name] = assignments
        episodes = build_episodes(
            dataset_name=dataset_name,
            class_name=class_name,
            split_name=split_name,
            ordered_queries=ordered_queries,
            assignments=assignments,
            asset_lookup={asset.image_id: asset for asset in assets},
            repo_root=repo_root,
            dataset_root=dataset_root,
        )
        if validate_paths:
            for episode in episodes:
                validate_episode_paths(resolve_episode_paths(episode, dataset_root))
        manifest_metadata = {
            "dataset": dataset_name,
            "class_name": class_name,
            "split": split_name,
            "seed": seed,
            "split_size": len(ordered_queries),
            "episode_count": len(episodes),
            "split_path": format_output_path(split_path, repo_root),
            "manifest_index_path": format_output_path(manifest_index_path, repo_root),
            "support_bank": support_bank_metadata,
            "support_assignment_count_per_query": support_assignments_per_query,
            "support_assignment_strategy": "seeded_query_independent_random_from_fixed_train_bank",
            "schema_version": 2,
            "data_root": ".",
            "split_seed": seed,
            "episode_seed": episode_seed,
            "support_assignments": assignments,
            "query_image_ids": [asset.image_id for asset in ordered_queries],
            "split_quantile_counts": quantile_counts(ordered_queries),
        }
        write_episode_manifest(manifest_paths[split_name], episodes, metadata=manifest_metadata)

    write_json(manifest_index_path, manifest_index_payload)

    return {
        "dataset_root": format_output_path(dataset_root, repo_root),
        "images": len(assets),
        "split_path": format_output_path(split_path, repo_root),
        "manifest_index_path": format_output_path(manifest_index_path, repo_root),
        "manifest_paths": {
            split: format_output_path(path, repo_root) for split, path in manifest_paths.items()
        },
        "split_sizes": {split: len(items) for split, items in split_assets.items()},
        "support_bank_size": len(support_bank),
        "support_assignment_count_per_query": support_assignments_per_query,
    }


def normalize_split_sizes(payload: dict[str, Any]) -> dict[str, int]:
    required = ("train", "val", "test")
    missing = [key for key in required if key not in payload]
    if missing:
        raise KeyError(f"Split sizes must define {required}; missing {missing}")
    split_sizes = {key: int(payload[key]) for key in required}
    if any(value <= 0 for value in split_sizes.values()):
        raise ValueError(f"Split sizes must be positive, got {split_sizes}")
    return split_sizes


def validate_split_mapping_keys(name: str, payload: dict[str, Any]) -> None:
    expected = {"train", "val", "test"}
    actual = set(payload)
    if actual != expected:
        raise KeyError(f"{name} must define exactly {sorted(expected)}, got {sorted(actual)}")


def discover_assets(*, images_dir: Path, masks_dir: Path) -> list[KvasirAsset]:
    image_map = index_image_stems(images_dir)
    mask_map = index_image_stems(masks_dir)
    image_ids = sorted(image_map)
    missing_masks = [image_id for image_id in image_ids if image_id not in mask_map]
    missing_images = [image_id for image_id in sorted(mask_map) if image_id not in image_map]
    if missing_masks or missing_images:
        raise ValueError(
            "Mismatched Kvasir stems between images and masks: "
            f"missing_masks={missing_masks[:10]} missing_images={missing_images[:10]}"
        )
    assets: list[KvasirAsset] = []
    for image_id in image_ids:
        image_path = image_map[image_id]
        mask_path = mask_map[image_id]
        with Image.open(image_path) as image_handle:
            image_width, image_height = image_handle.size
        with Image.open(mask_path) as mask_handle:
            mask = np.asarray(mask_handle.convert("L"), dtype=np.uint8)
        if mask.ndim != 2:
            raise ValueError(
                f"Expected grayscale-compatible mask for {mask_path}, got shape {mask.shape}"
            )
        mask_height, mask_width = mask.shape
        if (image_width, image_height) != (mask_width, mask_height):
            raise ValueError(
                "Image/mask dimension mismatch for "
                f"{image_id}: image={(image_width, image_height)} mask={(mask_width, mask_height)}"
            )
        mask_area_pixels = int(np.count_nonzero(mask > 0))
        mask_area_fraction = float(mask_area_pixels / float(mask_width * mask_height))
        assets.append(
            KvasirAsset(
                image_id=image_id,
                image_path=image_path,
                mask_path=mask_path,
                width=image_width,
                height=image_height,
                mask_area_pixels=mask_area_pixels,
                mask_area_fraction=mask_area_fraction,
                quantile_bin=-1,
            )
        )
    return assets


def index_image_stems(root: Path) -> dict[str, Path]:
    if not root.exists():
        raise FileNotFoundError(f"Missing expected directory: {root}")
    paths = [
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    by_stem: dict[str, Path] = {}
    duplicates: list[str] = []
    for path in sorted(paths):
        stem = path.stem
        if stem in by_stem:
            duplicates.append(stem)
            continue
        by_stem[stem] = path
    if duplicates:
        raise ValueError(f"Duplicate stems in {root}: {duplicates[:10]}")
    if not by_stem:
        raise ValueError(f"No supported image files found in {root}")
    return by_stem


def assign_quantile_bins(assets: list[KvasirAsset], *, quantile_bins: int) -> list[KvasirAsset]:
    if quantile_bins <= 0:
        raise ValueError(f"quantile_bins must be positive, got {quantile_bins}")
    ordered = sorted(assets, key=lambda asset: (asset.mask_area_fraction, asset.image_id))
    total = len(ordered)
    if total == 0:
        raise ValueError("No assets discovered for Kvasir manifest build.")
    effective_bins = min(int(quantile_bins), total)
    assigned: list[KvasirAsset] = []
    for rank, asset in enumerate(ordered):
        quantile_bin = min(effective_bins - 1, math.floor(rank * effective_bins / total))
        assigned.append(
            KvasirAsset(
                image_id=asset.image_id,
                image_path=asset.image_path,
                mask_path=asset.mask_path,
                width=asset.width,
                height=asset.height,
                mask_area_pixels=asset.mask_area_pixels,
                mask_area_fraction=asset.mask_area_fraction,
                quantile_bin=int(quantile_bin),
            )
        )
    return assigned


def stratified_split(
    assets: list[KvasirAsset],
    *,
    split_sizes: dict[str, int],
    seed: int,
) -> dict[str, list[KvasirAsset]]:
    total_assets = len(assets)
    expected_total = sum(split_sizes.values())
    if total_assets != expected_total:
        raise ValueError(
            f"Split targets sum to {expected_total}, but discovered {total_assets} assets."
        )
    grouped: dict[int, list[KvasirAsset]] = defaultdict(list)
    for asset in assets:
        grouped[int(asset.quantile_bin)].append(asset)
    bin_ids = sorted(grouped)
    ordered_bins = [
        seeded_asset_order(grouped[bin_id], seed=seed, salt=f"split-bin:{bin_id}")
        for bin_id in bin_ids
    ]
    bin_sizes = [len(bin_assets) for bin_assets in ordered_bins]

    train_counts = allocate_counts(
        bin_sizes, target_total=split_sizes["train"], seed=seed, salt="train"
    )
    remaining_after_train = [
        size - train for size, train in zip(bin_sizes, train_counts, strict=True)
    ]
    val_counts = allocate_counts(
        remaining_after_train, target_total=split_sizes["val"], seed=seed, salt="val"
    )
    test_counts = [
        size - train - val
        for size, train, val in zip(bin_sizes, train_counts, val_counts, strict=True)
    ]
    if sum(test_counts) != split_sizes["test"]:
        raise ValueError(f"Expected test size {split_sizes['test']}, got {sum(test_counts)}")

    split_assets: dict[str, list[KvasirAsset]] = {"train": [], "val": [], "test": []}
    for bin_assets, train_count, val_count, test_count in zip(
        ordered_bins,
        train_counts,
        val_counts,
        test_counts,
        strict=True,
    ):
        split_assets["train"].extend(bin_assets[:train_count])
        split_assets["val"].extend(bin_assets[train_count : train_count + val_count])
        split_assets["test"].extend(
            bin_assets[train_count + val_count : train_count + val_count + test_count]
        )
    return split_assets


def allocate_counts(
    group_sizes: list[int], *, target_total: int, seed: int, salt: str
) -> list[int]:
    available_total = sum(group_sizes)
    if target_total < 0 or target_total > available_total:
        raise ValueError(
            f"Cannot allocate target_total={target_total} from available_total={available_total} for {salt}"
        )
    if available_total == 0:
        return [0 for _ in group_sizes]
    raw = [(target_total * size) / available_total for size in group_sizes]
    counts = [min(size, math.floor(value)) for size, value in zip(group_sizes, raw, strict=True)]
    remainder = target_total - sum(counts)
    if remainder < 0:
        raise RuntimeError(f"Over-allocated {salt}: counts={counts} target_total={target_total}")
    if remainder == 0:
        return counts
    candidates = []
    for index, (size, count, value) in enumerate(zip(group_sizes, counts, raw, strict=True)):
        if count >= size:
            continue
        fractional = value - math.floor(value)
        tie_break = stable_key(seed, salt, str(index))
        candidates.append((-fractional, tie_break, index))
    for _, _, index in sorted(candidates)[:remainder]:
        counts[index] += 1
    if sum(counts) != target_total:
        raise RuntimeError(
            f"Failed to allocate {salt}: counts={counts} target_total={target_total}"
        )
    return counts


def select_support_bank(
    train_assets: list[KvasirAsset],
    *,
    bank_size: int,
    seed: int,
) -> tuple[list[KvasirAsset], list[dict[str, Any]]]:
    if bank_size <= 0:
        raise ValueError(f"support_bank_size must be positive, got {bank_size}")
    if bank_size > len(train_assets):
        raise ValueError(
            f"support_bank_size={bank_size} exceeds train split size={len(train_assets)}"
        )
    ordered = sorted(train_assets, key=lambda asset: (asset.mask_area_fraction, asset.image_id))
    bank: list[KvasirAsset] = []
    segments: list[dict[str, Any]] = []
    for segment_index in range(bank_size):
        start = math.floor(segment_index * len(ordered) / bank_size)
        end = math.floor((segment_index + 1) * len(ordered) / bank_size)
        if end <= start:
            end = start + 1
        segment = ordered[start:end]
        chosen = min(
            segment,
            key=lambda asset: (
                stable_key(seed, "support-bank", str(segment_index), asset.image_id),
                asset.image_id,
            ),
        )
        bank.append(chosen)
        segments.append(
            {
                "segment_index": segment_index,
                "rank_start": start,
                "rank_end": end,
                "selected_image_id": chosen.image_id,
                "selected_quantile_bin": chosen.quantile_bin,
                "selected_mask_area_fraction": round(chosen.mask_area_fraction, 8),
            }
        )
    return bank, segments


def assign_supports(
    query: KvasirAsset,
    *,
    support_bank: list[KvasirAsset],
    count: int,
    seed: int,
) -> list[KvasirAsset]:
    candidates = [asset for asset in support_bank if asset.image_id != query.image_id]
    if len(candidates) < count:
        raise ValueError(
            f"Not enough support candidates for query={query.image_id}: "
            f"needed={count} available={len(candidates)}"
        )
    ordered = sorted(
        candidates,
        key=lambda asset: (
            stable_key(seed, "support-assignment", query.image_id, asset.image_id),
            asset.image_id,
        ),
    )
    return ordered[:count]


def build_episodes(
    *,
    dataset_name: str,
    class_name: str,
    split_name: str,
    ordered_queries: list[KvasirAsset],
    assignments: dict[str, list[str]],
    asset_lookup: dict[str, KvasirAsset],
    repo_root: Path,
    dataset_root: Path,
) -> list[EpisodeSpec]:
    episodes: list[EpisodeSpec] = []
    for query in ordered_queries:
        for support_id in assignments[query.image_id]:
            support = asset_lookup[support_id]
            episodes.append(
                EpisodeSpec(
                    dataset=dataset_name,
                    split=split_name,
                    class_name=class_name,
                    support_id=support.image_id,
                    query_id=query.image_id,
                    support_image_path=Path(format_asset_path(support.image_path, dataset_root)),
                    support_mask_path=Path(format_asset_path(support.mask_path, dataset_root)),
                    query_image_path=Path(format_asset_path(query.image_path, dataset_root)),
                    query_mask_path=Path(format_asset_path(query.mask_path, dataset_root)),
                )
            )
    return episodes


def build_split_payload(
    *,
    dataset_name: str,
    dataset_root: Path,
    seed: int,
    episode_seed: int,
    quantile_bins: int,
    split_sizes: dict[str, int],
    split_assets: dict[str, list[KvasirAsset]],
    support_bank_metadata: dict[str, Any],
    repo_root: Path,
) -> dict[str, Any]:
    all_assets = [asset for assets in split_assets.values() for asset in assets]
    ordered_assets = sorted(
        all_assets, key=lambda asset: (asset.mask_area_fraction, asset.image_id)
    )
    area_values = np.asarray(
        [asset.mask_area_fraction for asset in ordered_assets], dtype=np.float64
    )
    quantile_edges = np.quantile(
        area_values, np.linspace(0.0, 1.0, min(quantile_bins, len(ordered_assets)) + 1)
    )
    split_by_id = {
        split_name: [
            asset.image_id
            for asset in seeded_asset_order(assets, seed=seed, salt=f"split-order:{split_name}")
        ]
        for split_name, assets in split_assets.items()
    }
    return {
        "dataset": dataset_name,
        "dataset_root": format_output_path(dataset_root, repo_root),
        "seed": seed,
        "split_seed": seed,
        "episode_seed": episode_seed,
        "split_sizes": split_sizes,
        "stratification": {
            "metric": "mask_area_fraction",
            "quantile_bin_count": min(int(quantile_bins), len(ordered_assets)),
            "quantile_edges": [round(float(value), 8) for value in quantile_edges.tolist()],
            "split_quantile_counts": {
                split_name: quantile_counts(assets) for split_name, assets in split_assets.items()
            },
        },
        "support_bank": support_bank_metadata,
        "items": [asset_record(asset, repo_root) for asset in ordered_assets],
        **split_by_id,
    }


def quantile_counts(assets: list[KvasirAsset]) -> dict[str, int]:
    counts = Counter(int(asset.quantile_bin) for asset in assets)
    return {str(bin_id): counts[bin_id] for bin_id in sorted(counts)}


def asset_record(asset: KvasirAsset, repo_root: Path) -> dict[str, Any]:
    return {
        "image_id": asset.image_id,
        "image_path": format_output_path(asset.image_path, repo_root),
        "mask_path": format_output_path(asset.mask_path, repo_root),
        "width": int(asset.width),
        "height": int(asset.height),
        "mask_area_pixels": int(asset.mask_area_pixels),
        "mask_area_fraction": round(float(asset.mask_area_fraction), 8),
        "quantile_bin": int(asset.quantile_bin),
    }


def seeded_asset_order(assets: list[KvasirAsset], *, seed: int, salt: str) -> list[KvasirAsset]:
    return sorted(
        assets,
        key=lambda asset: (
            stable_key(seed, salt, asset.image_id),
            asset.image_id,
        ),
    )


def stable_key(seed: int, *parts: str) -> str:
    data = "|".join([str(seed), *parts]).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def resolve_repo_path(path_value: str | Path, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def output_path(path: Path, repo_root: Path) -> Path:
    try:
        return path.relative_to(repo_root)
    except ValueError:
        return path


def format_output_path(path: Path, repo_root: Path) -> str:
    return str(output_path(path, repo_root))


def format_asset_path(path: Path, dataset_root: Path) -> str:
    try:
        return str(path.relative_to(dataset_root))
    except ValueError as exc:
        raise ValueError(f"Asset {path} is outside dataset root {dataset_root}") from exc


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
