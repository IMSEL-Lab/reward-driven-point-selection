from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from sam_rl.episode_manifest import EpisodeSpec, write_episode_manifest


@dataclass(frozen=True)
class KvasirImage:
    image_id: str
    image_path: Path
    mask_path: Path
    mask_area: float
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic Kvasir-SEG image splits and support-query episode manifests."
    )
    parser.add_argument("--root", default="data/Kvasir-SEG")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-count", type=int, default=700)
    parser.add_argument("--val-count", type=int, default=150)
    parser.add_argument("--test-count", type=int, default=150)
    parser.add_argument("--strata", type=int, default=10)
    parser.add_argument("--support-bank-size", type=int, default=10)
    parser.add_argument("--supports-per-query", type=int, default=3)
    parser.add_argument("--episode-seed", type=int)
    parser.add_argument(
        "--split-json", default="artifacts/generated_protocol/splits/kvasir_seg_v2.json"
    )
    parser.add_argument(
        "--manifest-dir", default="artifacts/generated_protocol/kvasir_manifests/kvasir_seg_v2"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    manifest_dir = Path(args.manifest_dir)
    split_json = Path(args.split_json)

    images = load_kvasir_images(root)
    episode_seed = int(args.seed if args.episode_seed is None else args.episode_seed)
    split_ids = build_stratified_split_ids(
        images,
        seed=int(args.seed),
        train_count=int(args.train_count),
        val_count=int(args.val_count),
        test_count=int(args.test_count),
        strata=int(args.strata),
    )
    image_by_id = {image.image_id: image for image in images}
    support_bank = choose_support_bank(
        [image_by_id[image_id] for image_id in split_ids["train"]],
        bank_size=int(args.support_bank_size),
        strata=int(args.support_bank_size),
    )

    split_payload = {
        "dataset": "kvasir_seg",
        "version": "kvasir_seg_v2",
        "schema_version": 2,
        "split_seed": int(args.seed),
        "episode_seed": episode_seed,
        "data_root": ".",
        "train": split_ids["train"],
        "val": split_ids["val"],
        "test": split_ids["test"],
        "support_bank": [image.image_id for image in support_bank],
        "supports_per_query": int(args.supports_per_query),
        "train_count": len(split_ids["train"]),
        "val_count": len(split_ids["val"]),
        "test_count": len(split_ids["test"]),
    }
    split_json.parent.mkdir(parents=True, exist_ok=True)
    split_json.write_text(
        json.dumps(split_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    support_bank_ids = [image.image_id for image in support_bank]
    manifest_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "val", "test"):
        split_images = [image_by_id[image_id] for image_id in split_ids[split_name]]
        episodes = build_split_episodes(
            split_name=split_name,
            split_images=split_images,
            support_bank_ids=support_bank_ids,
            image_by_id=image_by_id,
            seed=episode_seed,
            supports_per_query=int(args.supports_per_query),
        )
        write_episode_manifest(
            manifest_dir / f"{split_name}.json",
            episodes,
            metadata={
                "dataset": "kvasir_seg",
                "version": "kvasir_seg_v2",
                "schema_version": 2,
                "data_root": ".",
                "split": split_name,
                "image_count": len(split_images),
                "episode_count": len(episodes),
                "supports_per_query": int(args.supports_per_query),
                "support_bank": support_bank_ids,
                "split_seed": int(args.seed),
                "episode_seed": episode_seed,
            },
        )


def load_kvasir_images(root: Path) -> list[KvasirImage]:
    image_dir = root / "images"
    mask_dir = root / "masks"
    image_paths = sorted(image_dir.glob("*.jpg"))
    mask_paths = {path.stem: path for path in mask_dir.glob("*.jpg")}
    images: list[KvasirImage] = []
    for image_path in image_paths:
        stem = image_path.stem
        if stem not in mask_paths:
            raise FileNotFoundError(f"Missing mask for image: {image_path}")
        mask_path = mask_paths[stem]
        with Image.open(mask_path) as mask_image:
            mask = np.asarray(mask_image.convert("L"), dtype=np.uint8) >= 128
            width, height = mask_image.size
        images.append(
            KvasirImage(
                image_id=stem,
                image_path=Path("images") / image_path.name,
                mask_path=Path("masks") / mask_path.name,
                mask_area=float(mask.mean()),
                width=int(width),
                height=int(height),
            )
        )
    if len(images) != 1000:
        raise ValueError(f"Expected 1000 paired Kvasir images, found {len(images)}.")
    return images


def build_stratified_split_ids(
    images: list[KvasirImage],
    *,
    seed: int,
    train_count: int,
    val_count: int,
    test_count: int,
    strata: int,
) -> dict[str, list[str]]:
    total = train_count + val_count + test_count
    if total != len(images):
        raise ValueError(f"Requested split sizes sum to {total}, but found {len(images)} images.")
    ordered = sorted(images, key=lambda image: (image.mask_area, image.image_id))
    bins: list[list[KvasirImage]] = [[] for _ in range(int(strata))]
    for index, image in enumerate(ordered):
        bins[min(int(strata) - 1, index * int(strata) // len(ordered))].append(image)
    rng = random.Random(seed)
    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    train_target = train_count / len(images)
    val_target = val_count / len(images)
    for bucket in bins:
        rng.shuffle(bucket)
        bucket_size = len(bucket)
        train_n = round(bucket_size * train_target)
        val_n = round(bucket_size * val_target)
        train_slice = bucket[:train_n]
        val_slice = bucket[train_n : train_n + val_n]
        test_slice = bucket[train_n + val_n :]
        train_ids.extend(image.image_id for image in train_slice)
        val_ids.extend(image.image_id for image in val_slice)
        test_ids.extend(image.image_id for image in test_slice)
    rebalance_ids(train_ids, val_ids, test_ids, train_count, val_count, test_count)
    return {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }


def rebalance_ids(
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
    train_count: int,
    val_count: int,
    test_count: int,
) -> None:
    while len(train_ids) > train_count:
        test_ids.append(train_ids.pop())
    while len(val_ids) > val_count:
        test_ids.append(val_ids.pop())
    while len(test_ids) > test_count:
        if len(train_ids) < train_count:
            train_ids.append(test_ids.pop())
        elif len(val_ids) < val_count:
            val_ids.append(test_ids.pop())
        else:
            raise ValueError("Unable to rebalance Kvasir split sizes.")
    while len(train_ids) < train_count:
        train_ids.append(test_ids.pop())
    while len(val_ids) < val_count:
        val_ids.append(test_ids.pop())
    if len(test_ids) != test_count:
        raise ValueError(f"Expected {test_count} test IDs after rebalance, got {len(test_ids)}.")


def choose_support_bank(
    train_images: list[KvasirImage], *, bank_size: int, strata: int
) -> list[KvasirImage]:
    ordered = sorted(train_images, key=lambda image: (image.mask_area, image.image_id))
    bank: list[KvasirImage] = []
    for slot in range(int(bank_size)):
        low = slot * len(ordered) // int(strata)
        high = min(len(ordered), (slot + 1) * len(ordered) // int(strata))
        candidates = ordered[low:high] if high > low else ordered[low : low + 1]
        if not candidates:
            candidates = [ordered[min(low, len(ordered) - 1)]]
        bank.append(candidates[len(candidates) // 2])
    deduped: list[KvasirImage] = []
    seen: set[str] = set()
    for image in bank:
        if image.image_id not in seen:
            seen.add(image.image_id)
            deduped.append(image)
    if len(deduped) < bank_size:
        for image in ordered:
            if image.image_id in seen:
                continue
            deduped.append(image)
            seen.add(image.image_id)
            if len(deduped) >= bank_size:
                break
    return deduped[:bank_size]


def build_split_episodes(
    *,
    split_name: str,
    split_images: list[KvasirImage],
    support_bank_ids: list[str],
    image_by_id: dict[str, KvasirImage],
    seed: int,
    supports_per_query: int,
) -> list[EpisodeSpec]:
    episodes: list[EpisodeSpec] = []
    for query_image in split_images:
        support_ids = choose_support_ids_for_query(
            query_id=query_image.image_id,
            support_bank_ids=support_bank_ids,
            supports_per_query=supports_per_query,
            seed=seed,
        )
        for _support_rank, support_id in enumerate(support_ids, start=1):
            support_image = image_by_id[support_id]
            episodes.append(
                EpisodeSpec(
                    dataset="kvasir_seg",
                    split=split_name,
                    class_name="polyp",
                    support_id=support_image.image_id,
                    query_id=query_image.image_id,
                    support_image_path=support_image.image_path,
                    support_mask_path=support_image.mask_path,
                    query_image_path=query_image.image_path,
                    query_mask_path=query_image.mask_path,
                )
            )
    return episodes


def choose_support_ids_for_query(
    *,
    query_id: str,
    support_bank_ids: list[str],
    supports_per_query: int,
    seed: int,
) -> list[str]:
    candidates = [support_id for support_id in support_bank_ids if support_id != query_id]
    if len(candidates) < supports_per_query:
        raise ValueError(f"Not enough support candidates for query {query_id}.")
    rng = random.Random(f"{seed}:{query_id}")
    chosen = rng.sample(candidates, k=supports_per_query)
    return sorted(chosen)


if __name__ == "__main__":
    main()
