from __future__ import annotations

import argparse
import json
from pathlib import Path

from sam_rl.episode_manifest import validate_manifest_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate frozen FSS and Kvasir protocol manifests."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--fss-split", default="artifacts/protocol/splits/fss_class_disjoint_canonical66_v2.json"
    )
    parser.add_argument(
        "--canonical-split",
        default="artifacts/protocol/splits/fss_canonical_test_disjoint66_v2.json",
    )
    parser.add_argument("--kvasir-split", default="artifacts/protocol/splits/kvasir_seg_v2.json")
    parser.add_argument(
        "--kvasir-manifest-dir", default="artifacts/protocol/kvasir_manifests/kvasir_seg_v2"
    )
    parser.add_argument(
        "--fss-manifest-dir",
        default="artifacts/protocol/fss_manifests/fss_class_disjoint_canonical66_v2",
    )
    parser.add_argument(
        "--fss-canonical-manifest",
        default="artifacts/protocol/fss_manifests/fss_canonical66_v2/test.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    fss = json.loads((root / args.fss_split).read_text(encoding="utf-8"))
    canonical = json.loads((root / args.canonical_split).read_text(encoding="utf-8"))
    fss_pools = {key: set(fss[key]) for key in ("train", "val", "test")}
    canonical_ids = set(canonical["canonical66_test"])
    if [len(fss[key]) for key in ("train", "val", "test")] != [700, 119, 150]:
        raise ValueError("FSS v2 class counts must be 700/119/150")
    if len(canonical_ids) != 66:
        raise ValueError(f"Expected 66 distinct canonical classes; got {len(canonical_ids)}")
    for left, left_ids in fss_pools.items():
        for right, right_ids in fss_pools.items():
            if left < right and left_ids & right_ids:
                raise ValueError(f"FSS class split overlap between {left} and {right}")
        if left in {"train", "val"} and left_ids & canonical_ids:
            raise ValueError(f"Canonical classes overlap FSS {left}")
    fss_manifest_dir = root / args.fss_manifest_dir
    fss_manifests = {
        split: fss_manifest_dir / f"{split}.json" for split in ("train", "val", "test")
    }
    fss_report = validate_manifest_set(
        fss_manifests,
        expected_counts={"train": 2100, "val": 357, "test": 450},
    )
    fss_canonical_manifest = root / args.fss_canonical_manifest
    canonical_report = validate_manifest_set(
        {"canonical66_test": fss_canonical_manifest}, expected_counts={"canonical66_test": 660}
    )
    for split, manifest_path in fss_manifests.items():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        rows = payload["episodes"]
        classes = {str(row["class_name"]) for row in rows}
        expected_queries = 3
        class_counts: dict[str, dict[str, int]] = {}
        for row in rows:
            counts = class_counts.setdefault(str(row["class_name"]), {})
            query_id = str(row["query_id"])
            counts[query_id] = counts.get(query_id, 0) + 1
        if classes != fss_pools[split] or any(
            len(query_counts) != expected_queries or any(n != 1 for n in query_counts.values())
            for query_counts in class_counts.values()
        ):
            raise ValueError(f"FSS {split} manifest class/query coverage does not match its split")
    canonical_payload = json.loads(fss_canonical_manifest.read_text(encoding="utf-8"))
    canonical_rows = canonical_payload["episodes"]
    if {str(row["class_name"]) for row in canonical_rows} != canonical_ids:
        raise ValueError("Canonical manifest class coverage does not match the held-out list")
    kvasir = json.loads((root / args.kvasir_split).read_text(encoding="utf-8"))
    split_ids = {split: set(kvasir[split]) for split in ("train", "val", "test")}
    if [len(split_ids[s]) for s in ("train", "val", "test")] != [700, 150, 150]:
        raise ValueError("Kvasir v2 image counts must be 700/150/150")
    if any(
        split_ids[a] & split_ids[b]
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    ):
        raise ValueError("Kvasir v2 image splits overlap")
    support_ids = set(kvasir["support_bank"])
    if not support_ids or not support_ids <= split_ids["train"]:
        raise ValueError("Kvasir support bank must be nonempty and entirely train-split")
    manifest_dir = root / args.kvasir_manifest_dir
    manifests = {split: manifest_dir / f"{split}.json" for split in ("train", "val", "test")}
    report = validate_manifest_set(
        manifests,
        expected_counts={"train": 2100, "val": 450, "test": 450},
        expected_query_ids=split_ids,
        support_bank_ids=support_ids,
        supports_per_query=3,
    )
    report["fss"] = {
        "class_counts": {split: len(ids) for split, ids in fss_pools.items()},
        "clean_validation_episodes_at_three_per_class": 3 * len(fss_pools["val"]),
        "canonical66_classes": len(canonical_ids),
        "canonical66_test_class_overlap": len(fss_pools["test"] & canonical_ids),
        "manifest_hashes": {
            **fss_report["manifest_hashes"],
            **canonical_report["manifest_hashes"],
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
