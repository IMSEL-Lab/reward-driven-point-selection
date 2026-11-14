from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from sam_rl.cli.build_kvasir_episode_manifests import KvasirAsset, assign_supports
from sam_rl.episode_manifest import (
    episode_identity,
    load_episode_manifest,
    validate_manifest_set,
)

REPO = Path(__file__).resolve().parents[1]
PROTOCOL = REPO / "artifacts/protocol"


def asset(image_id: str, area: float, quantile_bin: int) -> KvasirAsset:
    return KvasirAsset(
        image_id=image_id,
        image_path=Path("images") / f"{image_id}.jpg",
        mask_path=Path("masks") / f"{image_id}.jpg",
        width=8,
        height=8,
        mask_area_pixels=int(area * 64),
        mask_area_fraction=area,
        quantile_bin=quantile_bin,
    )


class ManifestProtocolTests(unittest.TestCase):
    @unittest.skipUnless(
        (PROTOCOL / "splits/fss_class_disjoint_canonical66_v2.json").is_file(),
        "Install the protocol bundle to run dataset protocol checks.",
    )
    def test_fss_clean_split_counts_and_heldout_overlap(self) -> None:
        split = json.loads((PROTOCOL / "splits/fss_class_disjoint_canonical66_v2.json").read_text())
        canonical = json.loads(
            (PROTOCOL / "splits/fss_canonical_test_disjoint66_v2.json").read_text()
        )
        canonical_classes = set(canonical["canonical66_test"])
        self.assertEqual([len(split[name]) for name in ("train", "val", "test")], [700, 119, 150])
        self.assertEqual(len(canonical_classes), 66)
        self.assertFalse(canonical_classes.intersection(split["train"]))
        self.assertFalse(canonical_classes.intersection(split["val"]))
        self.assertEqual(len(canonical_classes.intersection(split["test"])), 35)
        self.assertEqual(3 * len(split["val"]), 357)

    @unittest.skipUnless(
        (PROTOCOL / "kvasir_manifests/kvasir_seg_v2/test.json").is_file(),
        "Install the protocol bundle to run dataset protocol checks.",
    )
    def test_kvasir_v2_preserves_frozen_pairs_and_is_relocatable(self) -> None:
        expected_pair_hashes = {
            "train": "27248efbacf212a411b019156e269f972a65ad55fb2ea78636e7f75e1f245742",
            "val": "272d657dc31667f8d8a6708121aa9340c805a54e6b78c6828d00d7d1f403d192",
            "test": "9764068fc50e59ee16f4a35bd7a2ddba3e808f56250daf632f2e2a93ec339577",
        }
        for split in ("train", "val", "test"):
            new = json.loads(
                (PROTOCOL / f"kvasir_manifests/kvasir_seg_v2/{split}.json").read_text()
            )
            pairs = [(e["support_id"], e["query_id"]) for e in new["episodes"]]
            pair_hash = hashlib.sha256(
                json.dumps(pairs, separators=(",", ":")).encode()
            ).hexdigest()
            self.assertEqual(pair_hash, expected_pair_hashes[split])
            self.assertEqual(len(new["episodes"]), {"train": 2100, "val": 450, "test": 450}[split])
            self.assertTrue(
                all(
                    not Path(e[key]).is_absolute()
                    for e in new["episodes"]
                    for key in (
                        "support_image_path",
                        "support_mask_path",
                        "query_image_path",
                        "query_mask_path",
                    )
                )
            )
        with self.assertRaises(FileNotFoundError):
            load_episode_manifest(PROTOCOL / "kvasir_manifests/kvasir_seg_v2/test.json")

    def test_explicit_data_root_relocates_relative_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            for folder in ("images", "masks"):
                (tmp_path / folder).mkdir()
            for filename in ("support.jpg", "query.jpg"):
                (tmp_path / "images" / filename).touch()
                (tmp_path / "masks" / filename).touch()
            manifest = tmp_path / "episodes.json"
            manifest.write_text(
                json.dumps(
                    {
                        "split": "test",
                        "episodes": [
                            {
                                "dataset": "kvasir_seg",
                                "class_name": "polyp",
                                "support_id": "support",
                                "query_id": "query",
                                "support_image_path": "images/support.jpg",
                                "support_mask_path": "masks/support.jpg",
                                "query_image_path": "images/query.jpg",
                                "query_mask_path": "masks/query.jpg",
                            }
                        ],
                    }
                )
            )
            loaded = load_episode_manifest(manifest, data_root=tmp_path)
            self.assertEqual(loaded[0].support_image_path, tmp_path / "images/support.jpg")
            self.assertEqual(loaded[0].query_mask_path, tmp_path / "masks/query.jpg")

    def test_query_label_values_do_not_change_support_assignment(self) -> None:
        bank = [asset(f"support-{i}", i / 20, i % 4) for i in range(10)]
        query = asset("query", 0.05, 0)
        changed = replace(query, mask_area_fraction=0.99, mask_area_pixels=63, quantile_bin=9)
        before = [
            item.image_id for item in assign_supports(query, support_bank=bank, count=3, seed=17)
        ]
        after = [
            item.image_id for item in assign_supports(changed, support_bank=bank, count=3, seed=17)
        ]
        self.assertEqual(before, after)

    def test_manifest_identity_is_stable_across_path_roots(self) -> None:
        from sam_rl.episode_manifest import EpisodeSpec

        first = EpisodeSpec(
            "kvasir_seg", "test", "polyp", "s1", "q1", *(Path("/a") / str(i) for i in range(4))
        )
        second = replace(first, support_image_path=Path("/b/s1"), query_mask_path=Path("/b/q1"))
        self.assertEqual(episode_identity(first), episode_identity(second))

    def test_manifest_validation_rejects_duplicate_and_mixed_split_queries(self) -> None:
        base = {
            "dataset": "kvasir_seg",
            "class_name": "polyp",
            "support_id": "s1",
            "query_id": "q1",
            "support_image_path": "images/s1.jpg",
            "support_mask_path": "masks/s1.jpg",
            "query_image_path": "images/q1.jpg",
            "query_mask_path": "masks/q1.jpg",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            tmp_path = Path(temp_dir)
            first = tmp_path / "train.json"
            first.write_text(json.dumps({"episodes": [base, base]}))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                validate_manifest_set({"train": first})
            first.write_text(json.dumps({"episodes": [base]}))
            second = tmp_path / "test.json"
            second.write_text(json.dumps({"episodes": [base | {"support_id": "s2"}]}))
            with self.assertRaisesRegex(ValueError, "appears in both"):
                validate_manifest_set({"train": first, "test": second})


if __name__ == "__main__":
    unittest.main()
