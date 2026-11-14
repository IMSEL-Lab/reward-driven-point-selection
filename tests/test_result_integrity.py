from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from sam_rl.episode_manifest import EpisodeSpec
from sam_rl.result_manifest import (
    manifest_hash,
    state_dict_hash,
    validate_result_shards,
    write_result_metadata,
)


def write_shard(root, name, rows, keys, *, condition="B2", split="test"):
    path = root / f"{name}.csv"
    fields = [
        "method",
        "N",
        "episode_key",
        "image_id",
        "seed",
        "iou",
        "dice",
        "boundary_iou",
        "wallclock_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".metadata.json").write_text(
        json.dumps(
            {
                "manifest_hash": manifest_hash(keys),
                "model_config_hash": "model-config-1",
                "dataset": "fss",
                "split": split,
                "condition": condition,
                "eval_seed": 7,
                "decoder": "vit_h",
                "model_hash": "model-1",
            }
        ),
        encoding="utf-8",
    )
    return path


def row(key):
    return {
        "method": "greedy",
        "N": 1,
        "episode_key": key,
        "image_id": key,
        "seed": 7,
        "iou": 0.5,
        "dice": 0.6,
        "boundary_iou": 0.4,
        "wallclock_ms": 2.0,
    }


class ResultIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.keys = ["episode-a", "episode-b", "episode-c", "episode-d"]

    def tearDown(self):
        self.tempdir.cleanup()

    def validate(self, paths, keys=None):
        return validate_result_shards(
            paths,
            expected_keys=keys or self.keys,
            expected_methods=["greedy"],
            budgets=[1],
            seeds=[7],
        )

    def test_complete_shard_partition(self):
        paths = [
            write_shard(self.root, "part-0", [row(k) for k in self.keys[::2]], self.keys),
            write_shard(self.root, "part-1", [row(k) for k in self.keys[1::2]], self.keys),
        ]
        self.assertEqual(len(self.validate(paths)), 4)

    def test_duplicate_identity_rejected(self):
        rows = [row("episode-a"), row("episode-a"), row("episode-c"), row("episode-d")]
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            self.validate([write_shard(self.root, "bad", rows, self.keys)])

    def test_missing_identity_rejected(self):
        rows = [row("episode-a"), row("episode-b"), row("episode-c")]
        with self.assertRaisesRegex(ValueError, "Missing"):
            self.validate([write_shard(self.root, "bad", rows, self.keys)])

    def test_nonfinite_metric_rejected(self):
        rows = [row("episode-a"), row("episode-b"), row("episode-c"), row("episode-d")]
        rows[0]["iou"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            self.validate([write_shard(self.root, "bad", rows, self.keys)])

    def test_extra_identity_rejected(self):
        rows = [row("episode-a"), row("episode-b"), row("episode-c"), row("unexpected")]
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            self.validate([write_shard(self.root, "bad", rows, self.keys)])

    def test_incompatible_metadata_rejected(self):
        keys = ["episode-a", "episode-b"]
        paths = [
            write_shard(self.root, "a", [row(keys[0])], keys),
            write_shard(self.root, "b", [row(keys[1])], keys, split="val"),
        ]
        with self.assertRaisesRegex(ValueError, "dataset, split"):
            self.validate(paths, keys)

    def test_training_and_evaluation_seeds_are_distinct_identity_fields(self):
        keys = ["episode-a"]
        path = self.root / "seeded.csv"
        fields = [
            "method",
            "N",
            "episode_key",
            "image_id",
            "seed",
            "training_seed",
            "eval_seed",
            "iou",
            "dice",
            "boundary_iou",
            "wallclock_ms",
        ]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for training_seed in (17, 29):
                writer.writerow(
                    {**row("episode-a"), "training_seed": training_seed, "eval_seed": 3}
                )
        path.with_suffix(".metadata.json").write_text(
            json.dumps(
                {
                    "manifest_hash": manifest_hash(keys),
                    "model_config_hash": "cfg",
                    "dataset": "fss",
                    "split": "test",
                    "condition": "R5",
                }
            ),
            encoding="utf-8",
        )
        actual = validate_result_shards(
            [path],
            expected_keys=keys,
            expected_methods=["greedy"],
            budgets=[1],
            seeds=[],
            training_seeds=[17, 29],
            eval_seeds=[3],
        )
        self.assertEqual(len(actual), 2)

    def test_shared_metadata_writer_round_trips_with_real_model_state_hash(self):
        episode = EpisodeSpec(
            dataset="fss",
            split="test",
            class_name="class-a",
            support_id="1",
            query_id="2",
            support_image_path=self.root / "support.png",
            support_mask_path=self.root / "support_mask.png",
            query_image_path=self.root / "query.png",
            query_mask_path=self.root / "query_mask.png",
        )
        csv_path = self.root / "roundtrip.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row("episode-a")))
            writer.writeheader()
            writer.writerow(row(episode.stable_id))
        payload = write_result_metadata(
            csv_path,
            episodes=[episode],
            dataset="fss",
            split="test",
            condition="B2",
            model_config={"architecture": "tiny-test", "grid": 37},
            expected_result_rows=1,
            eval_seed=7,
            decoder="vit_h",
            model_state={"weight": torch.tensor([1.0, 2.0])},
        )
        self.assertEqual(payload["manifest_hash"], manifest_hash([episode.stable_id]))
        self.assertEqual(len(str(payload["model_hash"])), 64)
        rows = validate_result_shards(
            [csv_path],
            expected_keys=[episode.stable_id],
            expected_methods=["greedy"],
            budgets=[1],
            seeds=[7],
        )
        self.assertEqual(len(rows), 1)
        self.assertNotEqual(
            state_dict_hash({"weight": torch.tensor([1.0, 2.0])}),
            state_dict_hash({"weight": torch.tensor([1.0, 3.0])}),
        )

    def test_distinct_training_seed_model_hashes_are_valid_partitions(self):
        episode = EpisodeSpec(
            dataset="fss",
            split="test",
            class_name="class-a",
            support_id="1",
            query_id="2",
            support_image_path=self.root / "support.png",
            support_mask_path=self.root / "support_mask.png",
            query_image_path=self.root / "query.png",
            query_mask_path=self.root / "query_mask.png",
        )
        keys = [episode.stable_id]
        paths = []
        for seed, weight in ((17, 1.0), (29, 2.0)):
            csv_path = self.root / f"seed-{seed}.csv"
            fields = [*row("episode-a"), "training_seed", "eval_seed"]
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({**row(episode.stable_id), "training_seed": seed, "eval_seed": 7})
            write_result_metadata(
                csv_path,
                episodes=[episode],
                dataset="fss",
                split="test",
                condition="R5",
                model_config={"architecture": "tiny-test"},
                expected_result_rows=1,
                training_seed=seed,
                eval_seed=7,
                decoder="vit_h",
                model_state={"weight": torch.tensor([weight])},
            )
            paths.append(csv_path)
        result_rows = validate_result_shards(
            paths,
            expected_keys=keys,
            expected_methods=["greedy"],
            budgets=[1],
            seeds=[],
            training_seeds=[17, 29],
            eval_seeds=[7],
        )
        self.assertEqual(len(result_rows), 2)


if __name__ == "__main__":
    unittest.main()
