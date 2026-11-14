from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from sam_rl.cli.evaluation_suite import _existing_status
from sam_rl.episode_manifest import EpisodeSpec, episode_identity
from sam_rl.mask_samples import (
    SAMPLE_RULE,
    mask_sample_dir_for_csv,
    validate_final_mask_samples,
    write_final_mask_samples,
)
from sam_rl.result_manifest import write_result_metadata


def _episodes() -> list[EpisodeSpec]:
    return [
        EpisodeSpec(
            dataset="fss1000",
            split="test",
            class_name="class_a",
            support_id="1",
            query_id=str(index),
            support_image_path=Path("support.jpg"),
            support_mask_path=Path("support.png"),
            query_image_path=Path(f"query_{index}.jpg"),
            query_mask_path=Path(f"query_{index}.png"),
        )
        for index in range(1, 4)
    ]


class MaskSampleTests(unittest.TestCase):
    def test_first_n_samples_are_deterministic_binary_and_linked_to_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episodes = _episodes()
            result_csv = root / "results.csv"
            checkpoint = root / "best.pt"
            checkpoint.write_bytes(b"checkpoint")
            metadata = write_result_metadata(
                result_csv,
                episodes=episodes,
                dataset="fss1000",
                split="test",
                condition="R3",
                model_config={"grid": 37, "use_current_mask": True},
                expected_result_rows=len(episodes),
                training_seed=17,
                eval_seed=0,
                decoder="vit_b",
                checkpoint_path=checkpoint,
                extra={
                    "checkpoint_sha256": "checkpoint-sha",
                    "evaluation_config_sha256": "config-sha",
                    "evaluation_source_sha256": "source-sha",
                    "protocol": "single|fss_test450",
                    "budget": 5,
                },
            )
            masks = [
                np.asarray([[0, 1, 0], [1, 1, 0]], dtype=np.uint8),
                np.asarray([[1, 0, 0], [0, 1, 1]], dtype=np.uint8),
            ]
            entries = list(zip(episodes[:2], masks, strict=True))
            sample_dir = root / "results.mask_samples"
            manifest_path = write_final_mask_samples(
                output_dir=sample_dir,
                episodes=episodes,
                mask_entries=entries,
                sample_count=2,
                result_csv=result_csv,
            )
            self.assertIsNotNone(manifest_path)
            assert manifest_path is not None
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["sampling_rule"], SAMPLE_RULE)
            self.assertEqual(
                manifest["sampled_episode_keys"], [episode_identity(ep) for ep in episodes[:2]]
            )
            self.assertEqual(manifest["eval_seed"], 0)
            self.assertEqual(manifest["evaluation_config_sha256"], "config-sha")
            self.assertEqual(manifest["evaluation_source_sha256"], "source-sha")
            self.assertEqual(manifest["checkpoint_sha256"], "checkpoint-sha")
            self.assertEqual(manifest["model_hash"], metadata["model_hash"])
            for sample in manifest["samples"]:
                pixels = np.asarray(Image.open(sample_dir / sample["mask_png"]))
                self.assertEqual(pixels.ndim, 2)
                self.assertEqual(set(np.unique(pixels)), {0, 255})
            validated = validate_final_mask_samples(
                manifest_path,
                metadata_path=result_csv.with_suffix(".metadata.json"),
                expected_episode_keys=[episode_identity(ep) for ep in episodes],
                sample_count=2,
            )
            self.assertEqual(validated["sample_count"], 2)

    def test_suite_completion_requires_linked_samples_when_enabled(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episodes = _episodes()[:2]
            keys = [episode_identity(episode) for episode in episodes]
            checkpoint = root / "best.pt"
            checkpoint.write_bytes(b"checkpoint")
            import hashlib

            checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            csv_path = root / "test_results.csv"
            csv_path.write_text(
                "method,N,image_id,seed,training_seed,eval_seed,episode_key,iou,dice,boundary_iou,wallclock_ms\n"
                + "\n".join(
                    f"R3,5,{episode.image_id},17,17,0,{key},0.5,0.6,0.4,10.0"
                    for episode, key in zip(episodes, keys, strict=True)
                )
                + "\n",
                encoding="utf-8",
            )
            per_step_path = csv_path.with_suffix(".per_step.csv")
            per_step_path.write_text(
                "episode_key,training_seed,eval_seed,step,iou,point_x,point_y,point_label\n"
                + "\n".join(
                    f"{key},17,0,{step},0.5,20,30,1" for key in keys for step in range(1, 6)
                )
                + "\n",
                encoding="utf-8",
            )
            write_result_metadata(
                csv_path,
                episodes=episodes,
                dataset="fss1000",
                split="test",
                condition="R3",
                model_config={"grid": 37},
                expected_result_rows=len(episodes),
                training_seed=17,
                eval_seed=0,
                decoder="vit_b",
                checkpoint_path=checkpoint,
                extra={
                    "protocol": "single|fss_test450",
                    "budget": 5,
                    "checkpoint_sha256": checkpoint_sha256,
                    "evaluation_config_sha256": "semantic-config-sha",
                    "evaluation_source_sha256": "source-sha",
                },
            )
            kwargs = {
                "keys": keys,
                "condition": "R3",
                "seed": 17,
                "dataset": "fss1000",
                "decoder": "vit_b",
                "protocol": "single|fss_test450",
                "budget": 5,
                "training_status": "completed",
                "checkpoint_exists": True,
                "checkpoint_path": checkpoint,
                "mask_sample_count": 2,
                "checkpoint_sha256": checkpoint_sha256,
                "evaluation_config_sha256": "semantic-config-sha",
                "evaluation_source_sha256": "source-sha",
            }
            status, reason = _existing_status(csv_path, per_step_path, **kwargs)
            self.assertEqual(status, "failed")
            self.assertIn("mask", reason)
            write_final_mask_samples(
                output_dir=mask_sample_dir_for_csv(csv_path),
                episodes=episodes,
                mask_entries=[
                    (episode, np.asarray([[0, 1], [1, 0]], dtype=np.uint8)) for episode in episodes
                ],
                sample_count=2,
                result_csv=csv_path,
            )
            status, reason = _existing_status(csv_path, per_step_path, **kwargs)
            self.assertEqual((status, reason), ("complete", None))

    def test_zero_sample_count_writes_nothing_and_seed_zero_is_valid(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "not-created"
            result = write_final_mask_samples(
                output_dir=output,
                episodes=[],
                mask_entries=[],
                sample_count=0,
                result_csv=Path(temporary) / "missing.csv",
            )
            self.assertIsNone(result)
            self.assertFalse(output.exists())

    def test_sample_entries_must_match_manifest_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            episodes = _episodes()
            result_csv = root / "results.csv"
            checkpoint = root / "best.pt"
            checkpoint.write_bytes(b"checkpoint")
            write_result_metadata(
                result_csv,
                episodes=episodes,
                dataset="fss1000",
                split="test",
                condition="R3",
                model_config={"grid": 37},
                expected_result_rows=len(episodes),
                training_seed=17,
                eval_seed=0,
                decoder="vit_b",
                checkpoint_path=checkpoint,
                extra={
                    "checkpoint_sha256": "checkpoint-sha",
                    "evaluation_config_sha256": "config-sha",
                    "evaluation_source_sha256": "source-sha",
                    "protocol": "single|fss_test450",
                },
            )
            with self.assertRaisesRegex(ValueError, "manifest order"):
                write_final_mask_samples(
                    output_dir=root / "samples",
                    episodes=episodes,
                    mask_entries=[(episodes[1], np.zeros((2, 2), dtype=np.uint8))],
                    sample_count=1,
                    result_csv=result_csv,
                )


if __name__ == "__main__":
    unittest.main()
