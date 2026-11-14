from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from PIL import Image

from sam_rl.cli.validate_oracle import (
    expected_oracle_provenance,
    validate_oracle_payload,
    validate_oracle_split,
    validate_shard_manifests,
)
from sam_rl.episode_manifest import EpisodeSpec, episode_identity


def make_episode(root: Path, index: int) -> EpisodeSpec:
    (root / "images").mkdir(exist_ok=True)
    (root / "masks").mkdir(exist_ok=True)
    for image_id in (f"support-{index}", f"query-{index}"):
        Image.new("RGB", (128, 128), (30, 40, 50)).save(root / "images" / f"{image_id}.jpg")
        Image.new("L", (128, 128), 255).save(root / "masks" / f"{image_id}.png")
    return EpisodeSpec(
        dataset="fss1000",
        split="train",
        class_name=f"class-{index}",
        support_id="1",
        query_id="10",
        support_image_path=root / "images" / f"support-{index}.jpg",
        support_mask_path=root / "masks" / f"support-{index}.png",
        query_image_path=root / "images" / f"query-{index}.jpg",
        query_mask_path=root / "masks" / f"query-{index}.png",
    )


def make_config(root: Path) -> dict[str, Any]:
    sam_path = root / "sam.ckpt"
    dino_path = root / "dino.ckpt"
    sam_path.write_bytes(b"sam weights test")
    dino_path.write_bytes(b"dino weights test")
    return {
        "models": {
            "image_size": 224,
            "precision": "fp32",
            "dino_checkpoint": str(dino_path),
            "oracle_sam_checkpoint": str(sam_path),
            "oracle_sam_model_type": "vit_b",
        },
        "oracle": {
            "max_budget": 5,
            "candidate_pool_size": None,
            "precision": "fp32",
            "decoder_policy": "single",
            "use_bfloat16": False,
        },
        "protocol": {"precision": "fp32", "decoder_policy": "single"},
        "model": {"grid": 37},
        "train": {"budget": 5},
    }


def make_payload(episode: EpisodeSpec, config: dict[str, Any]) -> dict[str, Any]:
    provenance = expected_oracle_provenance(episode, config, repo_root=Path.cwd())
    steps = []
    for step_index in range(1, 6):
        location = step_index - 1
        grid_x, grid_y = location % 37, location // 37
        steps.append(
            {
                "step_index": step_index,
                "location_idx": location,
                "grid_x": grid_x,
                "grid_y": grid_y,
                "point_x": round(((grid_x + 0.5) / 37) * 224),
                "point_y": round(((grid_y + 0.5) / 37) * 224),
                "iou": 0.5,
                "dice": 0.6,
                "boundary_iou": 0.4,
                "sim_value": 0.7,
                "wallclock_ms": 2.0,
            }
        )
    return {
        "episode_key": episode_identity(episode),
        "dataset": episode.dataset,
        "split": episode.split,
        "class_name": episode.class_name,
        "image_id": episode.image_id,
        "grid": 37,
        "image_size": [224, 224],
        "max_budget": 5,
        "candidate_pool_size": None,
        "oracle_sam_checkpoint": config["models"]["oracle_sam_checkpoint"],
        "oracle_sam_model_type": "vit_b",
        "sam_prompt_mode": "one_shot_positive_points",
        "decoder_policy": "single",
        "provenance": provenance,
        "steps": steps,
    }


class OracleValidationTests(unittest.TestCase):
    def test_payload_validates_identity_protocol_points_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            episode = make_episode(root, 0)
            config = make_config(root)
            payload = make_payload(episode, config)
            validate_oracle_payload(
                payload,
                episode=episode,
                expected_provenance=payload["provenance"],
                config=config,
            )
            bad_metric = json.loads(json.dumps(payload))
            bad_metric["steps"][0]["iou"] = float("nan")
            with self.assertRaisesRegex(ValueError, "finite and in \\[0, 1\\]"):
                validate_oracle_payload(
                    bad_metric,
                    episode=episode,
                    expected_provenance=payload["provenance"],
                    config=config,
                )
            bad_point = json.loads(json.dumps(payload))
            bad_point["steps"][1]["location_idx"] = 0
            bad_point["steps"][1]["grid_x"] = 0
            bad_point["steps"][1]["grid_y"] = 0
            bad_point["steps"][1]["point_x"] = bad_point["steps"][0]["point_x"]
            bad_point["steps"][1]["point_y"] = bad_point["steps"][0]["point_y"]
            with self.assertRaisesRegex(ValueError, "duplicate selected grid cell"):
                validate_oracle_payload(
                    bad_point,
                    episode=episode,
                    expected_provenance=payload["provenance"],
                    config=config,
                )
            bad_center = json.loads(json.dumps(payload))
            bad_center["steps"][0]["point_x"] += 1
            with self.assertRaisesRegex(ValueError, "does not match grid-cell center"):
                validate_oracle_payload(
                    bad_center,
                    episode=episode,
                    expected_provenance=payload["provenance"],
                    config=config,
                )
            wrong_processed_shape = json.loads(json.dumps(payload))
            wrong_processed_shape["image_size"] = [128, 128]
            with self.assertRaisesRegex(ValueError, "processed image_size"):
                validate_oracle_payload(
                    wrong_processed_shape,
                    episode=episode,
                    expected_provenance=payload["provenance"],
                    config=config,
                )

    def test_split_checks_expected_files_and_content_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cache_dir = root / "oracle" / "train"
            cache_dir.mkdir(parents=True)
            episodes = [make_episode(root, index) for index in range(2)]
            config = make_config(root)
            for episode in episodes:
                path = cache_dir / f"{episode.split}_{episode.image_id}.json"
                path.write_text(json.dumps(make_payload(episode, config)))
            report = validate_oracle_split(
                episodes, oracle_dir=cache_dir, config=config, repo_root=root
            )
            self.assertEqual(report["validated_label_files"], 2)
            extra = cache_dir / "unexpected.json"
            extra.write_text("{}")
            pilot_report = validate_oracle_split(
                episodes,
                oracle_dir=cache_dir,
                config=config,
                repo_root=root,
                max_episodes=1,
            )
            self.assertEqual(pilot_report["validated_label_files"], 1)
            with self.assertRaisesRegex(ValueError, "file set mismatch"):
                validate_oracle_split(episodes, oracle_dir=cache_dir, config=config, repo_root=root)
            extra.unlink()
            episodes[0].query_mask_path.write_bytes(b"changed query label")
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                validate_oracle_split(episodes, oracle_dir=cache_dir, config=config, repo_root=root)

    def test_shards_require_exact_complete_disjoint_union(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            episodes = [make_episode(root, index) for index in range(4)]
            ordered_ids = [episode_identity(episode) for episode in episodes]
            paths = []
            for shard_index in range(2):
                assigned = [
                    value for index, value in enumerate(ordered_ids) if index % 2 == shard_index
                ]
                path = root / f"shard-{shard_index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "shard_index": shard_index,
                            "split": "train",
                            "num_shards": 2,
                            "ordered_episode_ids": ordered_ids,
                            "expected_episode_ids": assigned,
                            "completed_episode_ids": assigned,
                            "status": "complete",
                        }
                    )
                )
                paths.append(path)
            self.assertEqual(validate_shard_manifests(paths, episodes)["episode_count"], 4)
            partial = json.loads(paths[1].read_text())
            partial["completed_episode_ids"] = partial["completed_episode_ids"][:-1]
            paths[1].write_text(json.dumps(partial))
            with self.assertRaisesRegex(ValueError, "missing, extra, or incomplete"):
                validate_shard_manifests(paths, episodes)


if __name__ == "__main__":
    unittest.main()
