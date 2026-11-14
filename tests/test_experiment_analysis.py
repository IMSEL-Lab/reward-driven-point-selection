from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from sam_rl.cli.analyze_experiments import analyze_collection
from sam_rl.episode_manifest import EpisodeSpec, episode_identity
from sam_rl.result_manifest import manifest_hash


def _episode_rows() -> list[dict[str, str]]:
    rows = []
    for class_name, query_id in (("class_a", "q1"), ("class_a", "q2"), ("class_b", "q3")):
        episode = EpisodeSpec(
            dataset="fss1000",
            split="test",
            class_name=class_name,
            support_id="support0",
            query_id=query_id,
            support_image_path=Path("support.png"),
            support_mask_path=Path("support_mask.png"),
            query_image_path=Path(f"{query_id}.png"),
            query_mask_path=Path(f"{query_id}_mask.png"),
        )
        rows.append({"episode_key": episode_identity(episode), "image_id": episode.image_id})
    return rows


def _collection(root: Path) -> dict:
    episodes = _episode_rows()
    keys = [row["episode_key"] for row in episodes]
    runs = []
    seed_values = {1: [0.1, 0.3, 0.5], 2: [0.5, 0.7, 0.9]}
    for condition in ("A", "B"):
        for train_seed, values in seed_values.items():
            csv_path = root / f"{condition}-{train_seed}.csv"
            metadata_path = csv_path.with_suffix(".metadata.json")
            per_step_path = csv_path.with_suffix(".per_step.csv")
            rows = []
            for episode, value in zip(episodes, values, strict=True):
                rows.append(
                    {
                        "method": condition,
                        "N": 2,
                        "image_id": episode["image_id"],
                        "episode_key": episode["episode_key"],
                        "seed": train_seed,
                        "training_seed": train_seed,
                        "eval_seed": 0,
                        "iou": value,
                        "dice": value,
                        "boundary_iou": value,
                        "wallclock_ms": 1.0,
                    }
                )
            _write_csv(csv_path, rows)
            _write_csv(
                per_step_path,
                [
                    {
                        "episode_key": episode["episode_key"],
                        "training_seed": train_seed,
                        "eval_seed": 0,
                        "step": step,
                        "iou": value,
                    }
                    for episode, value in zip(episodes, values, strict=True)
                    for step in (1, 2)
                ],
            )
            metadata_path.write_text(
                json.dumps(
                    {
                        "dataset": "fss1000",
                        "split": "test",
                        "condition": condition,
                        "decoder": "vit_b",
                        "training_seed": train_seed,
                        "eval_seed": 0,
                        "model_config_hash": f"config-{condition}",
                        "model_hash": f"model-{condition}-{train_seed}",
                        "manifest_hash": manifest_hash(keys),
                        "expected_episode_keys": keys,
                        "expected_result_rows": len(keys),
                        "shard_episode_keys": keys,
                        "budget": 2,
                    }
                ),
                encoding="utf-8",
            )
            runs.append(
                {
                    "condition": condition,
                    "dataset": "fss1000",
                    "split": "test",
                    "decoder": "vit_b",
                    "protocol": "single",
                    "training_seed": train_seed,
                    "eval_seed": 0,
                    "status": "complete",
                    "csv": str(csv_path),
                    "metadata": str(metadata_path),
                    "per_step_csv": str(per_step_path),
                }
            )
    return {
        "expected_conditions": ["A", "B"],
        "expected_training_seeds": [1, 2],
        "expected_eval_seeds": [0],
        "bootstrap_resamples": 100,
        "bootstrap_seed": 7,
        "runs": runs,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_seed_means_are_averaged_before_across_seed_spread_and_pairs(tmp_path: Path) -> None:
    collection = _collection(tmp_path)
    result = analyze_collection(collection, base_dir=tmp_path)
    summary = next(
        row
        for row in result["condition_summaries"]
        if row["condition"] == "A" and row["budget"] == 2
    )
    assert summary["per_training_seed_mean_iou"] == {
        "1": pytest.approx(0.3),
        "2": pytest.approx(0.7),
    }
    assert summary["mean_iou"] == pytest.approx(0.5)
    assert summary["std_iou_across_training_seeds"] == pytest.approx(0.2828427125)
    contrast = result["paired_contrasts"][0]
    assert contrast["right_minus_left_iou"] == pytest.approx(0.0)
    assert contrast["cluster_bootstrap_95_ci"] == [0.0, 0.0]
    assert contrast["n_clusters"] == 2
    assert len(result["per_step_summaries_and_contrasts"]) == 6


def test_missing_seed_ledger_entry_is_an_error(tmp_path: Path) -> None:
    collection = _collection(tmp_path)
    collection["runs"] = [
        run
        for run in collection["runs"]
        if not (run["condition"] == "B" and run["training_seed"] == 2)
    ]
    with pytest.raises(ValueError, match="Missing run ledger entry"):
        analyze_collection(collection, base_dir=tmp_path)


def test_nonfinite_result_metric_is_rejected(tmp_path: Path) -> None:
    collection = _collection(tmp_path)
    path = Path(collection["runs"][0]["csv"])
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["iou"] = "nan"
    _write_csv(path, rows)
    with pytest.raises(ValueError, match="Nonfinite iou"):
        analyze_collection(collection, base_dir=tmp_path)
