from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
from PIL import Image

from sam_rl.cli import eval_external
from sam_rl.cli.eval_external import (
    episode_metrics,
    evaluate_episodes,
    reduce_episode_scores,
    validate_episode_pairs,
)
from sam_rl.episode_manifest import EpisodeSpec, episode_to_record
from sam_rl.external_adapters import (
    EpisodeInput,
    _matcher_default_args,
    _stack_sansa_frames,
    build_adapter,
)


class RecordingPredictor:
    def __init__(self, prediction: np.ndarray):
        self.prediction = prediction
        self.inputs: list[EpisodeInput] = []

    def predict(self, episode: EpisodeInput) -> np.ndarray:
        self.inputs.append(episode)
        return self.prediction


class FailingPredictor(RecordingPredictor):
    def predict(self, episode: EpisodeInput) -> np.ndarray:
        self.inputs.append(episode)
        if episode.query_id == "2":
            raise RuntimeError("mock inference failure")
        return self.prediction


def _episode(
    tmp_path: Path, *, class_name: str, support_id: str, query_id: str, target: np.ndarray
) -> EpisodeSpec:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    mask = target.astype(np.uint8) * 255
    paths = {
        "support_image_path": tmp_path / f"{class_name}-s{support_id}.jpg",
        "support_mask_path": tmp_path / f"{class_name}-s{support_id}.png",
        "query_image_path": tmp_path / f"{class_name}-q{query_id}.jpg",
        "query_mask_path": tmp_path / f"{class_name}-q{query_id}.png",
    }
    Image.fromarray(image).save(paths["support_image_path"])
    Image.fromarray(mask).save(paths["support_mask_path"])
    Image.fromarray(image).save(paths["query_image_path"])
    Image.fromarray(mask).save(paths["query_mask_path"])
    return EpisodeSpec("FSS-1000", "test", class_name, support_id, query_id, **paths)


def test_adapter_input_does_not_expose_query_annotation() -> None:
    names = {field.name for field in fields(EpisodeInput)}
    assert "query_mask_path" not in names
    assert "query_image_path" in names


def test_evaluator_preserves_manifest_pairs_and_scores_episode_masks(tmp_path: Path) -> None:
    target_a = np.zeros((4, 5), dtype=np.uint8)
    target_a[1:3, 2:4] = 1
    target_b = np.zeros((4, 5), dtype=np.uint8)
    target_b[0:2, 0:2] = 1
    episodes = [
        _episode(tmp_path, class_name="cat", support_id="7", query_id="1", target=target_a),
        _episode(tmp_path, class_name="cat", support_id="1", query_id="2", target=target_b),
    ]
    predictor = RecordingPredictor(target_a)
    rows = evaluate_episodes(episodes, predictor, method="mock", output_dir=tmp_path / "run")
    assert [(item.support_id, item.query_id) for item in predictor.inputs] == [
        ("7", "1"),
        ("1", "2"),
    ]
    assert all(not hasattr(item, "query_mask_path") for item in predictor.inputs)
    assert [row["iou"] for row in rows] == [1.0, 0.0]
    assert [row["iou_native"] for row in rows] == [1.0, 0.0]
    assert all((tmp_path / "run" / row["prediction_path"]).is_file() for row in rows)


def test_evaluator_preserves_partial_outputs_and_failure_details(tmp_path: Path) -> None:
    target = np.zeros((4, 5), dtype=np.uint8)
    target[1:3, 2:4] = 1
    episodes = [
        _episode(tmp_path, class_name="cat", support_id="7", query_id="1", target=target),
        _episode(tmp_path, class_name="cat", support_id="1", query_id="2", target=target),
    ]
    output_dir = tmp_path / "partial-run"
    try:
        evaluate_episodes(
            episodes,
            FailingPredictor(target),
            method="mock",
            output_dir=output_dir,
        )
    except RuntimeError as error:
        assert str(error) == "mock inference failure"
    else:
        raise AssertionError("The evaluator did not surface the failed episode.")

    assert len(list((output_dir / "predictions").glob("*.png"))) == 1
    assert (output_dir / "partial_episode_results.csv").is_file()
    failure = json.loads((output_dir / "failure.json").read_text(encoding="utf-8"))
    assert failure["class_name"] == "cat"
    assert failure["support_id"] == "1"
    assert failure["query_id"] == "2"
    assert failure["completed_episode_count"] == 1
    assert failure["exception_message"] == "mock inference failure"


def test_metrics_resize_binary_prediction_with_nearest_neighbor() -> None:
    target = np.zeros((4, 4), dtype=np.uint8)
    target[1:3, 1:3] = 1
    prediction = np.ones((2, 2), dtype=np.uint8)
    metrics, pred_native = episode_metrics(prediction, target)
    iou, dice = metrics["iou"], metrics["dice"]
    assert iou == 0.25
    assert dice == 0.4
    assert pred_native.shape == target.shape
    assert metrics["iou_native"] == 0.25


def test_macro_reducer_averages_episode_then_class() -> None:
    rows = [
        {"class_name": "a", "iou": 0.0},
        {"class_name": "a", "iou": 1.0},
        {"class_name": "b", "iou": 0.5},
    ]
    reduced = reduce_episode_scores(rows)
    assert reduced["episode_macro_iou"] == 0.5
    assert reduced["class_macro_iou"] == 0.5
    assert reduced["class_count"] == 2.0
    assert reduced["episode_count"] == 3.0


def test_unknown_external_method_is_not_silently_scored() -> None:
    try:
        build_adapter("imaginary", {})
    except NotImplementedError as error:
        assert "faithful" in str(error)
    else:
        raise AssertionError("Unsupported method did not fail explicitly.")


def test_manifest_preflight_rejects_self_pair_by_id_or_asset(tmp_path: Path) -> None:
    target = np.zeros((4, 5), dtype=np.uint8)
    same_id = _episode(tmp_path, class_name="cat", support_id="1", query_id="1", target=target)
    try:
        validate_episode_pairs([same_id])
    except ValueError as error:
        assert "Self-pair" in str(error)
    else:
        raise AssertionError("Self-pair by identity was not rejected.")

    same_asset = EpisodeSpec(
        same_id.dataset,
        same_id.split,
        same_id.class_name,
        "7",
        "1",
        same_id.support_image_path,
        same_id.support_mask_path,
        same_id.support_image_path,
        same_id.query_mask_path,
    )
    try:
        validate_episode_pairs([same_asset])
    except ValueError as error:
        assert "Self-pair" in str(error)
    else:
        raise AssertionError("Self-pair by shared image asset was not rejected.")


def test_sansa_frames_have_batch_time_channel_height_width_order() -> None:
    import torch

    support = torch.zeros((3, 518, 518))
    query = torch.ones_like(support)
    frames = _stack_sansa_frames(torch, support, query)
    assert tuple(frames.shape) == (1, 2, 3, 518, 518)
    assert frames[0, 0, 0, 0, 0].item() == 0
    assert frames[0, 1, 0, 0, 0].item() == 1


def test_matcher_adapter_uses_official_fss_script_parameters(tmp_path: Path) -> None:
    import torch

    args = _matcher_default_args(torch.device("cpu"), tmp_path / "sam.pth", tmp_path / "dinov2.pth")
    assert (args.alpha, args.beta, args.exp) == (0.8, 0.2, 1.0)
    assert args.use_points_or_centers is False
    assert args.sample_range == (4, 6)
    assert args.max_sample_iterations == 30
    assert args.num_merging_mask == 10


def test_cli_writes_strict_manifest_and_checkpoint_metadata(tmp_path: Path, monkeypatch) -> None:
    target = np.zeros((4, 5), dtype=np.uint8)
    target[1:3, 1:3] = 1
    episode = _episode(tmp_path, class_name="cat", support_id="7", query_id="1", target=target)
    manifest = tmp_path / "episodes.json"
    manifest.write_text(json.dumps({"episodes": [episode_to_record(episode)]}), encoding="utf-8")
    revision = "a" * 40
    checkpoint_dir = tmp_path / revision
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "model.safetensors"
    checkpoint.write_bytes(b"mock-weights")
    (checkpoint_dir / "config.json").write_text('{"model_type":"seggpt"}', encoding="utf-8")
    (checkpoint_dir / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    config = tmp_path / "adapter.json"
    config.write_text(
        json.dumps({"model_id": str(checkpoint_dir), "revision": revision}), encoding="utf-8"
    )
    monkeypatch.setattr(
        eval_external, "build_adapter", lambda method, values: RecordingPredictor(target)
    )
    args = argparse.Namespace(
        method="seggpt",
        manifest=str(manifest),
        adapter_config=str(config),
        output_root=str(tmp_path / "outputs"),
        output_dir=None,
        data_root=None,
    )
    output = eval_external.run(args)
    metadata = json.loads((output / "episode_results.metadata.json").read_text(encoding="utf-8"))
    assert metadata["expected_episode_count"] == 1
    assert metadata["expected_result_rows"] == 1
    assert metadata["episode_manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert metadata["model_hash"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert (
        metadata["checkpoint_sha256"][str(checkpoint.resolve())]["sha256"] == metadata["model_hash"]
    )
