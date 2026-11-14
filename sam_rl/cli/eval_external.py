"""Evaluate a published external one-shot segmenter on a fixed episode manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sam_rl.baseline_eval import boundary_iou_score, dice_score, iou_score
from sam_rl.episode_manifest import EpisodeSpec, episode_identity, load_episode_manifest
from sam_rl.external_adapters import (
    EpisodeInput,
    Predictor,
    build_adapter,
    external_revision,
    load_binary_mask,
    resize_binary_mask,
)
from sam_rl.provenance import file_fingerprint
from sam_rl.result_manifest import write_result_metadata

EXPECTED_REVISIONS = {
    "persam": "a7e87245156e0b8efcba10a15a745e699c679d9d",
    "matcher": "d59b6c761f427d52b1ba00132a6b3488f169d039",
    "sansa": "f562b5642940e08a57e3da2608b804b99af3763b",
}
FIELDS = [
    "method",
    "episode_key",
    "dataset",
    "split",
    "class_name",
    "support_id",
    "query_id",
    "iou",
    "dice",
    "boundary_iou",
    "iou_native",
    "dice_native",
    "boundary_iou_native",
    "wallclock_ms",
    "prediction_path",
]


def episode_metrics(
    prediction: np.ndarray, target: np.ndarray, *, evaluation_size: int = 224
) -> tuple[dict[str, float], np.ndarray]:
    """Compute shared headline scores at 224-square plus native-size secondary scores."""
    gt = (np.asarray(target) > 0).astype(np.uint8)
    pred_native = resize_binary_mask(prediction, gt.shape[0], gt.shape[1])
    pred = resize_binary_mask(pred_native, evaluation_size, evaluation_size)
    target_small = resize_binary_mask(gt, evaluation_size, evaluation_size)
    metrics = {
        "iou": iou_score(pred, target_small),
        "dice": dice_score(pred, target_small),
        "boundary_iou": boundary_iou_score(pred, target_small),
        "iou_native": iou_score(pred_native, gt),
        "dice_native": dice_score(pred_native, gt),
        "boundary_iou_native": boundary_iou_score(pred_native, gt),
    }
    return metrics, pred_native


def validate_episode_pairs(episodes: list[EpisodeSpec]) -> None:
    """Prevent support-mask annotation from being the query label in any episode."""
    for episode in episodes:
        same_id = str(episode.support_id) == str(episode.query_id)
        same_asset = episode.support_image_path.resolve() == episode.query_image_path.resolve()
        if same_id or same_asset:
            raise ValueError(
                f"Self-pair detected for class {episode.class_name}: "
                f"support={episode.support_id}, query={episode.query_id}."
            )


def reduce_episode_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Report episode macro and class macro; fixed ten-query classes make these equal."""
    if not rows:
        raise ValueError("Cannot reduce an empty result set.")
    by_class: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_class[str(row["class_name"])].append(float(row["iou"]))
    return {
        "episode_macro_iou": float(np.mean([float(row["iou"]) for row in rows])),
        "class_macro_iou": float(np.mean([np.mean(values) for values in by_class.values()])),
        "class_count": float(len(by_class)),
        "episode_count": float(len(rows)),
    }


def evaluate_episodes(
    episodes: list[EpisodeSpec],
    predictor: Predictor,
    *,
    method: str,
    output_dir: Path,
    evaluation_size: int = 224,
) -> list[dict[str, Any]]:
    """Run the adapter with annotation-free inputs and write one row per episode."""
    rows: list[dict[str, Any]] = []
    predictions = output_dir / "predictions"
    predictions.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        try:
            start = time.perf_counter()
            prediction = predictor.predict(EpisodeInput.from_episode(episode))
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            ground_truth = load_binary_mask(episode.query_mask_path)
            metrics, pred = episode_metrics(
                prediction, ground_truth, evaluation_size=evaluation_size
            )
            pred_path = predictions / f"{episode_identity(episode).split(':', 1)[1]}.png"
            Image.fromarray(pred * 255).save(pred_path)
            rows.append(
                {
                    "method": method,
                    "episode_key": episode_identity(episode),
                    "dataset": episode.dataset,
                    "split": episode.split,
                    "class_name": episode.class_name,
                    "support_id": episode.support_id,
                    "query_id": episode.query_id,
                    **metrics,
                    "wallclock_ms": elapsed_ms,
                    "prediction_path": str(Path("predictions") / pred_path.name),
                }
            )
        except Exception as exc:
            _write_csv(output_dir / "partial_episode_results.csv", rows)
            (output_dir / "failure.json").write_text(
                json.dumps(
                    {
                        "stage": "episode_evaluation",
                        "episode_key": episode_identity(episode),
                        "class_name": episode.class_name,
                        "support_id": episode.support_id,
                        "query_id": episode.query_id,
                        "completed_episode_count": len(rows),
                        "exception_type": type(exc).__name__,
                        "exception_message": str(exc),
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            raise
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _checkpoint_files(config: dict[str, Any], method: str) -> list[Path]:
    paths: list[Path] = []
    for key, value in config.items():
        is_hf_model_dir = method == "seggpt" and key == "model_id"
        if (
            "checkpoint" not in key.lower()
            and key.lower() not in {"weights", "model_path"}
            and not is_hf_model_dir
        ):
            continue
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Required model weights not found for {key}: {path}")
        if path.is_file():
            paths.append(path)
        else:
            candidates = sorted(
                item
                for item in path.rglob("*")
                if item.is_file() and item.suffix in {".safetensors", ".bin", ".pt", ".pth"}
            )
            if not candidates:
                raise FileNotFoundError(
                    f"No model weight files found in checkpoint directory {path}"
                )
            paths.extend(candidates)
    if not paths:
        raise ValueError(
            "adapter-config must name existing checkpoint paths so weights can be fingerprinted."
        )
    unique = {str(path): path for path in paths}
    return list(unique.values())


def _validate_seggpt_snapshot(config: dict[str, Any]) -> tuple[Path, list[Path]]:
    model_value = config.get("model_id")
    revision = config.get("revision")
    if not isinstance(model_value, str) or not Path(model_value).expanduser().is_dir():
        raise ValueError(
            "SegGPT requires adapter-config['model_id'] to be a local Hugging Face snapshot "
            "directory; remote model IDs are disabled so weights cannot be fetched implicitly."
        )
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(char not in "0123456789abcdef" for char in revision.lower())
    ):
        raise ValueError(
            "SegGPT adapter-config['revision'] must be the pinned 40-character Hugging Face commit SHA."
        )
    snapshot = Path(model_value).expanduser().resolve()
    if snapshot.name.lower() != revision.lower():
        raise ValueError(
            "SegGPT model_id must name the Hugging Face snapshot directory whose basename "
            f"matches revision {revision}; got {snapshot.name!r}."
        )
    config_paths = [snapshot / "config.json"]
    processor_paths = [
        snapshot / name for name in ("preprocessor_config.json", "processor_config.json")
    ]
    if not config_paths[0].is_file():
        raise FileNotFoundError(f"SegGPT snapshot is missing model config: {config_paths[0]}")
    available_processor_configs = [path for path in processor_paths if path.is_file()]
    if not available_processor_configs:
        raise FileNotFoundError(
            f"SegGPT snapshot is missing processor config: expected one of {processor_paths}."
        )
    model_weights = [
        path
        for path in snapshot.rglob("*")
        if path.is_file() and path.suffix in {".safetensors", ".bin", ".pt", ".pth"}
    ]
    if not model_weights:
        raise FileNotFoundError(f"SegGPT snapshot has no local model weights: {snapshot}")
    return snapshot, [*model_weights, *config_paths, *available_processor_configs]


def _hash_paths(paths: list[Path]) -> dict[str, dict[str, Any]]:
    return {str(path): file_fingerprint(path, use_memo=False) for path in paths}


def _read_source_revision(path_value: str | None) -> dict[str, Any] | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Source revision record not found: {path}")
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **values}


def _runtime_environment(*, device: str | None) -> dict[str, Any]:
    distributions = (
        "torch",
        "torchvision",
        "numpy",
        "Pillow",
        "transformers",
        "accelerate",
        "opencv-python-headless",
        "hydra-core",
        "omegaconf",
        "scipy",
        "timm",
        "POT",
    )
    packages: dict[str, str] = {}
    for name in distributions:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    from torch import version as torch_version

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": device,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_build": torch_version.cuda,
        "packages": packages,
    }


def run(args: argparse.Namespace) -> Path:
    method = str(args.method).lower()
    config = json.loads(Path(args.adapter_config).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("adapter-config JSON must be an object.")
    evaluation_size = int(getattr(args, "evaluation_size", 224))
    if evaluation_size < 1:
        raise ValueError("evaluation-size must be positive.")
    snapshot_files: list[Path] = []
    seggpt_model_config: dict[str, Any] | None = None
    seggpt_processor_config: dict[str, Any] | None = None
    if method == "seggpt":
        snapshot, snapshot_files = _validate_seggpt_snapshot(config)
        seggpt_model_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
        processor_config_file = next(
            path
            for path in snapshot_files
            if path.name in {"preprocessor_config.json", "processor_config.json"}
        )
        seggpt_processor_config = json.loads(processor_config_file.read_text(encoding="utf-8"))
    checkpoint_paths = _checkpoint_files(config, method)
    repo = config.get("repo")
    source_revision: dict[str, object] | None = None
    if method in EXPECTED_REVISIONS:
        if not repo:
            raise ValueError(
                f"{method} requires an external source checkout in adapter-config['repo']."
            )
        source_revision = external_revision(repo)
        expected = EXPECTED_REVISIONS[method]
        if source_revision["commit"] != expected:
            raise ValueError(
                f"{method} source commit mismatch: expected {expected}, got {source_revision['commit']}."
            )
    max_episodes = getattr(args, "max_episodes", None)
    if max_episodes is not None and int(max_episodes) < 1:
        raise ValueError("max-episodes must be positive when supplied.")
    episodes = load_episode_manifest(
        args.manifest,
        max_episodes=max_episodes,
        validate_paths=True,
        data_root=args.data_root,
    )
    if not episodes:
        raise ValueError("Episode manifest is empty.")
    validate_episode_pairs(episodes)
    keys = [episode_identity(episode) for episode in episodes]
    if len(set(keys)) != len(keys):
        raise ValueError("Episode manifest contains duplicate episode identities.")
    output_root = Path(args.output_root).expanduser().resolve()
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = (
            output_root / f"{method}_{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
        )
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        predictor = build_adapter(method, config)
    except Exception as exc:
        (output_dir / "failure.json").write_text(
            json.dumps(
                {
                    "stage": "adapter_initialization",
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    if method == "seggpt":
        observed_revision = getattr(
            getattr(getattr(predictor, "model", None), "config", None), "_commit_hash", None
        )
        if observed_revision and observed_revision.lower() != str(config["revision"]).lower():
            raise ValueError(
                "Loaded SegGPT snapshot commit disagrees with adapter-config revision: "
                f"expected {config['revision']}, found {observed_revision}."
            )
    rows = evaluate_episodes(
        episodes,
        predictor,
        method=method,
        output_dir=output_dir,
        evaluation_size=evaluation_size,
    )
    csv_path = output_dir / "episode_results.csv"
    _write_csv(csv_path, rows)
    summary = reduce_episode_scores(rows)
    summary_payload = {"method": method, **summary}
    predictor_model = getattr(predictor, "model", None)
    hf_config = getattr(predictor_model, "config", None)
    resolved_model_commit = getattr(hf_config, "_commit_hash", None) or config.get("revision")
    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8"
    )
    extra: dict[str, object] = {
        "protocol": "fixed_manifest_one_shot_support_mask",
        "episode_manifest_path": str(Path(args.manifest).expanduser().resolve()),
        "episode_manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        "checkpoint_sha256": _hash_paths(checkpoint_paths),
        "model_snapshot_files_sha256": _hash_paths(snapshot_files) if snapshot_files else None,
        "metric_contract": {
            "headline_image_size": [evaluation_size, evaluation_size],
            "resize_interpolation": "nearest",
            "headline_metrics": ["iou", "dice", "boundary_iou"],
            "secondary_native_metrics": ["iou_native", "dice_native", "boundary_iou_native"],
        },
        "summary": summary_payload,
        "evaluation_environment": _runtime_environment(device=config.get("device")),
        "external_source": source_revision
        or {
            "source": "Hugging Face Transformers",
            "model_repo": "BAAI/seggpt-vit-large",
            "snapshot_path": config.get("model_id"),
            "revision": config.get("revision"),
            "resolved_model_commit": resolved_model_commit,
            "model_config": seggpt_model_config,
            "processor_config": seggpt_processor_config,
        },
    }
    environment_lock = getattr(args, "environment_lock", None)
    if environment_lock:
        lock_path = Path(environment_lock).expanduser().resolve()
        extra["environment_lock"] = {
            "path": str(lock_path),
            **file_fingerprint(lock_path, use_memo=False),
        }
    source_revision_path = getattr(args, "source_revision_file", None)
    if source_revision_path:
        extra["evaluation_source_revision"] = _read_source_revision(source_revision_path)
    primary_checkpoint = checkpoint_paths[0]
    write_result_metadata(
        csv_path,
        episodes=episodes,
        dataset=episodes[0].dataset,
        split=episodes[0].split,
        condition=f"external_{method}",
        model_config={"method": method, **config},
        expected_result_rows=len(rows),
        checkpoint_path=primary_checkpoint,
        extra=extra,
    )
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=("seggpt", "persam", "matcher", "sansa"))
    parser.add_argument("--manifest", required=True, help="Frozen portable episode manifest.")
    parser.add_argument(
        "--adapter-config", required=True, help="JSON object with source and checkpoint paths."
    )
    parser.add_argument(
        "--evaluation-size",
        type=int,
        default=224,
        help="Square resolution used to score shared headline metrics (default: 224).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional first-N manifest subset for a smoke run; omit for the full manifest.",
    )
    parser.add_argument(
        "--environment-lock",
        default=None,
        help="Optional frozen package list whose hash is recorded in result metadata.",
    )
    parser.add_argument(
        "--source-revision-file",
        default=None,
        help="Optional source-revision record for the isolated evaluator archive.",
    )
    parser.add_argument("--output-root", default="artifacts/external_eval")
    parser.add_argument(
        "--output-dir", default=None, help="Optional new directory; it must not already exist."
    )
    parser.add_argument(
        "--data-root", default=None, help="Resolve relative manifest asset paths under this root."
    )
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(result)


if __name__ == "__main__":
    main()
