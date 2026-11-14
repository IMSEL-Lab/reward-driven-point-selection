from __future__ import annotations

import argparse
import hashlib
import itertools
import time
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import numpy as np

from sam_rl.baseline_eval import (
    CsvResultWriter,
    DinoSimilarityCache,
    boundary_iou_score,
    dice_score,
    iou_score,
    load_dino_model,
    load_episode_arrays,
    load_sam_predictor,
    predict_mask_from_points,
    save_overlay_artifacts,
    snap_points_to_grid,
    write_json,
)
from sam_rl.cli.common import choose_device, load_config
from sam_rl.episode_manifest import episode_identity
from sam_rl.episode_source import build_episodes_from_dataset_cfg
from sam_rl.point_selectors import build_method
from sam_rl.provenance import file_fingerprint, stable_hash
from sam_rl.result_manifest import write_result_metadata

STOCHASTIC_FAMILIES = {"sim_weighted_random", "dpp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate non-trained point-selection baselines on cached DINO similarity maps."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("eval", "sweep", "frozen"), default="frozen")
    parser.add_argument("--split", default=None, help="Optional dataset split override.")
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Optional episode-count override for smoke runs.",
    )
    parser.add_argument("--output-csv", default=None, help="Optional CSV path override.")
    parser.add_argument("--overlay-dir", default=None, help="Optional overlay directory override.")
    parser.add_argument("--selection-metric", default=None, choices=("iou", "dice", "boundary_iou"))
    parser.add_argument(
        "--best-params-json", default=None, help="Where sweep mode writes frozen hyperparameters."
    )
    parser.add_argument(
        "--frozen-params", default=None, help="Frozen hyperparameter JSON to use in frozen mode."
    )
    parser.add_argument(
        "--shard-index", type=int, default=0, help="0-based shard index for episode partitioning."
    )
    parser.add_argument(
        "--num-shards", type=int, default=1, help="Total number of episode shards to evaluate."
    )
    parser.add_argument(
        "--prompt-snap-grid",
        type=int,
        default=None,
        help="Optional grid size that snaps selected points to grid-cell centers before SAM.",
    )
    parser.add_argument(
        "--legacy-duplicate-protocol",
        action="store_true",
        help="Reproduce historical full-resolution selection with duplicate snapped prompts.",
    )
    parser.add_argument(
        "--overwrite-results",
        action="store_true",
        help="Replace an existing CSV and metadata pair at the selected output path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dataset_cfg = dict(cfg["dataset"])
    eval_cfg = dict(cfg["evaluation"])
    model_cfg = dict(cfg["models"])
    selection_cfg = dict(cfg.get("selection", {}))
    split_name = str(args.split or dataset_cfg["split"])
    if args.mode == "sweep" and split_name not in {"val", "validation"}:
        raise ValueError("Parameter sweeps are restricted to validation splits.")
    max_episodes = (
        args.max_episodes if args.max_episodes is not None else dataset_cfg.get("max_episodes")
    )
    device = choose_device(model_cfg.get("device", "cpu"))
    shard_index = int(args.shard_index)
    num_shards = int(args.num_shards)
    if num_shards < 1:
        raise ValueError(f"--num-shards must be >= 1, got {num_shards}")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"--shard-index must be in [0, {num_shards - 1}], got {shard_index}")

    methods = _resolve_methods(
        cfg["methods"], args.mode, args.frozen_params or selection_cfg.get("best_params_json")
    )
    if not methods:
        raise ValueError("At least one baseline method is required.")
    episodes = build_episodes_from_dataset_cfg(dataset_cfg, split_name, max_episodes=max_episodes)
    if not episodes:
        raise ValueError("No evaluation episodes were constructed.")
    all_episodes = list(episodes)
    if num_shards > 1:
        episodes = [
            episode for index, episode in enumerate(episodes) if index % num_shards == shard_index
        ]
        if not episodes:
            raise ValueError(f"No episodes assigned to shard {shard_index} of {num_shards}.")

    output_csv = Path(args.output_csv or eval_cfg["output_csv"])
    if num_shards > 1:
        output_csv = output_csv.with_name(
            f"{output_csv.stem}.shard-{shard_index:04d}-of-{num_shards:04d}{output_csv.suffix}"
        )
    metadata_path = output_csv.with_suffix(".metadata.json")
    if (output_csv.exists() or metadata_path.exists()) and not args.overwrite_results:
        raise FileExistsError(
            f"Result output already exists ({output_csv} or {metadata_path}); choose a new path or pass --overwrite-results."
        )
    if args.overwrite_results:
        output_csv.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
    overlay_dir = Path(args.overlay_dir or eval_cfg["overlay_dir"])
    overlay_limit = int(eval_cfg.get("overlay_limit", 0))
    overlay_budgets = {int(value) for value in eval_cfg.get("overlay_budgets", [])}
    point_budgets = [int(value) for value in eval_cfg["point_budgets"]]
    seeds = [int(value) for value in eval_cfg["seeds"]]
    deterministic_seeds = [int(value) for value in eval_cfg.get("deterministic_seeds", seeds)]
    prompt_snap_grid = (
        None
        if args.legacy_duplicate_protocol
        else (
            int(args.prompt_snap_grid)
            if args.prompt_snap_grid is not None
            else int(eval_cfg.get("prompt_snap_grid", 37) or 37)
        )
    )
    if not args.legacy_duplicate_protocol and prompt_snap_grid != 37:
        raise ValueError("The new evaluation protocol requires the 37x37 prompt grid.")
    selection_metric = str(args.selection_metric or selection_cfg.get("metric", "iou"))
    best_params_json = args.best_params_json or selection_cfg.get("best_params_json")
    method_seeds = {
        method.method: (seeds if method.family in STOCHASTIC_FAMILIES else deterministic_seeds)
        for method in methods
    }
    total_rows = len(episodes) * sum(
        len(point_budgets) * len(method_seeds[method.method]) for method in methods
    )

    print(f"Loading DINO on {device}...")
    dino_model = load_dino_model(model_cfg["dino_checkpoint"], device)
    cache = DinoSimilarityCache(
        cache_dir=eval_cfg["sim_map_cache_dir"],
        dino_model=dino_model,
        dino_checkpoint=model_cfg["dino_checkpoint"],
        device=device,
        image_size=int(model_cfg["image_size"]),
    )
    print(f"Precomputing or reusing {len(episodes)} cached similarity maps...")
    for index, episode in enumerate(episodes, start=1):
        cache.ensure(episode)
        if index == len(episodes) or index % 10 == 0:
            print(f"  cached {index}/{len(episodes)}")

    print(f"Loading SAM ({model_cfg['sam_model_type']}) on {device}...")
    predictor = load_sam_predictor(
        checkpoint=model_cfg["sam_checkpoint"],
        model_type=str(model_cfg.get("sam_model_type", "vit_h")),
        device=device,
    )

    writer = CsvResultWriter(output_csv)
    overlay_count = 0
    run_metadata = {
        "config": str(Path(args.config).resolve()),
        **_resolve_episode_source_metadata(dataset_cfg, split_name),
        "split": split_name,
        "max_episodes": None if max_episodes is None else int(max_episodes),
        "device": device,
        "episode_count": len(episodes),
        "expected_episode_count": len(all_episodes),
        "expected_result_rows": total_rows,
        "expected_episode_keys": [episode_identity(episode) for episode in all_episodes],
        "manifest_hash": _episode_manifest_hash(all_episodes),
        "shard_episode_keys": [episode_identity(episode) for episode in episodes],
        "methods": [method.method for method in methods],
        "point_budgets": point_budgets,
        "seeds": seeds,
        "deterministic_seeds": deterministic_seeds,
        "output_csv": str(output_csv),
        "overlay_dir": str(overlay_dir),
        "sim_map_cache_dir": str(eval_cfg["sim_map_cache_dir"]),
        "shard_index": shard_index,
        "num_shards": num_shards,
        "prompt_snap_grid": None if prompt_snap_grid is None else int(prompt_snap_grid),
        "condition": str(cfg.get("condition", cfg.get("run", {}).get("condition", "baseline"))),
        "dataset": str(dataset_cfg.get("name", dataset_cfg.get("kind", "unknown"))),
        "selection_mode": args.mode,
        "legacy_duplicate_protocol": bool(args.legacy_duplicate_protocol),
        "sweep_scope": "pooled_across_budgets" if args.mode == "sweep" else None,
        "selection_metric": selection_metric,
        "timing_boundary": "point selection plus SAM decoding; excludes image encoding and similarity-map preparation",
    }
    reserved_metadata = {
        "dataset",
        "split",
        "condition",
        "model_config_hash",
        "model_config",
        "manifest_hash",
        "expected_episode_keys",
        "expected_episode_count",
        "shard_episode_keys",
        "episode_count",
        "expected_result_rows",
        "training_seed",
        "eval_seed",
        "decoder",
        "model_hash",
    }
    write_result_metadata(
        output_csv,
        episodes=all_episodes,
        shard_episodes=episodes,
        dataset=str(dataset_cfg.get("name", dataset_cfg.get("kind", "unknown"))),
        split=split_name,
        condition=str(cfg.get("condition", cfg.get("run", {}).get("condition", "baseline"))),
        model_config=_model_config_payload(model_cfg),
        expected_result_rows=total_rows,
        decoder=str(model_cfg.get("sam_model_type", "vit_h")),
        extra={key: value for key, value in run_metadata.items() if key not in reserved_metadata},
    )

    row_index = 0
    rows: list[dict[str, object]] = []
    try:
        for episode in episodes:
            _, _, query_rgb, query_mask = load_episode_arrays(
                episode, image_size=int(model_cfg["image_size"])
            )
            sim_map = cache.load(episode)
            predictor.set_image(query_rgb, image_format="RGB")
            for method in methods:
                for n_points in point_budgets:
                    for seed in method_seeds[method.method]:
                        row_index += 1
                        rng = np.random.default_rng(seed)
                        started = time.perf_counter()
                        selection_map = (
                            sim_map
                            if args.legacy_duplicate_protocol
                            else _resize_map_nearest(sim_map, prompt_snap_grid)
                            if prompt_snap_grid
                            else sim_map
                        )
                        if (
                            method.family == "mask_aware_greedy"
                            and not args.legacy_duplicate_protocol
                        ):
                            grid_points = _mask_aware_greedy_points(
                                selection_map,
                                n_points,
                                predictor,
                                query_rgb.shape[:2],
                                int(prompt_snap_grid or 37),
                            )
                        else:
                            grid_points = np.asarray(
                                method.select_points(selection_map, n_points, rng), dtype=np.int32
                            )
                        if args.legacy_duplicate_protocol:
                            points = snap_points_to_grid(
                                grid_points, 37, int(query_rgb.shape[1]), int(query_rgb.shape[0])
                            )
                        elif prompt_snap_grid is not None:
                            grid_points = _dedupe_refill_grid(grid_points, selection_map, n_points)
                            points = _grid_coordinates_to_image(
                                grid_points,
                                int(prompt_snap_grid),
                                query_rgb.shape[1],
                                query_rgb.shape[0],
                            )
                        else:
                            points = grid_points
                        if points.shape != (n_points, 2):
                            raise ValueError(
                                f"{method.method} returned shape {points.shape}, expected {(n_points, 2)} for {episode.image_id}"
                            )
                        pred_mask = predict_mask_from_points(predictor, points)
                        elapsed_ms = float((time.perf_counter() - started) * 1000.0)
                        metrics = {
                            "iou": iou_score(pred_mask, query_mask),
                            "dice": dice_score(pred_mask, query_mask),
                            "boundary_iou": boundary_iou_score(pred_mask, query_mask),
                            "wallclock_ms": elapsed_ms,
                        }
                        row = {
                            "method": method.method,
                            "N": int(n_points),
                            "image_id": episode.image_id,
                            "episode_key": episode_identity(episode),
                            "seed": int(seed),
                            "iou": metrics["iou"],
                            "dice": metrics["dice"],
                            "boundary_iou": metrics["boundary_iou"],
                            "wallclock_ms": metrics["wallclock_ms"],
                        }
                        rows.append(row)
                        writer.write(row)
                        if overlay_count < overlay_limit and (
                            not overlay_budgets or n_points in overlay_budgets
                        ):
                            save_overlay_artifacts(
                                output_dir=overlay_dir,
                                episode=episode,
                                method=method.method,
                                n_points=n_points,
                                seed=seed,
                                query_rgb=query_rgb,
                                sim_map=sim_map,
                                pred_mask=pred_mask,
                                target_mask=query_mask,
                                points=points,
                                metrics=metrics,
                            )
                            overlay_count += 1
                        if row_index == total_rows or row_index % 25 == 0:
                            print(f"  evaluated {row_index}/{total_rows}")
    finally:
        writer.close()

    print(f"CSV written to {output_csv}")
    if overlay_limit > 0:
        print(f"Sample overlays written under {overlay_dir}")
    if args.mode == "sweep":
        if not best_params_json:
            raise ValueError(
                "Sweep mode requires selection.best_params_json in the config or --best-params-json."
            )
        summary = _summarize_sweep(rows, methods, selection_metric, split_name, len(episodes))
        write_json(best_params_json, summary)
        print(f"Frozen hyperparameters written to {best_params_json}")


def _resolve_methods(
    method_specs: list[dict[str, object]],
    mode: str,
    frozen_params_path: str | Path | None,
) -> list:
    if mode == "frozen":
        if not frozen_params_path:
            raise ValueError("Frozen mode requires --frozen-params or selection.best_params_json.")
        return _load_frozen_methods(frozen_params_path)
    if mode == "sweep":
        return _expand_search_methods(method_specs)
    return [
        build_method(str(spec["family"]), _object_mapping(spec.get("params", {})))
        for spec in method_specs
    ]


def _expand_search_methods(method_specs: list[dict[str, object]]) -> list:
    methods = []
    for spec in method_specs:
        family = str(spec["family"])
        base_params = _object_mapping(spec.get("params", {}))
        search = _object_mapping(spec.get("search", {}))
        if not search:
            methods.append(build_method(family, base_params))
            continue
        keys = sorted(search)
        values = [_as_search_values(search[key]) for key in keys]
        for combo in itertools.product(*values):
            params = dict(base_params)
            params.update({key: value for key, value in zip(keys, combo, strict=True)})
            methods.append(build_method(family, params))
    return methods


def _load_frozen_methods(path: str | Path) -> list:
    payload = load_config(path) if str(path).endswith((".yaml", ".yml")) else None
    if payload is None:
        import json

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    methods = []
    for item in payload["best_methods"]:
        methods.append(build_method(str(item["family"]), _object_mapping(item.get("params", {}))))
    return methods


def _summarize_sweep(
    rows: list[dict[str, object]],
    methods: list,
    selection_metric: str,
    split_name: str,
    episode_count: int,
) -> dict:
    if selection_metric not in {"iou", "dice", "boundary_iou"}:
        raise ValueError(f"Unsupported selection metric: {selection_metric}")
    method_lookup = {method.method: method for method in methods}
    aggregates: dict[str, list[float]] = {}
    for row in rows:
        aggregates.setdefault(str(row["method"]), []).append(float(str(row[selection_metric])))
    ranked = []
    for method_name, values in aggregates.items():
        method = method_lookup[method_name]
        mean_metric = float(sum(values) / max(len(values), 1))
        ranked.append(
            {
                "family": method.family,
                "method": method.method,
                "params": method.params,
                f"mean_{selection_metric}": mean_metric,
            }
        )
    ranked.sort(key=lambda item: (-float(item[f"mean_{selection_metric}"]), str(item["method"])))
    best_by_family: dict[str, dict] = {}
    for item in ranked:
        family = str(item["family"])
        current = best_by_family.get(family)
        if current is None or float(item[f"mean_{selection_metric}"]) > float(
            current[f"mean_{selection_metric}"]
        ):
            best_by_family[family] = item
    best_methods = [best_by_family[family] for family in sorted(best_by_family)]
    return {
        "selection_metric": selection_metric,
        "split": split_name,
        "episode_count": int(episode_count),
        "best_methods": best_methods,
        "ranked_methods": ranked,
    }


def _resolve_episode_source_metadata(
    dataset_cfg: dict[str, object], split_name: str
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "dataset_kind": None if dataset_cfg.get("kind") is None else str(dataset_cfg["kind"]),
    }
    if dataset_cfg.get("episode_manifest"):
        metadata["episode_source"] = "episode_manifest"
        metadata["episode_manifest"] = str(dataset_cfg["episode_manifest"])
        return metadata
    manifests = dataset_cfg.get("episode_manifests")
    if manifests:
        manifest_path = _object_mapping(manifests)[str(split_name)]
        metadata["episode_source"] = "episode_manifests"
        metadata["episode_manifest"] = str(manifest_path)
        return metadata
    manifest_pattern = dataset_cfg.get("episode_manifest_pattern")
    if manifest_pattern:
        metadata["episode_source"] = "episode_manifest_pattern"
        metadata["episode_manifest"] = str(manifest_pattern).format(split=split_name)
        return metadata
    metadata["episode_source"] = "fss_split"
    metadata["dataset_root"] = str(dataset_cfg["root"])
    metadata["split_file"] = str(dataset_cfg["split_file"])
    metadata["pair_schedule"] = [
        list(pair) for pair in cast(list[list[int]], dataset_cfg["pair_schedule"])
    ]
    return metadata


def _episode_manifest_hash(episodes) -> str:
    keys = [episode_identity(episode) for episode in episodes]
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def _model_config_hash(model_cfg: dict[str, object]) -> str:
    return stable_hash(_model_config_payload(model_cfg))


def _model_config_payload(model_cfg: dict[str, object]) -> dict[str, object]:
    semantic_cfg = {key: value for key, value in model_cfg.items() if key != "device"}
    for checkpoint_key in ("dino_checkpoint", "sam_checkpoint"):
        checkpoint = semantic_cfg.get(checkpoint_key)
        if isinstance(checkpoint, (str, Path)) and checkpoint:
            semantic_cfg[checkpoint_key] = file_fingerprint(checkpoint)
    return semantic_cfg


def _object_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"Expected a string-keyed mapping, got {type(value).__name__}.")
    return cast(dict[str, object], dict(value))


def _as_search_values(value: object) -> list[object]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("Each search dimension must be a list or tuple.")
    return list(value)


def _resize_map_nearest(values: np.ndarray, grid: int | None) -> np.ndarray:
    if grid is None:
        return np.asarray(values, dtype=np.float32)
    rows = np.clip(
        np.round((np.arange(grid) + 0.5) * values.shape[0] / grid - 0.5).astype(int),
        0,
        values.shape[0] - 1,
    )
    cols = np.clip(
        np.round((np.arange(grid) + 0.5) * values.shape[1] / grid - 0.5).astype(int),
        0,
        values.shape[1] - 1,
    )
    return np.asarray(values[np.ix_(rows, cols)], dtype=np.float32)


def _dedupe_refill_grid(points: np.ndarray, sim_map: np.ndarray, count: int) -> np.ndarray:
    selected: list[tuple[int, int]] = []
    for x, y in np.asarray(points).reshape(-1, 2):
        cell = (int(np.clip(x, 0, sim_map.shape[1] - 1)), int(np.clip(y, 0, sim_map.shape[0] - 1)))
        if cell not in selected:
            selected.append(cell)
        if len(selected) == count:
            return np.asarray(selected, dtype=np.int32)
    order = np.argsort(-sim_map.reshape(-1), kind="stable")
    for flat in order:
        cell = (int(flat % sim_map.shape[1]), int(flat // sim_map.shape[1]))
        if cell not in selected:
            selected.append(cell)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Cannot select {count} unique cells from grid {sim_map.shape}.")
    return np.asarray(selected, dtype=np.int32)


def _grid_coordinates_to_image(
    points: np.ndarray, grid: int, width: int, height: int
) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).copy()
    pts[:, 0] = np.clip((pts[:, 0] + 0.5) * width / grid, 0, width - 1)
    pts[:, 1] = np.clip((pts[:, 1] + 0.5) * height / grid, 0, height - 1)
    return np.rint(pts).astype(np.int32)


def _mask_aware_greedy_points(
    sim_map: np.ndarray, count: int, predictor, image_shape, grid: int
) -> np.ndarray:
    flat_order = np.argsort(-sim_map.reshape(-1), kind="stable")
    selected: list[tuple[int, int]] = []
    prefix = np.zeros((0, 2), dtype=np.int32)
    for _ in range(count):
        predicted = None if not len(prefix) else predict_mask_from_points(predictor, prefix)
        eligible = []
        for flat in flat_order:
            x, y = int(flat % sim_map.shape[1]), int(flat // sim_map.shape[1])
            if (x, y) in selected:
                continue
            if predicted is not None:
                pixel = _grid_coordinates_to_image(
                    np.asarray([[x, y]], dtype=np.int32),
                    grid,
                    image_shape[1],
                    image_shape[0],
                )[0]
                px, py = int(pixel[0]), int(pixel[1])
                if predicted[py, px]:
                    continue
            eligible.append((x, y))
            break
        if not eligible:
            fallback = next(
                (
                    int(flat)
                    for flat in flat_order
                    if (int(flat % sim_map.shape[1]), int(flat // sim_map.shape[1])) not in selected
                ),
                None,
            )
            if fallback is None:
                raise ValueError(f"Requested {count} points from {sim_map.size} grid cells.")
            eligible = [(fallback % sim_map.shape[1], fallback // sim_map.shape[1])]
        cell = eligible[0]
        selected.append(cell)
        prefix = _grid_coordinates_to_image(
            np.asarray(selected), grid, image_shape[1], image_shape[0]
        )
    return np.asarray(selected, dtype=np.int32)


if __name__ == "__main__":
    main()
