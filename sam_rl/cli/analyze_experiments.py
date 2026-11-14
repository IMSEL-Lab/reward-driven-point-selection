"""Analyze validated multi-seed experiment results with paired cluster bootstrap.

Collection manifest example::

    {
      "expected_conditions": ["B0", "B1"],
      "expected_training_seeds": [17, 29, 43, 71, 101],
      "expected_eval_seeds": [0],
      "bootstrap_resamples": 5000,
      "bootstrap_seed": 731,
      "runs": [
        {"condition":"B0", "dataset":"fss1000", "split":"test",
         "decoder":"vit_b", "protocol":"single", "training_seed":17,
         "eval_seed":0, "status":"complete", "csv":"runs/B0-17/test.csv"},
        {"condition":"B1", "dataset":"fss1000", "split":"test",
         "decoder":"vit_b", "protocol":"single", "training_seed":17,
         "eval_seed":0, "status":"failed", "reason":"OOM"}
      ]
    }

Each complete run requires the CSV and its ``.metadata.json`` sidecar from
``sam_rl.result_manifest.write_result_metadata``. A ``per_step_csv`` may be
provided, or is derived by replacing the CSV suffix with ``.per_step.csv``.
Episode metadata must include ``image_id`` using the repository's
``class_sSUPPORT_qQUERY`` convention for class/query cluster resampling, or
the collection manifest must provide ``episode_clusters`` as a mapping from
episode key to a class name (FSS) or query image ID (Kvasir).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from sam_rl.result_manifest import manifest_hash, validate_result_shards

GROUP_FIELDS = ("dataset", "split", "decoder", "protocol")
RESULT_METRICS = ("iou", "dice", "boundary_iou", "wallclock_ms")


def analyze_collection(
    collection: dict[str, Any],
    *,
    base_dir: str | Path = ".",
    bootstrap_resamples: int | None = None,
    bootstrap_seed: int | None = None,
) -> dict[str, Any]:
    """Validate a collection, summarize by trained seed, and compare conditions."""
    base = Path(base_dir)
    expected_conditions = _int_or_str_list(collection.get("expected_conditions"), "conditions")
    expected_training_seeds = _int_list(
        collection.get("expected_training_seeds"), "expected_training_seeds"
    )
    expected_eval_seeds = _int_list(
        collection.get("expected_eval_seeds", [0]), "expected_eval_seeds"
    )
    if not expected_conditions or not expected_training_seeds or not expected_eval_seeds:
        raise ValueError("Expected conditions, training seeds, and evaluation seeds are required.")
    if len(set(expected_conditions)) != len(expected_conditions):
        raise ValueError("expected_conditions contains duplicates.")
    if len(set(expected_training_seeds)) != len(expected_training_seeds):
        raise ValueError("expected_training_seeds contains duplicates.")
    if len(set(expected_eval_seeds)) != len(expected_eval_seeds):
        raise ValueError("expected_eval_seeds contains duplicates.")

    resamples = int(
        bootstrap_resamples
        if bootstrap_resamples is not None
        else collection.get("bootstrap_resamples", 5000)
    )
    seed = int(
        bootstrap_seed if bootstrap_seed is not None else collection.get("bootstrap_seed", 731)
    )
    if resamples < 1:
        raise ValueError("bootstrap_resamples must be positive.")

    runs = [_normalize_run(item, base) for item in collection.get("runs", [])]
    if not runs:
        raise ValueError("Collection has no runs.")
    run_keys = [
        tuple(run[field] for field in (*GROUP_FIELDS, "condition", "training_seed", "eval_seed"))
        for run in runs
    ]
    if len(set(run_keys)) != len(run_keys):
        raise ValueError("Collection has duplicate condition/seed run entries.")

    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        if run["condition"] not in expected_conditions:
            raise ValueError(f"Unexpected condition {run['condition']!r}.")
        if run["training_seed"] not in expected_training_seeds:
            raise ValueError(f"Unexpected training seed {run['training_seed']}.")
        if run["eval_seed"] not in expected_eval_seeds:
            raise ValueError(f"Unexpected eval seed {run['eval_seed']}.")
        groups[tuple(run[field] for field in GROUP_FIELDS)].append(run)

    all_summaries: list[dict[str, Any]] = []
    all_contrasts: list[dict[str, Any]] = []
    all_per_step: list[dict[str, Any]] = []
    failure_ledger: list[dict[str, Any]] = []
    validated_manifest_hashes: dict[str, str] = {}

    for group_key, group_runs in sorted(groups.items()):
        dataset, split, decoder, protocol = group_key
        conditions = list(expected_conditions)
        run_map = {
            (run["condition"], run["training_seed"], run["eval_seed"]): run for run in group_runs
        }
        for condition in conditions:
            for train_seed in expected_training_seeds:
                for eval_seed in expected_eval_seeds:
                    run = run_map.get((condition, train_seed, eval_seed))
                    if run is None:
                        raise ValueError(
                            f"Missing run ledger entry for {condition}/{train_seed}/{eval_seed} "
                            f"in {dataset}/{split}/{decoder}/{protocol}."
                        )
                    if run["status"] != "complete":
                        failure_ledger.append(_failure_row(run))

        condition_seed_status: dict[str, dict[int, bool]] = {
            condition: {} for condition in conditions
        }
        for condition in conditions:
            for train_seed in expected_training_seeds:
                condition_seed_status[condition][train_seed] = all(
                    run_map[(condition, train_seed, eval_seed)]["status"] == "complete"
                    for eval_seed in expected_eval_seeds
                )

        complete_runs = [
            run
            for run in group_runs
            if run["status"] == "complete"
            and condition_seed_status[run["condition"]][run["training_seed"]]
        ]
        condition_rows: dict[str, dict[int, list[dict[str, Any]]]] = {
            condition: defaultdict(list) for condition in conditions
        }
        step_rows: dict[str, dict[int, list[dict[str, Any]]]] = {
            condition: defaultdict(list) for condition in conditions
        }
        group_budget_set: set[int] = set()
        if complete_runs:
            keys, meta_by_path = _load_and_validate_metadata(complete_runs, collection)
            run_budgets = set()
            for run in complete_runs:
                meta = meta_by_path[str(run["metadata"])]
                run_budgets.add(int(run.get("budget", meta.get("budget", 0))))
            group_budget_set = run_budgets
            if 0 in run_budgets:
                raise ValueError("Each completed result run must identify its budget in metadata.")
            validated_manifest_hashes["/".join(group_key)] = manifest_hash(keys)

            # Validate each condition independently so a declared failed seed remains visible
            # while complete seeds still receive strict episode/row validation.
            for condition in conditions:
                condition_completed_seeds = [
                    train_seed
                    for train_seed in expected_training_seeds
                    if condition_seed_status[condition][train_seed]
                ]
                if not condition_completed_seeds:
                    continue
                condition_runs = [r for r in complete_runs if r["condition"] == condition]
                budgets = sorted(
                    {
                        int(r.get("budget", meta_by_path[str(r["metadata"])].get("budget", 0)))
                        for r in condition_runs
                    }
                )
                methods = set()
                for run in condition_runs:
                    with run["csv"].open("r", encoding="utf-8", newline="") as handle:
                        methods.update(str(row["method"]) for row in csv.DictReader(handle))
                if methods != {condition}:
                    raise ValueError(
                        f"CSV method values {sorted(methods)} do not match condition {condition!r}."
                    )
                rows = validate_result_shards(
                    [r["csv"] for r in condition_runs],
                    expected_keys=keys,
                    expected_methods=[condition],
                    budgets=budgets,
                    seeds=[],
                    training_seeds=condition_completed_seeds,
                    eval_seeds=expected_eval_seeds,
                    metadata_paths=[r["metadata"] for r in condition_runs],
                )
                for row in rows:
                    image_map = collection.setdefault("episode_image_ids", {}).setdefault(
                        dataset, {}
                    )
                    key = str(row.get("episode_key") or row.get("image_id", ""))
                    image_id = str(row.get("image_id", ""))
                    if key in image_map and str(image_map[key]) != image_id:
                        raise ValueError(f"Conflicting image_id values for episode {key}.")
                    image_map[key] = image_id
                    condition_rows[condition][int(row["training_seed"])].append(
                        {
                            **row,
                            "episode_key": key,
                            "training_seed": int(row["training_seed"]),
                            "eval_seed": int(row["eval_seed"]),
                            "budget": int(row["N"]),
                        }
                    )

                for run in condition_runs:
                    meta = meta_by_path[str(run["metadata"])]
                    budget = int(run.get("budget", meta.get("budget", 0)))
                    per_step_path = run["per_step_csv"] or run["csv"].with_suffix(".per_step.csv")
                    if not per_step_path.is_file():
                        raise FileNotFoundError(f"Missing per-step result CSV: {per_step_path}")
                    loaded_steps = _read_per_step_csv(
                        per_step_path,
                        expected_keys=keys,
                        training_seed=run["training_seed"],
                        eval_seed=run["eval_seed"],
                        budget=budget,
                    )
                    step_rows[condition][run["training_seed"]].extend(loaded_steps)

        for condition in conditions:
            all_budgets = sorted(
                {
                    int(row["N"])
                    for seed_rows in condition_rows[condition].values()
                    for row in seed_rows
                }
                | {
                    int(row["budget"])
                    for seed_rows in step_rows[condition].values()
                    for row in seed_rows
                }
                | group_budget_set
            )
            for budget in all_budgets:
                by_seed = {
                    train_seed: [
                        row
                        for row in condition_rows[condition].get(train_seed, [])
                        if int(row["N"]) == budget
                    ]
                    for train_seed in expected_training_seeds
                }
                completed = [s for s in expected_training_seeds if by_seed[s]]
                per_seed_means = {
                    str(s): _mean([float(row["iou"]) for row in by_seed[s]]) for s in completed
                }
                seed_values = list(per_seed_means.values())
                all_summaries.append(
                    {
                        **_group_dict(group_key),
                        "condition": condition,
                        "budget": budget,
                        "expected_training_seeds": len(expected_training_seeds),
                        "completed_training_seeds": len(completed),
                        "failed_training_seeds": [
                            s for s in expected_training_seeds if s not in completed
                        ],
                        "per_training_seed_mean_iou": per_seed_means,
                        "mean_iou": _mean(seed_values) if seed_values else None,
                        "std_iou_across_training_seeds": _sample_std(seed_values),
                    }
                )
                all_per_step.extend(
                    _summarize_steps(
                        group_key,
                        condition,
                        budget,
                        expected_training_seeds,
                        step_rows[condition],
                    )
                )

        for budget in sorted(group_budget_set):
            for left_index, left in enumerate(conditions):
                for right in conditions[left_index + 1 :]:
                    left_data = _episode_seed_means(condition_rows[left], budget)
                    right_data = _episode_seed_means(condition_rows[right], budget)
                    common = sorted(set(left_data).intersection(right_data))
                    contrast = _paired_contrast(
                        left_data,
                        right_data,
                        common,
                        group_key,
                        left,
                        right,
                        budget,
                        collection,
                        resamples,
                        seed,
                    )
                    all_contrasts.append(contrast)
                    all_per_step.extend(
                        _paired_step_contrasts(
                            group_key,
                            left,
                            right,
                            budget,
                            expected_training_seeds,
                            step_rows[left],
                            step_rows[right],
                            collection,
                            resamples,
                            seed,
                        )
                    )

    return {
        "schema_version": 1,
        "bootstrap": {"resamples": resamples, "seed": seed, "interval": "percentile_95"},
        "mean_semantics": "episodes averaged within each training seed; mean/std across training seeds",
        "failure_ledger": failure_ledger,
        "validated_manifest_hashes": validated_manifest_hashes,
        "condition_summaries": all_summaries,
        "paired_contrasts": all_contrasts,
        "per_step_summaries_and_contrasts": all_per_step,
    }


def write_analysis_outputs(result: dict[str, Any], output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _atomic_json(destination / "analysis.json", result)
    _write_dict_rows(destination / "condition_summary.csv", result["condition_summaries"])
    _write_dict_rows(destination / "paired_contrasts.csv", result["paired_contrasts"])
    _write_dict_rows(
        destination / "per_step_summary_and_contrasts.csv",
        result["per_step_summaries_and_contrasts"],
    )
    _write_dict_rows(destination / "failure_ledger.csv", result["failure_ledger"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", required=True, help="Collection JSON mapping runs to CSV/metadata paths."
    )
    parser.add_argument("--output-dir", required=True, help="Directory for JSON and CSV summaries.")
    parser.add_argument(
        "--bootstrap-resamples", type=int, help="Override manifest bootstrap count."
    )
    parser.add_argument("--bootstrap-seed", type=int, help="Override deterministic bootstrap seed.")
    args = parser.parse_args()
    manifest_path = Path(args.manifest).resolve()
    collection = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = analyze_collection(
        collection,
        base_dir=manifest_path.parent,
        bootstrap_resamples=args.bootstrap_resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    write_analysis_outputs(result, args.output_dir)
    print(
        f"Wrote {len(result['condition_summaries'])} condition summaries and "
        f"{len(result['paired_contrasts'])} paired contrasts to {Path(args.output_dir).resolve()}"
    )


def _normalize_run(item: dict[str, Any], base: Path) -> dict[str, Any]:
    required = (*GROUP_FIELDS, "condition", "training_seed", "status")
    missing = [field for field in required if field not in item]
    if missing:
        raise ValueError(f"Run entry is missing required fields {missing}.")
    status = str(item["status"]).lower()
    if status not in {"complete", "failed", "incomplete", "pending"}:
        raise ValueError(f"Invalid run status {status!r}.")
    run = dict(item)
    run["condition"] = str(item["condition"])
    run["training_seed"] = int(item["training_seed"])
    run["eval_seed"] = int(item.get("eval_seed", 0))
    run["status"] = status
    for field in GROUP_FIELDS:
        run[field] = str(item[field])
    if status == "complete":
        if not item.get("csv"):
            raise ValueError("Completed run entries must include csv.")
        run["csv"] = _resolve(base, item["csv"])
        run["metadata"] = _resolve(
            base, item.get("metadata", run["csv"].with_suffix(".metadata.json"))
        )
        step = item.get("per_step_csv")
        run["per_step_csv"] = _resolve(base, step) if step else None
    else:
        run["reason"] = str(item.get("reason", item.get("status", "run did not complete")))
        if item.get("csv"):
            run["csv"] = _resolve(base, item["csv"])
    return run


def _load_and_validate_metadata(
    runs: list[dict[str, Any]], collection: dict[str, Any]
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    docs: dict[str, dict[str, Any]] = {}
    expected_keys: list[str] | None = None
    for run in runs:
        meta_path = run["metadata"]
        if not meta_path.is_file():
            raise FileNotFoundError(f"Missing result metadata sidecar: {meta_path}")
        doc = json.loads(meta_path.read_text(encoding="utf-8"))
        required = (
            "dataset",
            "split",
            "condition",
            "manifest_hash",
            "expected_episode_keys",
            "decoder",
        )
        missing = [field for field in required if field not in doc]
        if missing:
            raise ValueError(f"Metadata {meta_path} is missing fields {missing}.")
        if (
            str(doc["dataset"]) != run["dataset"]
            or str(doc["split"]) != run["split"]
            or str(doc["condition"]) != run["condition"]
            or str(doc["decoder"]) != run["decoder"]
            or int(doc.get("training_seed", -1)) != run["training_seed"]
            or int(doc.get("eval_seed", -1)) != run["eval_seed"]
        ):
            raise ValueError(f"Metadata identity disagrees with collection entry {meta_path}.")
        keys = list(map(str, doc["expected_episode_keys"]))
        if not keys or len(set(keys)) != len(keys) or manifest_hash(keys) != doc["manifest_hash"]:
            raise ValueError(
                f"Metadata has invalid or duplicate expected episode keys: {meta_path}."
            )
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise ValueError("Completed runs have different ordered expected episode identities.")
        budgets = doc.get("budget")
        if budgets is not None:
            run["budget"] = int(budgets)
        docs[str(meta_path)] = doc
    if expected_keys is None:
        raise ValueError("No completed runs provide expected episode identities.")
    expected_override = collection.get("expected_episode_keys")
    if expected_override is not None:
        group = f"{runs[0]['dataset']}|{runs[0]['split']}"
        supplied = (
            expected_override.get(group, expected_override)
            if isinstance(expected_override, dict)
            else expected_override
        )
        if list(map(str, supplied)) != expected_keys:
            raise ValueError("Collection expected_episode_keys disagree with result metadata.")
    return expected_keys, docs


def _read_per_step_csv(
    path: Path,
    *,
    expected_keys: list[str],
    training_seed: int,
    eval_seed: int,
    budget: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for source in csv.DictReader(handle):
            key = str(source["episode_key"])
            train = int(source["training_seed"])
            eval_value = int(source["eval_seed"])
            step = int(source["step"])
            value = float(source["iou"])
            if key not in set(expected_keys) or train != training_seed or eval_value != eval_seed:
                raise ValueError(f"Unexpected per-step identity in {path}: {source}.")
            if step < 1 or step > budget or not math.isfinite(value):
                raise ValueError(f"Invalid per-step IoU or step in {path}: {source}.")
            identity = (key, step)
            if identity in seen:
                raise ValueError(f"Duplicate per-step identity {identity} in {path}.")
            seen.add(identity)
            rows.append(
                {
                    "episode_key": key,
                    "training_seed": train,
                    "eval_seed": eval_value,
                    "step": step,
                    "iou": value,
                    "budget": budget,
                }
            )
    expected = {(key, step) for key in expected_keys for step in range(1, budget + 1)}
    if seen != expected:
        raise ValueError(
            f"Incomplete per-step CSV {path}: missing {len(expected - seen)} and extra {len(seen - expected)} rows."
        )
    return rows


def _summarize_steps(
    group_key: tuple[str, str, str, str],
    condition: str,
    budget: int,
    expected_seeds: list[int],
    rows_by_seed: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for step in range(1, budget + 1):
        seed_means = {}
        for train_seed in expected_seeds:
            values = [
                float(row["iou"])
                for row in rows_by_seed.get(train_seed, [])
                if int(row["budget"]) == budget and int(row["step"]) == step
            ]
            if values:
                seed_means[str(train_seed)] = _mean(values)
        values = list(seed_means.values())
        results.append(
            {
                **_group_dict(group_key),
                "condition": condition,
                "budget": budget,
                "step": step,
                "per_training_seed_mean_iou": seed_means,
                "mean_iou": _mean(values) if values else None,
                "std_iou_across_training_seeds": _sample_std(values),
                "completed_training_seeds": len(values),
            }
        )
    return results


def _episode_seed_means(
    rows_by_seed: dict[int, list[dict[str, Any]]], budget: int
) -> dict[tuple[int, str], float]:
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for train_seed, rows in rows_by_seed.items():
        for row in rows:
            if int(row["N"]) == budget:
                grouped[(train_seed, row["episode_key"])].append(float(row["iou"]))
    return {key: _mean(values) for key, values in grouped.items()}


def _paired_contrast(
    left_data: dict[tuple[int, str], float],
    right_data: dict[tuple[int, str], float],
    common: list[tuple[int, str]],
    group_key: tuple[str, str, str, str],
    left: str,
    right: str,
    budget: int,
    collection: dict[str, Any],
    resamples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    differences = {key: right_data[key] - left_data[key] for key in common}
    clusters = _cluster_map(collection, group_key[0], common)
    by_cluster: dict[str, list[float]] = defaultdict(list)
    by_seed: dict[int, list[float]] = defaultdict(list)
    for key, delta in differences.items():
        cluster = clusters[key[1]]
        by_cluster[cluster].append(delta)
        by_seed[key[0]].append(delta)
    point = _mean(list(differences.values())) if differences else None
    low, high = _cluster_bootstrap(
        by_cluster, resamples, _derived_seed(bootstrap_seed, group_key, left, right, budget)
    )
    return {
        **_group_dict(group_key),
        "left_condition": left,
        "right_condition": right,
        "right_minus_left_iou": point,
        "std_across_training_seed_differences": _sample_std([_mean(x) for x in by_seed.values()]),
        "per_training_seed_difference": {str(k): _mean(v) for k, v in sorted(by_seed.items())},
        "n_paired_training_seeds": len(by_seed),
        "n_paired_episode_seed_rows": len(differences),
        "n_clusters": len(by_cluster),
        "cluster_bootstrap_95_ci": [low, high],
        "bootstrap_resamples": resamples,
    }


def _paired_step_contrasts(
    group_key: tuple[str, str, str, str],
    left: str,
    right: str,
    budget: int,
    expected_seeds: list[int],
    left_rows: dict[int, list[dict[str, Any]]],
    right_rows: dict[int, list[dict[str, Any]]],
    collection: dict[str, Any],
    resamples: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    left_index = _step_index(left_rows, budget)
    right_index = _step_index(right_rows, budget)
    output = []
    for step in range(1, budget + 1):
        common = sorted(set(left_index.get(step, {})).intersection(right_index.get(step, {})))
        grouped: dict[str, list[float]] = defaultdict(list)
        seed_deltas: dict[int, list[float]] = defaultdict(list)
        cluster_lookup = _cluster_map(collection, group_key[0], common)
        for key in common:
            delta = right_index[step][key] - left_index[step][key]
            grouped[cluster_lookup[key[1]]].append(delta)
            seed_deltas[key[0]].append(delta)
        low, high = _cluster_bootstrap(
            grouped,
            resamples,
            _derived_seed(bootstrap_seed, group_key, left, right, f"{budget}:{step}"),
        )
        output.append(
            {
                **_group_dict(group_key),
                "left_condition": left,
                "right_condition": right,
                "budget": budget,
                "step": step,
                "right_minus_left_iou": _mean([v for values in grouped.values() for v in values])
                if grouped
                else None,
                "per_training_seed_difference": {
                    str(k): _mean(v) for k, v in sorted(seed_deltas.items())
                },
                "n_paired_training_seeds": len(seed_deltas),
                "n_paired_episode_seed_rows": len(common),
                "n_clusters": len(grouped),
                "cluster_bootstrap_95_ci": [low, high],
                "analysis_type": "per_step_paired_contrast",
            }
        )
    return output


def _step_index(
    rows_by_seed: dict[int, list[dict[str, Any]]], budget: int
) -> dict[int, dict[tuple[int, str], float]]:
    result: dict[int, dict[tuple[int, str], float]] = defaultdict(dict)
    for seed, rows in rows_by_seed.items():
        for row in rows:
            if int(row["budget"]) == budget:
                result[int(row["step"])][(seed, str(row["episode_key"]))] = float(row["iou"])
    return result


def _cluster_map(collection: dict[str, Any], dataset: str, identities: list[Any]) -> dict[str, str]:
    configured = collection.get("episode_clusters", {})
    if (
        isinstance(configured, dict)
        and dataset in configured
        and isinstance(configured[dataset], dict)
    ):
        configured = configured[dataset]
    mapped: dict[str, str] = {}
    missing = []
    for item in identities:
        key = str(item[1] if isinstance(item, tuple) else item)
        cluster = configured.get(key) if isinstance(configured, dict) else None
        if cluster is None:
            missing.append(key)
        else:
            mapped[key] = str(cluster)
    if missing:
        if isinstance(configured, dict) and all(key in configured for key in missing):
            mapped.update({key: str(configured[key]) for key in missing})
        else:
            # Stable image_id fallback works for the repository's class_sSUPPORT_qQUERY form.
            images = collection.get("episode_image_ids", {})
            if isinstance(images, dict) and dataset in images and isinstance(images[dataset], dict):
                images = images[dataset]
            for key in missing:
                image_id = images.get(key) if isinstance(images, dict) else None
                if image_id is None:
                    raise ValueError(
                        f"Missing cluster identity for episode {key}; provide episode_clusters "
                        "or episode_image_ids in the collection manifest."
                    )
                mapped[key] = _cluster_from_image_id(str(image_id), dataset)
    return mapped


def _cluster_from_image_id(image_id: str, dataset: str) -> str:
    if "_s" not in image_id or "_q" not in image_id:
        raise ValueError(f"Cannot infer class/query cluster from image_id {image_id!r}.")
    class_name, _, suffix = image_id.partition("_s")
    _, _, query_id = suffix.rpartition("_q")
    if not class_name or not query_id:
        raise ValueError(f"Cannot infer class/query cluster from image_id {image_id!r}.")
    return f"query:{query_id}" if "kvasir" in dataset.lower() else f"class:{class_name}"


def _cluster_bootstrap(
    values_by_cluster: dict[str, list[float]], resamples: int, seed: int
) -> tuple[float | None, float | None]:
    clusters = sorted(values_by_cluster)
    if not clusters:
        return None, None
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(resamples):
        sampled = [clusters[rng.randrange(len(clusters))] for _ in clusters]
        estimates.append(_mean([_mean(values_by_cluster[cluster]) for cluster in sampled]))
    low, high = np.quantile(np.asarray(estimates, dtype=np.float64), [0.025, 0.975])
    return float(low), float(high)


def _derived_seed(seed: int, group: tuple[str, ...], left: str, right: str, suffix: Any) -> int:
    payload = "|".join([str(seed), *group, left, right, str(suffix)])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def _failure_row(run: dict[str, Any]) -> dict[str, Any]:
    return {
        **_group_dict(tuple(run[field] for field in GROUP_FIELDS)),
        "condition": run["condition"],
        "training_seed": run["training_seed"],
        "eval_seed": run["eval_seed"],
        "status": run["status"],
        "reason": run["reason"],
    }


def _group_dict(group_key: tuple[str, str, str, str]) -> dict[str, str]:
    return dict(zip(GROUP_FIELDS, group_key, strict=True))


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def _sample_std(values: list[float]) -> float | None:
    return statistics.stdev(values) if len(values) > 1 else None


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _int_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list of integers.")
    return [int(item) for item in value]


def _int_or_str_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"expected_{name} must be a list.")
    return [str(item) for item in value]


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _write_dict_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        if fields:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
