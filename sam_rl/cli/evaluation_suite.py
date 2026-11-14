"""Plan and supervise fixed-protocol evaluation of chair_v2 checkpoints."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import yaml

from sam_rl.cli.experiment_suite import REPO_ROOT
from sam_rl.episode_manifest import episode_identity, load_episode_manifest
from sam_rl.evaluation_provenance import (
    evaluation_config_sha256,
    evaluation_source_fingerprint,
)
from sam_rl.mask_samples import (
    SAMPLE_COUNT as FINAL_MASK_SAMPLE_COUNT,
)
from sam_rl.mask_samples import (
    mask_sample_dir_for_csv,
    validate_final_mask_samples,
)
from sam_rl.provenance import atomic_write_json
from sam_rl.result_manifest import validate_result_shards

CONDITIONS = ("B0", "B1", "B2", "B3", "R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8")
SEEDS = (17, 29, 43, 71, 101)
EVAL_SEEDS = (0,)
BENCHMARKS = (
    ("fss1000_test450", "fss1000", "fss_test450", "fss_test450"),
    ("fss1000_canonical660", "fss1000", "fss_canonical660", "fss_canonical660"),
    ("kvasir_seg_test450", "kvasir_seg", "kvasir_test450", "kvasir_test450"),
)
DECODERS = ("vit_b", "vit_h")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluation_identities(
    conditions: tuple[str, ...] = CONDITIONS,
    seeds: tuple[int, ...] = SEEDS,
    benchmarks: tuple[tuple[str, str, str, str], ...] = BENCHMARKS,
    decoders: tuple[str, ...] = DECODERS,
) -> list[tuple[str, int, str, str]]:
    return [
        (condition, seed, dataset, decoder)
        for condition in conditions
        for seed in seeds
        for dataset, _, _, _ in benchmarks
        for decoder in decoders
    ]


def generate_suite(
    *,
    spec_path: str | Path,
    registry_path: str | Path,
    repo_root: str | Path = REPO_ROOT,
    output_root: str | Path | None = None,
    config_root: str | Path | None = None,
    conditions: tuple[str, ...] = CONDITIONS,
    seeds: tuple[int, ...] = SEEDS,
    decoders: tuple[str, ...] = DECODERS,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    source_fingerprint = evaluation_source_fingerprint(root)
    spec_file = _resolve(root, spec_path)
    registry_file = _resolve(root, registry_path)
    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    registry_jobs = registry.get("jobs", {})
    suite_output = _resolve(root, output_root or spec["output_root"])
    config_output = _resolve(root, config_root or "configs/generated/chair_v2_evaluation")
    cache_root = suite_output / "evaluation_cache"
    config_by_run: dict[tuple[str, int], tuple[dict[str, Any], Path, dict[str, Any]]] = {}
    for condition in conditions:
        if condition not in CONDITIONS:
            raise ValueError(f"Unknown chair_v2 condition {condition!r}.")
        for seed in seeds:
            train_job_id = f"{condition}-s{seed}"
            record = registry_jobs.get(train_job_id)
            if record is None:
                raise ValueError(f"Training registry is missing {train_job_id}.")
            train_config_path = _resolve(root, record["config_path"])
            train_config = _load_yaml(train_config_path)
            _validate_selected_training_protocol(train_config, spec, train_job_id)
            config_by_run[(condition, seed)] = (record, train_config_path, train_config)

    runs: list[dict[str, Any]] = []
    job_records: list[dict[str, Any]] = []
    # Emit one isolated evaluation per trained condition, pairing manifest, and SAM decoder.
    for condition, seed, benchmark_name, decoder in evaluation_identities(
        conditions, seeds, tuple(BENCHMARKS), decoders
    ):
        benchmark = next(item for item in BENCHMARKS if item[0] == benchmark_name)
        _benchmark_label, source_dataset, protocol_name, manifest_config_key = benchmark
        record, train_config_path, train_config = config_by_run[(condition, seed)]
        manifest_path = _resolve(root, spec["evaluation_protocol"][manifest_config_key])
        dataset_cfg = _evaluation_dataset_config(
            spec, train_config, source_dataset, manifest_path, root
        )
        episodes = load_episode_manifest(
            manifest_path, validate_paths=False, data_root=dataset_cfg["data_root"]
        )
        expected_count = _expected_count(spec, benchmark_name)
        if len(episodes) != expected_count:
            raise ValueError(
                f"Manifest {manifest_path} has {len(episodes)} episodes, expected {expected_count}."
            )
        keys = [episode_identity(episode) for episode in episodes]
        config = _evaluation_config(
            spec,
            train_config,
            condition=condition,
            seed=seed,
            benchmark=protocol_name,
            source_dataset=source_dataset,
            manifest_path=manifest_path,
            decoder=decoder,
            output_run_root=suite_output
            / "evaluation_runs"
            / condition
            / f"s{seed}"
            / benchmark_name
            / decoder,
            cache_root=cache_root / source_dataset / decoder,
            repo_root=root,
        )
        cfg_path = config_output / condition / f"s{seed}" / f"{benchmark_name}_{decoder}.yaml"
        _write_yaml_immutable(cfg_path, config)
        config_hash = sha256_file(cfg_path)
        semantic_config_hash = evaluation_config_sha256(_load_yaml(cfg_path))
        expected_manifest_hash = sha256_file(manifest_path)
        checkpoint = _checkpoint_for_record(record, train_config, root, condition, seed)
        checkpoint_hash = (
            sha256_file(checkpoint) if checkpoint.is_file() else record.get("checkpoint_sha256")
        )
        training_status = str(record.get("status", "planned"))
        if (
            checkpoint.is_file()
            and record.get("checkpoint_sha256")
            and checkpoint_hash != record["checkpoint_sha256"]
        ):
            training_status = "failed"
        output_csv = _expected_csv_path(config, condition, seed)
        per_step_csv = output_csv.with_suffix(".per_step.csv")
        status, reason = _existing_status(
            output_csv,
            per_step_csv,
            keys=keys,
            condition=condition,
            seed=seed,
            dataset=source_dataset,
            decoder=decoder,
            protocol=f"single|{protocol_name}",
            budget=int(spec["click_budget"]),
            training_status=training_status,
            checkpoint_exists=checkpoint.is_file(),
            checkpoint_path=checkpoint,
            mask_sample_count=FINAL_MASK_SAMPLE_COUNT,
            checkpoint_sha256=checkpoint_hash,
            evaluation_config_sha256=semantic_config_hash,
            evaluation_source_sha256=source_fingerprint["evaluation_source_sha256"],
        )
        command = _command_for(
            kind=str(record["kind"]),
            config_path=cfg_path,
            checkpoint=checkpoint,
            condition=condition,
            seed=seed,
            budget=int(spec["click_budget"]),
            repo_root=root,
        )
        run = {
            "training_job_id": record["job_id"],
            "condition": condition,
            "training_seed": seed,
            "eval_seed": 0,
            "dataset": source_dataset,
            "source_dataset": source_dataset,
            "split": "test",
            "benchmark": protocol_name,
            "decoder": decoder,
            "protocol": f"single|{protocol_name}",
            "status": status,
            "reason": reason,
            "csv": _portable(output_csv, root),
            "per_step_csv": _portable(per_step_csv, root),
            "metadata": _portable(output_csv.with_suffix(".metadata.json"), root),
            "config": _portable(cfg_path, root),
            "config_sha256": config_hash,
            "training_config": _portable(train_config_path, root),
            "training_config_sha256": record.get("config_sha256") or sha256_file(train_config_path),
            "checkpoint": _portable(checkpoint, root),
            "checkpoint_sha256": checkpoint_hash,
            "evaluation_config_sha256": semantic_config_hash,
            "evaluation_source_sha256": source_fingerprint["evaluation_source_sha256"],
            "manifest": _portable(manifest_path, root),
            "manifest_sha256": expected_manifest_hash,
            "manifest_hash": _identity_hash(keys),
            "expected_episode_count": len(keys),
            "expected_result_rows": len(keys),
            "mask_sample_count": FINAL_MASK_SAMPLE_COUNT,
            "mask_sample_manifest": _portable(
                mask_sample_dir_for_csv(output_csv) / "sample_manifest.json", root
            ),
            "expected_episode_keys": keys,
            "checkpoint_selection": "fss_val357_vit_b",
            "method": condition,
            "budget": int(spec["click_budget"]),
            "command": command,
            "cwd": ".",
        }
        runs.append(run)
        job_records.append(
            {**run, "job_id": evaluation_job_id(condition, seed, benchmark_name, decoder)}
        )

    _assert_output_isolation(job_records, config_by_run, root)
    collection = {
        "expected_conditions": list(conditions),
        "expected_training_seeds": list(map(int, seeds)),
        "expected_eval_seeds": [0],
        "bootstrap_resamples": 5000,
        "bootstrap_seed": 731,
        "evaluation_source_sha256": source_fingerprint["evaluation_source_sha256"],
        "runs": runs,
    }
    plan = {
        "version": 1,
        "suite": "chair_v2_evaluation",
        "checkpoint_selection": "fss_val357_vit_b_selected_best_checkpoint",
        "kvasir_adaptation": False,
        "repo_root": _portable(root, root),
        "registry_path": _portable(registry_file, root),
        **source_fingerprint,
        "jobs": job_records,
        "collection": collection,
    }
    plan_path = suite_output / "evaluation_plan.json"
    collection_path = suite_output / "collection.json"
    atomic_write_json(plan_path, plan)
    atomic_write_json(collection_path, collection)
    plan["plan_path"] = _portable(plan_path, root)
    plan["collection_path"] = _portable(collection_path, root)
    return plan


def run_suite(
    plan_path: str | Path,
    *,
    repo_root: str | Path = REPO_ROOT,
    job_ids: tuple[str, ...] | None = None,
    device: str = "cuda:0",
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    target = _resolve(root, plan_path)
    plan = json.loads(target.read_text(encoding="utf-8"))
    selected = set(job_ids or [job["job_id"] for job in plan["jobs"]])
    unknown = selected - {job["job_id"] for job in plan["jobs"]}
    if unknown:
        raise KeyError(f"Unknown evaluation jobs: {sorted(unknown)}")
    for identifier in sorted(selected):
        _refresh_job_from_registry(target, identifier, root)
        job = _claim_job(target, identifier)
        if job is None:
            continue
        dispatch_error = _dispatch_integrity_error(job, target, root)
        if dispatch_error is not None:
            _finish_job(target, identifier, status="failed", reason=dispatch_error)
            continue
        command = list(job["command"])
        checkpoint = _resolve(root, job["checkpoint"])
        if "--device" not in command:
            command.extend(["--device", device])
        try:
            result = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)
        except OSError as exc:
            _finish_job(
                target,
                identifier,
                status="failed",
                reason=f"Could not start evaluation command: {exc}",
            )
            continue
        if result.returncode != 0:
            _finish_job(
                target,
                identifier,
                status="failed",
                reason=f"Evaluation CLI exited with code {result.returncode}.",
                returncode=int(result.returncode),
                stdout_tail=result.stdout[-4000:],
                stderr_tail=result.stderr[-4000:],
            )
        else:
            status, reason = _existing_status(
                _resolve(root, job["csv"]),
                _resolve(root, job["per_step_csv"]),
                keys=job["expected_episode_keys"],
                condition=job["condition"],
                seed=int(job["training_seed"]),
                dataset=job["dataset"],
                decoder=job["decoder"],
                protocol=job["protocol"],
                budget=int(job["budget"]),
                training_status="completed",
                checkpoint_exists=True,
                checkpoint_path=checkpoint,
                mask_sample_count=int(job.get("mask_sample_count", FINAL_MASK_SAMPLE_COUNT)),
                checkpoint_sha256=job["checkpoint_sha256"],
                evaluation_config_sha256=job["evaluation_config_sha256"],
                evaluation_source_sha256=job["evaluation_source_sha256"],
            )
            _finish_job(
                target,
                identifier,
                status=status,
                reason=reason,
                returncode=0,
                stdout_tail=result.stdout[-4000:],
                stderr_tail=result.stderr[-4000:],
            )
    with _plan_lock(target):
        plan = json.loads(target.read_text(encoding="utf-8"))
    return plan


def _plan_lock(plan_path):
    lock_path = Path(plan_path).with_suffix(Path(plan_path).suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def _claim_job(plan_path, identifier):
    with _plan_lock(plan_path):
        try:
            plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
            job = next(job for job in plan["jobs"] if job["job_id"] == identifier)
        except (OSError, StopIteration, json.JSONDecodeError):
            return None
        if job["status"] in {"complete", "failed"}:
            return None
        if job["status"] == "running":
            runner_pid = int(job.get("runner_pid", -1))
            try:
                os.kill(runner_pid, 0)
            except (ProcessLookupError, ValueError):
                job["status"] = "pending"
                job.pop("runner_pid", None)
            except PermissionError:
                return None
            else:
                return None
        job["status"] = "running"
        job["runner_pid"] = os.getpid()
        job["reason"] = None
        _sync_collection(plan)
        atomic_write_json(plan_path, plan)
        atomic_write_json(Path(plan_path).parent / "collection.json", plan["collection"])
        return job


def _refresh_job_from_registry(plan_path, identifier, root):
    with _plan_lock(plan_path):
        plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
        job = next(job for job in plan["jobs"] if job["job_id"] == identifier)
        registry = json.loads(_resolve(root, plan["registry_path"]).read_text(encoding="utf-8"))
        record = registry.get("jobs", {}).get(job["training_job_id"])
        if record and record.get("status") == "completed":
            checkpoint = _checkpoint_for_record_from_record(record, root)
            if checkpoint.is_file():
                checkpoint_hash = sha256_file(checkpoint)
                recorded_hash = record.get("checkpoint_sha256")
                if recorded_hash and recorded_hash != checkpoint_hash:
                    job["status"] = "failed"
                    job["reason"] = (
                        "Training registry checkpoint hash does not match its checkpoint file."
                    )
                else:
                    job["checkpoint"] = _portable(checkpoint, root)
                    job["checkpoint_sha256"] = checkpoint_hash
                    job["training_config_sha256"] = record.get(
                        "config_sha256", job["training_config_sha256"]
                    )
                    job["status"] = "pending" if job["status"] == "pending" else job["status"]
                    command = list(job["command"])
                    if "--checkpoint" in command:
                        command[command.index("--checkpoint") + 1] = _portable(checkpoint, root)
                    job["command"] = command
        _sync_collection(plan)
        atomic_write_json(plan_path, plan)
        atomic_write_json(Path(plan_path).parent / "collection.json", plan["collection"])


def _checkpoint_for_record_from_record(record, root):
    checkpoint = record.get("checkpoint")
    if not checkpoint:
        return Path()
    return _resolve(root, checkpoint)


def _finish_job(plan_path, identifier, *, status, reason, **details):
    with _plan_lock(plan_path):
        plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
        job = next(job for job in plan["jobs"] if job["job_id"] == identifier)
        job["status"] = status
        job["reason"] = reason
        job.pop("runner_pid", None)
        job.update(details)
        _sync_collection(plan)
        atomic_write_json(plan_path, plan)
        atomic_write_json(Path(plan_path).parent / "collection.json", plan["collection"])


def _dispatch_integrity_error(job, plan_path, root):
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    registry_path = _resolve(root, plan["registry_path"])
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    train_record = registry.get("jobs", {}).get(job["training_job_id"])
    if not train_record or train_record.get("status") != "completed":
        return "Selected training job is not recorded complete in the current registry."
    checkpoint = _resolve(root, job["checkpoint"])
    if not checkpoint.is_file():
        return "Selected training checkpoint is not available."
    if checkpoint.name != "best.pt":
        return "Evaluation command does not point at the validation-selected best checkpoint."
    if train_record.get("checkpoint"):
        registry_checkpoint = _resolve(root, train_record["checkpoint"])
        if registry_checkpoint.resolve() != checkpoint.resolve():
            return "Registry selected checkpoint path differs from the evaluation plan."
    if not job.get("checkpoint_sha256") or sha256_file(checkpoint) != job["checkpoint_sha256"]:
        return "Selected checkpoint hash changed after evaluation planning."
    source_sha256 = evaluation_source_fingerprint(root)["evaluation_source_sha256"]
    if source_sha256 != job.get("evaluation_source_sha256"):
        return "Evaluation source fingerprint changed after planning."
    registry_checkpoint_hash = train_record.get("checkpoint_sha256")
    if registry_checkpoint_hash and registry_checkpoint_hash != job["checkpoint_sha256"]:
        return "Registry checkpoint hash disagrees with the planned checkpoint."
    config_path = _resolve(root, job["config"])
    if not config_path.is_file() or sha256_file(config_path) != job["config_sha256"]:
        return "Evaluation configuration is missing or changed after planning."
    if evaluation_config_sha256(_load_yaml(config_path)) != job.get("evaluation_config_sha256"):
        return "Semantic evaluation configuration hash changed after planning."
    training_config = _resolve(root, job["training_config"])
    if (
        not training_config.is_file()
        or sha256_file(training_config) != job["training_config_sha256"]
    ):
        return "Training configuration is missing or changed after planning."
    manifest = _resolve(root, job["manifest"])
    if not manifest.is_file() or sha256_file(manifest) != job["manifest_sha256"]:
        return "Evaluation episode manifest is missing or changed after planning."
    if str(train_record.get("config_sha256", "")) not in {
        "",
        str(job["training_config_sha256"]),
    }:
        return "Training registry configuration hash disagrees with the planned config."
    return None


def evaluation_job_id(condition: str, seed: int, dataset: str, decoder: str) -> str:
    return f"{condition}-s{int(seed)}-{dataset}-{decoder}"


def _evaluation_dataset_config(spec, train_config, source_dataset, manifest_path, repo_root):
    cfg = dict(train_config["dataset"])
    if source_dataset == "kvasir_seg":
        data_root = _portable(_resolve(repo_root, spec["kvasir"]["root"]), repo_root)
        split_file = spec["kvasir"]["split_file"]
    else:
        data_root = str(cfg.get("data_root", cfg["root"]))
        split_file = spec["fss_split_file"]
    cfg["root"] = data_root
    cfg["data_root"] = data_root
    cfg["split_file"] = split_file
    cfg["name"] = source_dataset
    cfg["dataset"] = source_dataset
    cfg["episode_manifests"] = {"test": _portable(manifest_path, repo_root)}
    cfg["train_split"] = "train"
    cfg["val_split"] = "val"
    cfg["test_split"] = "test"
    cfg["max_test_episodes"] = None
    return cfg


def _evaluation_config(
    spec,
    train_config,
    *,
    condition,
    seed,
    benchmark,
    source_dataset,
    manifest_path,
    decoder,
    output_run_root,
    cache_root,
    repo_root,
):
    cfg = json.loads(json.dumps(train_config))
    cfg["dataset"] = _evaluation_dataset_config(
        spec, train_config, source_dataset, manifest_path, repo_root
    )
    cfg["seed"] = int(seed)
    cfg["condition"] = str(condition)
    cfg["experiment"] = {
        **dict(cfg.get("experiment", {})),
        "suite": "chair_v2_evaluation",
        "condition": condition,
        "seed": int(seed),
        "benchmark": benchmark,
        "checkpoint_selection": "fss_val357_vit_b",
        "kvasir_adaptation": False,
        "evaluation_only": True,
    }
    cfg["evaluation"]["method_name"] = str(condition)
    cfg["evaluation"]["eval_seed"] = 0
    cfg["evaluation"]["overlay_limit"] = 0
    cfg["evaluation"]["write_contact_sheet"] = False
    cfg["evaluation"]["final_mask_sample_count"] = FINAL_MASK_SAMPLE_COUNT
    cfg["models"]["eval_sam_checkpoint"] = _sam_checkpoint(spec, train_config, decoder, repo_root)
    cfg["models"]["eval_sam_model_type"] = str(decoder)
    cfg["models"]["eval_precision"] = "fp32"
    cfg["models"]["precision"] = "fp32"
    cfg["models"]["precision_policy"] = "fp32"
    cfg["models"]["dino_query_cache_dir"] = _portable(cache_root / "dino_query", repo_root)
    cfg["models"]["sim_map_cache_dir"] = _portable(cache_root / "similarity", repo_root)
    if "eval_sam_embedding_cache_dir" in cfg["models"]:
        cfg["models"]["eval_sam_embedding_cache_dir"] = _portable(
            cache_root / "sam_embeddings", repo_root
        )
    if "sam_embedding_cache_dir" in cfg["models"]:
        cfg["models"]["sam_embedding_cache_dir"] = _portable(
            cache_root / "sam_embeddings", repo_root
        )
    cfg["protocol"] = {
        **dict(cfg.get("protocol", {})),
        "decoder_policy": "single",
        "precision": "fp32",
    }
    cfg["models"]["decoder_policy"] = "single"
    cfg["paths"]["run_root"] = _portable(output_run_root, repo_root)
    cfg.setdefault("train", {})["budget"] = int(spec["click_budget"])
    if str(source_dataset) == "kvasir_seg":
        cfg["dataset"]["max_train_episodes"] = 0
        cfg["dataset"]["max_val_episodes"] = 0
        cfg["experiment"]["adaptation"] = False
    if "train" in cfg and "run_name" in cfg["train"]:
        cfg["train"]["run_name"] = f"chair_v2_eval_{condition}"
        cfg["train"]["condition_id"] = condition
    if "rl" in cfg:
        cfg["rl"]["budget"] = int(spec["click_budget"])
        cfg["rl"]["run_name"] = f"chair_v2_eval_{condition}"
        cfg["train"]["condition_id"] = condition
    return cfg


def _sam_checkpoint(spec, train_config, decoder, repo_root):
    explicit = spec.get("sam_checkpoints", {}).get(decoder)
    if explicit:
        return str(explicit)
    models = train_config["models"]
    if decoder == str(models.get("eval_sam_model_type", models.get("sam_model_type", "vit_b"))):
        return models.get("eval_sam_checkpoint", models.get("sam_checkpoint"))
    filename = "sam_vit_b_01ec64.pth" if decoder == "vit_b" else "sam_vit_h_4b8939.pth"
    current = Path(
        str(models.get("eval_sam_checkpoint", models.get("sam_checkpoint", "artifacts/weights")))
    )
    candidates = [current.parent / filename, Path("artifacts/weights") / filename]
    for candidate in candidates:
        resolved = _resolve(repo_root, candidate)
        if resolved.is_file():
            return _portable(resolved, repo_root)
    return str(candidates[-1])


def _checkpoint_for_record(record, train_config, repo_root, condition, seed):
    if record.get("checkpoint"):
        return _resolve(repo_root, record["checkpoint"])
    if record["kind"] == "bc":
        run_root = _resolve(repo_root, train_config["paths"]["run_root"])
        run_name = train_config["train"]["run_name"]
        budget = int(train_config["train"]["budget"])
        return run_root / f"{run_name}_seed{seed}_N{budget}" / "best.pt"
    run_root = _resolve(repo_root, train_config["paths"]["run_root"])
    run_name = train_config["train"]["run_name"]
    return run_root / f"{run_name}__{condition}__seed{seed}" / "best.pt"


def _validate_selected_training_protocol(cfg, spec, job_id):
    expected = int(spec["validation_episodes"])
    if "rl" in cfg:
        selected = int(cfg.get("validation", {}).get("selection_episodes", 0))
        training_decoder = str(cfg.get("models", {}).get("train_sam_model_type", ""))
    else:
        selected = int(cfg.get("validation", {}).get("eval_max_episodes", 0))
        training_decoder = str(cfg.get("models", {}).get("oracle_sam_model_type", ""))
    if selected != expected or expected != 357:
        raise ValueError(f"{job_id} does not select on the frozen 357-episode validation set.")
    if training_decoder != "vit_b":
        raise ValueError(f"{job_id} must select its checkpoint with SAM ViT-B.")
    policy = str(
        cfg.get("protocol", {}).get(
            "decoder_policy", cfg.get("models", {}).get("decoder_policy", "")
        )
    )
    if policy != "single":
        raise ValueError(f"{job_id} was not trained under the single-mask decoder policy.")


def _command_for(kind, config_path, checkpoint, condition, seed, budget, repo_root):
    module = "train_greedy_bc" if kind == "bc" else "train_rl_finetune"
    command = [
        "uv",
        "run",
        "python",
        "-m",
        f"sam_rl.cli.{module}",
        "--config",
        _portable(config_path, repo_root),
        "--mode",
        "eval",
        "--split",
        "test",
        "--checkpoint",
        _portable(checkpoint, repo_root),
    ]
    if kind == "bc":
        command.extend(["--budget", str(budget)])
    else:
        command.extend(["--seed", str(seed)])
    return command


def _expected_csv_path(cfg, condition, seed):
    root = Path(cfg["paths"]["run_root"])
    if "rl" in cfg:
        return root / f"{cfg['train']['run_name']}__{condition}__seed{seed}" / "test_results.csv"
    return (
        root
        / f"{cfg['train']['run_name']}_seed{seed}_N{cfg['train']['budget']}"
        / "test_results.csv"
    )


def _existing_status(
    csv_path,
    per_step_path,
    *,
    keys,
    condition,
    seed,
    dataset,
    decoder,
    protocol,
    budget,
    training_status,
    checkpoint_exists,
    checkpoint_path,
    mask_sample_count,
    checkpoint_sha256,
    evaluation_config_sha256,
    evaluation_source_sha256,
):
    if training_status in {"failed", "cancelled"}:
        return "failed", f"Training job is {training_status}."
    if training_status != "completed":
        return "pending", f"Training job is {training_status}; waiting for selected checkpoint."
    if not checkpoint_exists:
        return "failed", "Training job completed without its selected checkpoint."
    if Path(checkpoint_path).name != "best.pt":
        return "failed", "Evaluation must use the validation-selected best checkpoint."
    found_output = (
        csv_path.exists()
        or csv_path.with_suffix(".metadata.json").exists()
        or per_step_path.exists()
        or mask_sample_dir_for_csv(csv_path).exists()
    )
    if (
        not csv_path.is_file()
        or not csv_path.with_suffix(".metadata.json").is_file()
        or not per_step_path.is_file()
    ):
        if found_output:
            return "failed", "Evaluation output is partial; refusing to overwrite it."
        return "pending", "Evaluation CSV, metadata, or per-step CSV is not complete."
    try:
        rows = validate_result_shards(
            [csv_path],
            expected_keys=keys,
            expected_methods=[condition],
            budgets=[budget],
            seeds=[],
            training_seeds=[seed],
            eval_seeds=[0],
        )
        if len(rows) != len(keys):
            raise ValueError("Unexpected aggregate row count.")
        seen = set()
        with per_step_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                key, train_seed, eval_seed, step = (
                    str(row["episode_key"]),
                    int(row["training_seed"]),
                    int(row["eval_seed"]),
                    int(row["step"]),
                )
                value = float(row["iou"])
                identity = (key, train_seed, eval_seed, step)
                if (
                    key not in set(keys)
                    or train_seed != seed
                    or eval_seed != 0
                    or step not in range(1, budget + 1)
                ):
                    raise ValueError(f"Unexpected per-step row {identity}.")
                if not math.isfinite(value) or identity in seen:
                    raise ValueError(f"Invalid or duplicate per-step row {identity}.")
                seen.add(identity)
        expected_steps = {(key, seed, 0, step) for key in keys for step in range(1, budget + 1)}
        if seen != expected_steps:
            raise ValueError(
                f"Incomplete per-step results: {len(expected_steps - seen)} missing rows."
            )
        metadata = json.loads(csv_path.with_suffix(".metadata.json").read_text(encoding="utf-8"))
        if metadata.get("dataset") != dataset or metadata.get("decoder") != decoder:
            raise ValueError("Result metadata dataset/decoder does not match requested evaluation.")
        if int(metadata.get("budget", -1)) != budget:
            raise ValueError("Result metadata budget does not match requested evaluation.")
        if (
            metadata.get("condition") != condition
            or int(metadata.get("training_seed", -1)) != seed
            or int(metadata.get("eval_seed", -1)) != 0
            or metadata.get("protocol") != protocol
            or not metadata.get("model_hash")
            or metadata.get("checkpoint_sha256") != checkpoint_sha256
            or metadata.get("evaluation_config_sha256") != evaluation_config_sha256
            or metadata.get("evaluation_source_sha256") != evaluation_source_sha256
        ):
            raise ValueError(
                "Result metadata condition, seed, protocol, or model hash is incompatible."
            )
        if metadata.get("expected_episode_keys") != list(keys):
            raise ValueError(
                "Result metadata episode identities differ from the frozen benchmark manifest."
            )
        validate_final_mask_samples(
            mask_sample_dir_for_csv(csv_path) / "sample_manifest.json",
            metadata_path=csv_path.with_suffix(".metadata.json"),
            expected_episode_keys=keys,
            sample_count=mask_sample_count,
        )
    except (ValueError, KeyError, OSError, json.JSONDecodeError) as exc:
        return "failed", f"Existing evaluation output failed strict validation: {exc}"
    return "complete", None


def _expected_count(spec, benchmark_name):
    values = {
        "fss1000_test450": int(spec["test_episodes"]),
        "fss1000_canonical660": int(spec["canonical_test_episodes"]),
        "kvasir_seg_test450": int(spec["kvasir"]["test_episodes"]),
    }
    return values[benchmark_name]


def _identity_hash(keys):
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def _assert_output_isolation(jobs, configs, repo_root):
    paths = [str(job["csv"]) for job in jobs]
    if len(paths) != len(set(paths)):
        raise ValueError("Evaluation jobs do not have unique CSV output paths.")
    for (condition, seed), (_, _train_config_path, train_config) in configs.items():
        run_root = Path(train_config["paths"]["run_root"])
        if not run_root.is_absolute():
            run_root = repo_root / run_root
        for job in jobs:
            if job["condition"] == condition and int(job["training_seed"]) == seed:
                output = _resolve(repo_root, job["csv"])
                if run_root in output.parents:
                    raise ValueError(
                        f"Evaluation output overlaps training output for {condition}/{seed}."
                    )


def _sync_collection(plan):
    plan["collection"]["runs"] = [
        {
            key: job[key]
            for key in (
                "condition",
                "training_seed",
                "eval_seed",
                "dataset",
                "split",
                "decoder",
                "protocol",
                "expected_episode_count",
                "expected_result_rows",
                "status",
                "reason",
                "csv",
                "per_step_csv",
                "metadata",
                "mask_sample_count",
                "mask_sample_manifest",
                "checkpoint_sha256",
                "evaluation_config_sha256",
                "evaluation_source_sha256",
                "budget",
            )
            if key in job
        }
        | {"status": "pending" if job["status"] == "running" else job["status"]}
        for job in plan["jobs"]
    ]


def _load_yaml(path):
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {path}.")
    return payload


def _write_yaml_immutable(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(payload, sort_keys=False).encode("utf-8")
    if path.exists() and path.read_bytes() != rendered:
        raise FileExistsError(f"Refusing to replace a different evaluation config: {path}")
    if not path.exists():
        path.write_bytes(rendered)


def _resolve(root, path):
    value = Path(path)
    return value if value.is_absolute() else root / value


def _portable(path, root):
    value = Path(path)
    if not value.is_absolute():
        value = _resolve(root, value)
    try:
        return value.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(value)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", default="configs/experiments/chair_v2.json")
    parser.add_argument("--registry", default="artifacts/experiments/chair_v2/registry.json")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--config-root", default=None)
    parser.add_argument(
        "--run-plan", default=None, help="Execute pending jobs from a generated plan."
    )
    parser.add_argument("--job-id", action="append", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.run_plan:
        result = run_suite(
            args.run_plan,
            repo_root=args.repo_root,
            job_ids=tuple(args.job_id or ()),
            device=args.device,
        )
    else:
        result = generate_suite(
            spec_path=args.spec,
            registry_path=args.registry,
            repo_root=args.repo_root,
            output_root=args.output_root,
            config_root=args.config_root,
            seeds=tuple(args.seeds),
        )
    print(
        json.dumps(
            {
                "jobs": len(result["jobs"]),
                "plan": result.get("plan_path"),
                "collection": result.get("collection_path"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
