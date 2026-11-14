"""Generate and supervise the frozen chair-response experiment suite.

This module only schedules the existing BC and PPO CLIs. It does not implement
training, alter checkpoints, or decide benchmark identities.
"""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = REPO_ROOT / "configs/experiments/chair_v2.json"
DEFAULT_REGISTRY = REPO_ROOT / "artifacts/experiments/chair_v2/registry.json"
SEEDS = (17, 29, 43, 71, 101)
BC_CONDITIONS = {
    "B0": (False, False),
    "B1": (True, False),
    "B2": (False, True),
    "B3": (True, True),
}
PPO_CONDITIONS = {
    "R0": (False, "bc", False, "B2"),
    "R1": (True, "bc", False, "B2"),
    "R2": (False, "random", False, None),
    "R3": (True, "random", False, None),
    "R4": (False, "bc", True, "B2"),
    "R5": (True, "bc", True, "B2"),
    "R6": (False, "random", True, "B2"),
    "R7": (True, "random", True, "B2"),
    "R8": (True, "bc", True, "B3"),
}
CODE_FINGERPRINT_PATHS = (
    "sam_rl/cli/experiment_suite.py",
    "sam_rl/cli/common.py",
    "sam_rl/cli/train_greedy_bc.py",
    "sam_rl/cli/train_rl_finetune.py",
    "sam_rl/greedy_bc.py",
    "sam_rl/rl_policy.py",
    "sam_rl/rl_env.py",
    "sam_rl/baseline_eval.py",
    "sam_rl/sam_cache.py",
    "sam_rl/episode_manifest.py",
    "sam_rl/episode_source.py",
    "sam_rl/protocol.py",
    "sam_rl/provenance.py",
)


def canonical_json(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_identities(cfg: dict[str, Any]) -> dict[str, str]:
    paths: set[str] = set()
    dataset = cfg.get("dataset", {})
    if dataset.get("split_file"):
        paths.add(str(dataset["split_file"]))
    paths.update(str(value) for value in dataset.get("episode_manifests", {}).values())
    for value in cfg.get("experiment", {}).get("evaluation_protocol", {}).values():
        if isinstance(value, str) and value.endswith(".json"):
            paths.add(value)
    identities = {}
    for value in sorted(paths):
        path = Path(value)
        path = path if path.is_absolute() else REPO_ROOT / path
        if path.is_file():
            identities[str(path)] = sha256_file(path)
    return identities


def code_fingerprint() -> str:
    content = {
        relative: sha256_file(REPO_ROOT / relative)
        for relative in CODE_FINGERPRINT_PATHS
        if (REPO_ROOT / relative).is_file()
    }
    return hashlib.sha256(canonical_json(content)).hexdigest()


def atomic_write(path: str | Path, content: bytes) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


@contextmanager
def file_lock(path: str | Path):
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        result = yaml.safe_load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"Expected YAML mapping in {path}.")
    return result


def nested_set(payload: dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    current = payload
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    current[keys[-1]] = value


def job_id(condition: str, seed: int, *, pilot: bool = False) -> str:
    return f"{'pilot-' if pilot else ''}{condition}-s{int(seed)}"


def configured_host_layout(
    spec_path: str | Path = DEFAULT_SPEC,
) -> dict[str, dict[str, tuple[int, ...]]]:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    raw_layout = spec.get("host_layout")
    if not isinstance(raw_layout, dict) or not raw_layout:
        raise ValueError("host_layout must define at least one host.")
    layout: dict[str, dict[str, tuple[int, ...]]] = {}
    for host, values in raw_layout.items():
        if not isinstance(host, str) or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", host) is None:
            raise ValueError(
                f"Invalid host name {host!r}; use letters, numbers, dots, underscores, or hyphens."
            )
        if not isinstance(values, dict):
            raise ValueError(f"host_layout entry for {host} must be an object.")
        if not isinstance(values.get("seeds"), list) or not isinstance(values.get("gpu_ids"), list):
            raise ValueError(f"host_layout seeds and gpu_ids for {host} must be JSON lists.")
        try:
            seeds = tuple(map(int, values.get("seeds", ())))
            gpu_ids = tuple(map(int, values.get("gpu_ids", ())))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"host_layout seeds and gpu_ids for {host} must be integer lists."
            ) from exc
        layout[host] = {"seeds": seeds, "gpu_ids": gpu_ids}
    primary_seeds = tuple(map(int, spec.get("learner_seeds", SEEDS)))
    if len(primary_seeds) != len(set(primary_seeds)):
        raise ValueError("learner_seeds must not contain duplicates.")
    all_seeds = [seed for values in layout.values() for seed in values["seeds"]]
    if len(all_seeds) != len(set(all_seeds)) or set(all_seeds) != set(primary_seeds):
        raise ValueError(f"host_layout must assign each primary seed once: {primary_seeds}")
    for host, values in layout.items():
        if not values["gpu_ids"] or len(values["gpu_ids"]) != len(set(values["gpu_ids"])):
            raise ValueError(f"host_layout for {host} must contain unique GPU ids.")
        if any(gpu_id < 0 for gpu_id in values["gpu_ids"]):
            raise ValueError(f"host_layout for {host} must use nonnegative physical GPU ids.")
    return layout


def host_seeds(host: str, spec_path: str | Path = DEFAULT_SPEC) -> tuple[int, ...]:
    layout = configured_host_layout(spec_path)
    if host not in layout:
        raise KeyError(f"Unknown host {host!r}; configured hosts: {', '.join(sorted(layout))}.")
    return layout[host]["seeds"]


def build_host_assignment(spec_path: str | Path = DEFAULT_SPEC) -> dict[str, Any]:
    """Partition whole learner-seed domains across configured compute hosts."""
    layout_by_host = configured_host_layout(spec_path)
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    rows = make_job_records(seeds=tuple(map(int, spec.get("learner_seeds", SEEDS))))
    seed_owner = {seed: host for host, layout in layout_by_host.items() for seed in layout["seeds"]}
    if set(seed_owner) != set(map(int, spec.get("learner_seeds", SEEDS))):
        raise ValueError("Host layout must assign every primary seed exactly once.")
    assignments: dict[str, Any] = {}
    job_assignments: dict[str, Any] = {}
    for host, layout in layout_by_host.items():
        host_rows = [row for row in rows if seed_owner[row["seed"]] == host]
        jobs_by_gpu = {str(gpu): [] for gpu in layout["gpu_ids"]}
        for index, row in enumerate(host_rows):
            gpu = layout["gpu_ids"][index % len(layout["gpu_ids"])]
            jobs_by_gpu[str(gpu)].append(row["job_id"])
            job_assignments[row["job_id"]] = {
                "host": host,
                "gpu": gpu,
                "seed": row["seed"],
                "condition": row["condition"],
                "dependencies": row["dependencies"],
            }
        assignments[host] = {
            "gpu_ids": list(layout["gpu_ids"]),
            "seeds": list(layout["seeds"]),
            "job_count": len(host_rows),
            "config_dir": "configs/generated/chair_v2/full",
            "registry": f"artifacts/experiments/chair_v2/registry-{host.lower()}.json",
            "jobs_by_gpu": jobs_by_gpu,
            "supervisor_commands": {
                str(gpu): [
                    "python",
                    "-m",
                    "sam_rl.cli.experiment_suite",
                    "run",
                    "--registry",
                    f"artifacts/experiments/chair_v2/registry-{host.lower()}.json",
                    "--jobs",
                    *jobs_by_gpu[str(gpu)],
                    "--gpus",
                    str(gpu),
                    "--wait-dependencies",
                ]
                for gpu in layout["gpu_ids"]
            },
        }
    if set(job_assignments) != {row["job_id"] for row in rows}:
        raise ValueError("Host plan does not cover the full job matrix exactly once.")
    return {
        "version": 1,
        "suite": "chair_v2",
        "allocation_status": "proposed; recheck availability and rebalance after the pilot",
        "allocation_rule": "Keep each seed's BC and dependent PPO jobs on one host; round-robin jobs across its three listed GPUs.",
        "total_jobs": len(rows),
        "hosts": assignments,
        "jobs": job_assignments,
    }


def make_job_records(
    *,
    seeds: tuple[int, ...] = SEEDS,
    conditions: tuple[str, ...] | None = None,
    pilot: bool = False,
) -> list[dict[str, Any]]:
    selected = set(conditions or (*BC_CONDITIONS, *PPO_CONDITIONS))
    unknown = selected - set(BC_CONDITIONS) - set(PPO_CONDITIONS)
    if unknown:
        raise ValueError(f"Unknown experiment conditions: {sorted(unknown)}")
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        for condition, (mask, auxiliary) in BC_CONDITIONS.items():
            if condition not in selected:
                continue
            rows.append(
                {
                    "job_id": job_id(condition, seed, pilot=pilot),
                    "condition": condition,
                    "seed": seed,
                    "kind": "bc",
                    "use_current_mask": mask,
                    "aux_iou": auxiliary,
                    "dependencies": [],
                }
            )
        for condition, (mask, init_mode, anchor, source_bc) in PPO_CONDITIONS.items():
            if condition not in selected:
                continue
            dependencies = []
            if source_bc:
                dependencies = [job_id(source_bc, seed, pilot=pilot)]
            rows.append(
                {
                    "job_id": job_id(condition, seed, pilot=pilot),
                    "condition": condition,
                    "seed": seed,
                    "kind": "ppo",
                    "use_current_mask": mask,
                    "init_mode": init_mode,
                    "anchor": anchor,
                    "source_bc": source_bc,
                    "dependencies": dependencies,
                }
            )
    return rows


def _configured_job(
    row: dict[str, Any],
    *,
    bc_template: dict[str, Any],
    rl_template: dict[str, Any],
    spec: dict[str, Any],
    repo_root: Path,
    pilot: bool,
) -> dict[str, Any]:
    condition = str(row["condition"])
    seed = int(row["seed"])
    namespace = "pilot" if pilot else "full"
    output_root = str(Path(spec["output_root"]) / namespace)
    if row["kind"] == "bc":
        cfg = copy.deepcopy(bc_template)
        cfg["seed"] = seed
        cfg["dataset"]["split_file"] = spec["fss_split_file"]
        cfg["dataset"]["data_root"] = cfg["dataset"]["root"]
        cfg["dataset"]["episode_manifests"] = spec["fss_episode_manifests"]
        cfg["train"]["run_name"] = f"chair_v2_{namespace}_{condition}_s{seed}"
        cfg["train"]["condition_id"] = condition
        cfg["evaluation"]["method_name"] = condition
        cfg["train"]["aux_rollout_iou"]["enabled"] = bool(row["aux_iou"])
        cfg["train"]["aux_rollout_iou"]["weight"] = (
            float(spec["bc_aux_iou_weight"]) if row["aux_iou"] else 0.0
        )
        cfg["model"]["use_current_mask"] = bool(row["use_current_mask"])
        cfg["model"]["reserve_mask_channel"] = True
        cfg["models"]["precision"] = "fp32"
        cfg["models"]["decoder_policy"] = "single"
        cfg["oracle"]["precision"] = "fp32"
        cfg["oracle"]["decoder_policy"] = "single"
        cfg["oracle"]["max_budget"] = int(spec["click_budget"])
        cfg["validation"]["eval_max_episodes"] = spec["validation_episodes"]
        cfg["paths"]["run_root"] = f"{output_root}/bc"
        cfg["experiment"] = {
            "suite": "chair_v2",
            "namespace": namespace,
            "evaluation_protocol": spec["evaluation_protocol"],
            **row,
        }
        if pilot:
            cfg["dataset"]["max_train_episodes"] = int(spec["pilot_train_episodes"])
            cfg["dataset"]["max_val_episodes"] = int(spec["pilot_val_episodes"])
            cfg["train"]["epochs"] = 1
            cfg["oracle"]["max_budget"] = int(spec["click_budget"])
        return cfg

    cfg = copy.deepcopy(rl_template)
    cfg["seed"] = seed
    cfg["dataset"]["split_file"] = spec["fss_split_file"]
    cfg["dataset"]["data_root"] = cfg["dataset"]["root"]
    cfg["dataset"]["episode_manifests"] = spec["fss_episode_manifests"]
    cfg["model"]["use_current_mask"] = bool(row["use_current_mask"])
    cfg["model"]["reserve_mask_channel"] = True
    cfg["models"]["train_sam_use_bfloat16"] = False
    cfg["models"]["eval_sam_use_bfloat16"] = False
    cfg["models"]["decoder_policy"] = "single"
    cfg["models"]["precision"] = "fp32"
    cfg["rl"]["run_name"] = f"chair_v2_{namespace}_{condition}_s{seed}"
    cfg["train"]["run_name"] = cfg["rl"]["run_name"]
    cfg["rl"]["init_mode"] = row["init_mode"]
    cfg["rl"]["init_checkpoint"] = (
        _bc_checkpoint(spec, row, namespace, repo_root) if row["init_mode"] == "bc" else None
    )
    cfg["rl"]["kl_reference_mode"] = "bc" if row["anchor"] else "none"
    cfg["rl"]["kl_reference_checkpoint"] = (
        _bc_checkpoint(spec, row, namespace, repo_root) if row["anchor"] else None
    )
    cfg["rl"]["kl_beta_start"] = float(spec["kl_beta_start"]) if row["anchor"] else 0.0
    cfg["rl"]["kl_beta_end"] = float(spec["kl_beta_end"]) if row["anchor"] else 0.0
    cfg["rl"]["total_env_steps"] = int(spec["ppo_env_steps"])
    cfg["rl"]["budget"] = int(spec["click_budget"])
    cfg["validation"]["evaluate_every_updates"] = int(spec["validation_interval_updates"])
    cfg["validation"]["selection_episodes"] = int(spec["validation_episodes"])
    cfg["paths"]["run_root"] = f"{output_root}/ppo"
    cfg["experiment"] = {
        "suite": "chair_v2",
        "namespace": namespace,
        "evaluation_protocol": spec["evaluation_protocol"],
        **row,
    }
    cfg["train"]["condition_id"] = condition
    cfg["evaluation"]["method_name"] = condition
    if pilot:
        cfg["dataset"]["max_train_episodes"] = int(spec["pilot_train_episodes"])
        cfg["dataset"]["max_val_episodes"] = int(spec["pilot_val_episodes"])
        cfg["rl"]["total_env_steps"] = int(spec["pilot_env_steps"])
        cfg["validation"]["selection_episodes"] = int(spec["pilot_val_episodes"])
        cfg["validation"]["evaluate_every_updates"] = int(spec["pilot_validation_updates"])
    return cfg


def _bc_checkpoint(
    spec: dict[str, Any], row: dict[str, Any], namespace: str, repo_root: Path
) -> str | None:
    source = row.get("source_bc")
    if source is None and row["kind"] == "ppo" and row.get("init_mode") == "bc":
        source = "B2"
    if source is None:
        return None
    return str(
        Path(spec["output_root"])
        / namespace
        / "bc"
        / f"chair_v2_{namespace}_{source}_s{row['seed']}_seed{row['seed']}_N{spec['click_budget']}"
        / "best.pt"
    )


def generate_configs(
    spec_path: str | Path = DEFAULT_SPEC,
    *,
    output_dir: str | Path | None = None,
    repo_root: str | Path = REPO_ROOT,
    pilot: bool = False,
    conditions: tuple[str, ...] | None = None,
    seeds: tuple[int, ...] | None = None,
    pilot_train_episodes: int | None = None,
    pilot_val_episodes: int | None = None,
    pilot_env_steps: int | None = None,
) -> list[dict[str, Any]]:
    spec_path, repo_root = Path(spec_path), Path(repo_root).resolve()
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if seeds is None:
        seeds = tuple(map(int, spec.get("learner_seeds", SEEDS)))
    if pilot_train_episodes is not None:
        spec["pilot_train_episodes"] = int(pilot_train_episodes)
    if pilot_val_episodes is not None:
        spec["pilot_val_episodes"] = int(pilot_val_episodes)
    if pilot_env_steps is not None:
        spec["pilot_env_steps"] = int(pilot_env_steps)
    base = spec_path.parent / "templateYAMLs"
    bc_template = read_yaml(base / "bc.yaml")
    rl_template = read_yaml(base / "ppo.yaml")
    config_root = (
        Path(output_dir)
        if output_dir
        else repo_root / "configs/generated/chair_v2" / ("pilot" if pilot else "full")
    )
    config_root = config_root.resolve()
    rows = make_job_records(seeds=seeds, conditions=conditions, pilot=pilot)
    configured: list[dict[str, Any]] = []
    code_hash = code_fingerprint()
    for row in rows:
        cfg = _configured_job(
            row,
            bc_template=bc_template,
            rl_template=rl_template,
            spec=spec,
            repo_root=repo_root,
            pilot=pilot,
        )
        config_path = config_root / f"{row['job_id']}.yaml"
        cfg["experiment"]["config_path"] = str(config_path)
        rendered = yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True).encode()
        if config_path.exists() and config_path.read_bytes() != rendered:
            raise FileExistsError(f"Refusing to replace different experiment config: {config_path}")
        if not config_path.exists():
            atomic_write(config_path, rendered)
        row = {
            **row,
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "manifest_hashes": manifest_identities(cfg),
            "code_sha256": code_hash,
        }
        configured.append(row)
    return configured


def command_for(job: dict[str, Any], *, device: str) -> list[str]:
    module = "train_greedy_bc" if job["kind"] == "bc" else "train_rl_finetune"
    command = [
        sys.executable,
        "-m",
        f"sam_rl.cli.{module}",
        "--config",
        job["config_path"],
        "--mode",
        "train",
        "--device",
        device,
    ]
    if job["kind"] == "ppo":
        command.extend(["--seed", str(job["seed"])])
    return command


def load_registry(path: str | Path) -> dict[str, Any]:
    registry_path = Path(path)
    if not registry_path.exists():
        return {"version": 1, "jobs": {}}
    return json.loads(registry_path.read_text(encoding="utf-8"))


def register_jobs(path: str | Path, jobs: list[dict[str, Any]]) -> dict[str, Any]:
    registry_path = Path(path)
    with file_lock(registry_path.with_suffix(registry_path.suffix + ".lock")):
        registry = load_registry(registry_path)
        for job in jobs:
            current = registry["jobs"].get(job["job_id"])
            if current and current.get("config_sha256") != job["config_sha256"]:
                raise ValueError(
                    f"Job identity already registered with another config: {job['job_id']}"
                )
            if current:
                continue
            registry["jobs"][job["job_id"]] = {
                **job,
                "status": "planned",
                "attempts": 0,
                "history": [],
            }
        atomic_write(registry_path, canonical_json(registry))
        return registry


def update_job(path: str | Path, identifier: str, mutate) -> dict[str, Any]:
    registry_path = Path(path)
    with file_lock(registry_path.with_suffix(registry_path.suffix + ".lock")):
        registry = load_registry(registry_path)
        job = registry["jobs"][identifier]
        mutate(job)
        atomic_write(registry_path, canonical_json(registry))
        return job


def claim_job(
    path: str | Path,
    identifier: str,
    *,
    gpu: int | None,
    command: list[str],
    log_path: Path,
    started: float,
    retry: bool,
) -> bool:
    registry_path = Path(path)
    with file_lock(registry_path.with_suffix(registry_path.suffix + ".lock")):
        registry = load_registry(registry_path)
        job = registry["jobs"][identifier]
        ready, _ = dependency_state(job, registry["jobs"])
        allowed = job["status"] == "planned" or (retry and job["status"] in {"failed", "cancelled"})
        if not allowed or not ready:
            return False
        _mark_running(
            job,
            gpu=gpu,
            command=command,
            log_path=log_path,
            started=started,
            retry=retry,
        )
        atomic_write(registry_path, canonical_json(registry))
        return True


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def dependency_state(job: dict[str, Any], jobs: dict[str, Any]) -> tuple[bool, list[str]]:
    missing = []
    for dependency in job.get("dependencies", []):
        parent = jobs.get(dependency)
        checkpoint = Path(parent.get("checkpoint", "")) if parent else Path()
        if not parent or parent.get("status") != "completed" or not checkpoint.is_file():
            missing.append(dependency)
            continue
        expected_hash = parent.get("checkpoint_sha256")
        if expected_hash and sha256_file(checkpoint) != expected_hash:
            missing.append(dependency)
    return not missing, missing


def dependency_failures(job: dict[str, Any], jobs: dict[str, Any]) -> list[str]:
    failures = []
    for dependency in job.get("dependencies", []):
        parent = jobs.get(dependency)
        if not parent or parent.get("status") in {"failed", "cancelled"}:
            failures.append(dependency)
            continue
        if parent.get("status") == "completed":
            checkpoint = Path(parent.get("checkpoint", ""))
            expected_hash = parent.get("checkpoint_sha256")
            if not checkpoint.is_file() or (
                expected_hash and sha256_file(checkpoint) != expected_hash
            ):
                failures.append(dependency)
    return failures


def wait_for_dependencies(
    registry_path: str | Path,
    identifier: str,
    *,
    poll_seconds: int = 20,
    timeout_seconds: int = 86_400,
) -> None:
    """Wait for planned/running parents, failing on bad state or a fixed timeout."""
    if not 15 <= int(poll_seconds) <= 30:
        raise ValueError("Dependency polling must be between 15 and 30 seconds.")
    if timeout_seconds <= 0:
        raise ValueError("Dependency timeout must be positive.")
    started = time.monotonic()
    while True:
        registry = load_registry(registry_path)
        jobs = registry["jobs"]
        if identifier not in jobs:
            raise KeyError(f"Unknown job id: {identifier}")
        job = jobs[identifier]
        failures = dependency_failures(job, jobs)
        if failures:
            detail = ", ".join(
                f"{name}={jobs[name]['status']}" if name in jobs else f"{name}=missing"
                for name in failures
            )
            raise RuntimeError(
                f"Cannot run {identifier}; dependency failed or is missing: {detail}."
            )
        ready, missing = dependency_state(job, jobs)
        if ready:
            return
        completed_but_invalid = [
            name for name in missing if jobs[name].get("status") == "completed"
        ]
        if completed_but_invalid:
            raise RuntimeError(
                f"Completed dependency checkpoint missing or hash changed for {identifier}: "
                + ", ".join(completed_but_invalid)
            )
        remaining = timeout_seconds - (time.monotonic() - started)
        if remaining <= 0:
            raise TimeoutError(
                f"Timed out waiting for {identifier} dependencies: {', '.join(missing)}."
            )
        time.sleep(min(int(poll_seconds), remaining))


def validate_jobs(registry: dict[str, Any], *, require_dependencies: bool = False) -> list[str]:
    errors: list[str] = []
    jobs = registry["jobs"]
    current_code_hash = code_fingerprint()
    for identifier, job in jobs.items():
        config = Path(job["config_path"])
        if not config.is_file():
            errors.append(f"{identifier}: missing config {config}")
            continue
        if sha256_file(config) != job["config_sha256"]:
            errors.append(f"{identifier}: config hash mismatch")
        if job.get("code_sha256") and current_code_hash != job["code_sha256"]:
            errors.append(f"{identifier}: code hash mismatch")
        try:
            cfg = read_yaml(config)
        except (OSError, yaml.YAMLError, ValueError) as exc:
            errors.append(f"{identifier}: invalid YAML config: {exc}")
            continue
        if not isinstance(cfg.get("seed"), int):
            errors.append(f"{identifier}: config has no integer seed")
            continue
        if int(cfg["seed"]) != int(job["seed"]):
            errors.append(f"{identifier}: seed mismatch")
        for path_string, expected_hash in job.get("manifest_hashes", {}).items():
            manifest_path = Path(path_string)
            if not manifest_path.is_file():
                errors.append(f"{identifier}: missing manifest {manifest_path}")
            elif sha256_file(manifest_path) != expected_hash:
                errors.append(f"{identifier}: manifest hash mismatch {manifest_path}")
        if job["kind"] == "ppo":
            if int(cfg["model"].get("grid", 0)) != 37:
                errors.append(f"{identifier}: expected 37x37 action grid")
            if bool(cfg["model"].get("reserve_mask_channel")) is not True:
                errors.append(f"{identifier}: actor must reserve the mask channel")
            if cfg["rl"].get("init_mode") != job["init_mode"]:
                errors.append(f"{identifier}: initialization mode mismatch")
        if require_dependencies:
            ready, dependencies = dependency_state(job, jobs)
            if not ready:
                errors.append(f"{identifier}: dependencies incomplete: {', '.join(dependencies)}")
    return errors


def _artifact_record(job: dict[str, Any], cfg: dict[str, Any]) -> tuple[str, str]:
    if job["kind"] == "bc":
        run_dir = (
            _repo_path(cfg["paths"]["run_root"])
            / f"{cfg['train']['run_name']}_seed{job['seed']}_N{cfg['train']['budget']}"
        )
    else:
        suffix = f"__{cfg['train']['condition_id']}__seed{job['seed']}"
        run_dir = _repo_path(cfg["paths"]["run_root"]) / f"{cfg['train']['run_name']}{suffix}"
    checkpoint = run_dir / "best.pt"
    return str(run_dir), str(checkpoint)


def _repo_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def run_queue(
    registry_path: str | Path,
    *,
    gpus: tuple[int, ...],
    job_ids: tuple[str, ...] | None = None,
    dry_run: bool = False,
    resume_failed: bool = False,
    cpu: bool = False,
    wait_dependencies: bool = False,
    dependency_poll_seconds: int = 20,
    dependency_timeout_seconds: int = 86_400,
) -> list[dict[str, Any]]:
    if not gpus and not cpu:
        raise ValueError("Provide at least one explicit GPU id; use --cpu for CPU execution.")
    if len(gpus) > 1:
        raise ValueError(
            "Run one supervisor per GPU with disjoint --jobs lists; each supervisor owns one serial queue."
        )
    registry_path = Path(registry_path).resolve()
    registry = load_registry(registry_path)
    errors = validate_jobs(registry, require_dependencies=False)
    if errors:
        raise ValueError("Registry validation failed:\n" + "\n".join(errors))
    selected = set(job_ids or registry["jobs"].keys())
    unknown = selected - registry["jobs"].keys()
    if unknown:
        raise KeyError(f"Unknown job ids: {sorted(unknown)}")
    records = []
    for identifier in registry["jobs"]:
        if identifier not in selected:
            continue
        live = load_registry(registry_path)["jobs"]
        job = live[identifier]
        if job["status"] == "running" and job.get("pid") and not _pid_alive(job["pid"]):
            update_job(
                registry_path,
                identifier,
                partial(
                    _mark_failed,
                    code=-1,
                    finished=time.time(),
                    error="supervisor or training process exited before ledger completion",
                ),
            )
            job = load_registry(registry_path)["jobs"][identifier]
            live = load_registry(registry_path)["jobs"]
        ready, missing = dependency_state(job, live)
        failed_dependencies = dependency_failures(job, live)
        if failed_dependencies:
            detail = ", ".join(
                f"{name}={live[name]['status']}" if name in live else f"{name}=missing"
                for name in failed_dependencies
            )
            raise RuntimeError(
                f"Cannot run {identifier}; dependency failed or is missing: {detail}."
            )
        if not ready and wait_dependencies:
            wait_for_dependencies(
                registry_path,
                identifier,
                poll_seconds=dependency_poll_seconds,
                timeout_seconds=dependency_timeout_seconds,
            )
            live = load_registry(registry_path)["jobs"]
            job = live[identifier]
            ready, missing = dependency_state(job, live)
        if not ready:
            if dry_run:
                records.append(
                    {
                        "job_id": identifier,
                        "status": "blocked",
                        "dependencies": missing,
                        "command": command_for(job, device="cpu" if cpu else "cuda:0"),
                    }
                )
            else:
                records.append({"job_id": identifier, "status": "blocked", "dependencies": missing})
            continue
        if job["status"] == "completed":
            records.append({"job_id": identifier, "status": "completed", "skipped": True})
            continue
        if job["status"] in {"running", "failed", "cancelled"} and not resume_failed:
            records.append({"job_id": identifier, "status": job["status"], "skipped": True})
            continue
        gpu = None if cpu else gpus[0]
        cfg = read_yaml(Path(job["config_path"]))
        command = command_for(job, device="cpu" if cpu else "cuda:0")
        run_dir_string, checkpoint = _artifact_record(job, cfg)
        run_dir = Path(run_dir_string)
        if job["status"] in {"failed", "cancelled"} and resume_failed:
            if job.get("process_started"):
                resume_name = "last.pt" if job["kind"] == "bc" else "resume.pt"
                resume_checkpoint = run_dir / resume_name
                if not resume_checkpoint.is_file():
                    raise FileNotFoundError(
                        f"No complete resume checkpoint for {identifier}: {resume_checkpoint}"
                    )
                command.extend(
                    [
                        "--resume" if job["kind"] == "bc" else "--resume-checkpoint",
                        str(resume_checkpoint),
                    ]
                )
            elif run_dir.exists() and any(run_dir.iterdir()):
                raise FileExistsError(
                    f"Cannot restart prelaunch failure over nonempty output: {run_dir}"
                )
        elif run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")
        log_path = registry_path.parent / "logs" / f"{identifier}.log"
        if dry_run:
            records.append(
                {
                    "job_id": identifier,
                    "gpu": gpu,
                    "command": command,
                    "log": str(log_path),
                    "status": "planned",
                    "launched": True,
                }
            )
            continue
        gpu_lock = registry_path.parent / "gpu-locks" / ("cpu.lock" if cpu else f"gpu-{gpu}.lock")
        with file_lock(gpu_lock):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "" if cpu else str(gpu)
            run_dir.mkdir(parents=True, exist_ok=True)
            started = time.time()
            if not claim_job(
                registry_path,
                identifier,
                gpu=gpu,
                command=command,
                log_path=log_path,
                started=started,
                retry=resume_failed,
            ):
                records.append({"job_id": identifier, "status": "already claimed", "skipped": True})
                continue
            return_code = -1
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("ab") as log:
                    log.write(("\n[launcher] " + shlex.join(command) + "\n").encode())
                    log.flush()
                    process = subprocess.Popen(
                        command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
                    )
                    update_job(registry_path, identifier, partial(_set_pid, pid=process.pid))
                    try:
                        return_code = process.wait()
                    except KeyboardInterrupt:
                        process.terminate()
                        return_code = process.wait()
                        update_job(
                            registry_path,
                            identifier,
                            partial(
                                _mark_cancelled,
                                code=return_code,
                                finished=time.time(),
                            ),
                        )
                        raise
            except KeyboardInterrupt:
                raise
            except BaseException as error:
                update_job(
                    registry_path,
                    identifier,
                    partial(
                        _mark_failed,
                        code=return_code,
                        finished=time.time(),
                        error=str(error),
                    ),
                )
                raise
            if return_code == 0 and Path(checkpoint).is_file():
                update_job(
                    registry_path,
                    identifier,
                    partial(
                        _mark_completed,
                        run_dir=run_dir_string,
                        checkpoint=checkpoint,
                        finished=time.time(),
                    ),
                )
                records.append(
                    {"job_id": identifier, "gpu": gpu, "status": "completed", "launched": True}
                )
            else:
                update_job(
                    registry_path,
                    identifier,
                    partial(
                        _mark_failed,
                        code=return_code,
                        finished=time.time(),
                        error="missing checkpoint"
                        if return_code == 0
                        else "training process failed",
                    ),
                )
                records.append(
                    {
                        "job_id": identifier,
                        "gpu": gpu,
                        "status": "failed",
                        "return_code": return_code,
                        "launched": True,
                    }
                )
    return records


def _mark_running(job, gpu, command, log_path, started, retry):
    job["status"] = "running"
    job["gpu"] = int(gpu) if gpu is not None else None
    job["process_started"] = False
    job["command"] = command
    job["log"] = str(log_path)
    job["started_at"] = started
    job["runner"] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "device": "cpu" if gpu is None else f"cuda:{int(gpu)}",
    }
    job["attempts"] = int(job.get("attempts", 0)) + 1
    job.setdefault("history", []).append(
        {"status": "running", "started_at": started, "retry": bool(retry)}
    )


def _set_pid(job, pid):
    job["pid"] = int(pid)
    job["process_started"] = True


def _mark_failed(job, code, finished, error):
    job["status"] = "failed"
    job["return_code"] = int(code)
    job["finished_at"] = finished
    job["error"] = str(error)
    job["failure_phase"] = "training" if job.get("process_started") else "prelaunch"
    job["history"].append(
        {"status": "failed", "finished_at": finished, "return_code": int(code), "error": str(error)}
    )


def _mark_cancelled(job, code, finished):
    job["status"] = "cancelled"
    job["return_code"] = int(code)
    job["finished_at"] = finished
    job["error"] = "cancelled by supervisor interrupt"
    job["history"].append(
        {"status": "cancelled", "finished_at": finished, "return_code": int(code)}
    )


def _mark_completed(job, run_dir, checkpoint, finished):
    job["status"] = "completed"
    job["run_dir"] = str(run_dir)
    job["checkpoint"] = str(checkpoint)
    job["checkpoint_sha256"] = sha256_file(checkpoint)
    job["finished_at"] = finished
    job["history"].append(
        {
            "status": "completed",
            "finished_at": finished,
            "checkpoint_sha256": job["checkpoint_sha256"],
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("generate", "pilot", "host-plan", "validate", "deps", "dryrun", "run"),
    )
    parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    parser.add_argument("--registry")
    parser.add_argument("--config-dir")
    parser.add_argument("--host", help="Host name from host_layout in the suite spec.")
    parser.add_argument("--output", help="Output path for the host-assignment JSON report.")
    parser.add_argument("--conditions", nargs="*")
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--jobs", nargs="*")
    parser.add_argument("--gpus", nargs="*", type=int, default=[0])
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume-failed", action="store_true")
    parser.add_argument("--wait-dependencies", action="store_true")
    parser.add_argument("--dependency-poll-seconds", type=int, default=20)
    parser.add_argument("--dependency-timeout-seconds", type=int, default=86_400)
    parser.add_argument("--pilot-train-episodes", type=int)
    parser.add_argument("--pilot-val-episodes", type=int)
    parser.add_argument("--pilot-steps", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec_path = Path(args.spec)
    try:
        configured_layout = configured_host_layout(spec_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"Invalid host layout in {spec_path}: {exc}") from exc
    if args.host and args.host not in configured_layout:
        available = ", ".join(sorted(configured_layout))
        raise SystemExit(f"Unknown --host {args.host!r}; configured hosts: {available}.")
    if args.registry:
        registry_path = Path(args.registry)
    elif args.host:
        registry_path = (
            REPO_ROOT / "artifacts/experiments/chair_v2" / f"registry-{args.host.lower()}.json"
        )
    else:
        registry_path = DEFAULT_REGISTRY
    if args.command == "host-plan":
        plan = build_host_assignment(spec_path)
        if args.output:
            atomic_write(args.output, canonical_json(plan))
        print(json.dumps(plan, indent=2))
        return
    if args.command in {"generate", "pilot"}:
        pilot = args.command == "pilot"
        if pilot and args.host:
            raise SystemExit("Pilot generation uses its isolated seed-17 registry; omit --host.")
        conditions = (
            tuple(args.conditions)
            if args.conditions
            else (("B2", "B3", "R3", "R5", "R8") if pilot else None)
        )
        if args.host:
            if args.seeds is not None:
                raise SystemExit("Use either --host or --seeds for full generation, not both.")
            seeds = host_seeds(args.host, spec_path)
        elif args.seeds is not None:
            seeds = tuple(args.seeds)
        else:
            configured_seeds = tuple(
                map(
                    int,
                    json.loads(spec_path.read_text(encoding="utf-8")).get("learner_seeds", SEEDS),
                )
            )
            seeds = (configured_seeds[0],) if pilot else configured_seeds
        generated = generate_configs(
            spec_path,
            output_dir=args.config_dir,
            pilot=pilot,
            conditions=conditions,
            seeds=seeds,
            pilot_train_episodes=args.pilot_train_episodes,
            pilot_val_episodes=args.pilot_val_episodes,
            pilot_env_steps=args.pilot_steps,
        )
        target_registry = registry_path.with_name("pilot_registry.json") if pilot else registry_path
        register_jobs(target_registry, generated)
        print(
            json.dumps(
                {
                    "count": len(generated),
                    "config_dir": str(Path(generated[0]["config_path"]).parent)
                    if generated
                    else None,
                    "registry": str(target_registry),
                },
                indent=2,
            )
        )
        return
    registry = load_registry(registry_path)
    if args.command == "validate":
        errors = validate_jobs(registry)
        if errors:
            raise SystemExit("\n".join(errors))
        print(f"Validated {len(registry['jobs'])} registered jobs.")
        return
    if args.command == "deps":
        for identifier, job in registry["jobs"].items():
            ready, missing = dependency_state(job, registry["jobs"])
            print(f"{identifier}: {'ready' if ready else 'waiting for ' + ', '.join(missing)}")
        return
    result = run_queue(
        registry_path,
        gpus=tuple(args.gpus) if not args.cpu else (),
        job_ids=tuple(args.jobs) if args.jobs else None,
        dry_run=args.command == "dryrun",
        resume_failed=args.resume_failed,
        cpu=args.cpu,
        wait_dependencies=args.wait_dependencies,
        dependency_poll_seconds=args.dependency_poll_seconds,
        dependency_timeout_seconds=args.dependency_timeout_seconds,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
