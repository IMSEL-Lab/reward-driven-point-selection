"""Finish one GPU's training queue, then evaluate only its selected runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from sam_rl.cli.evaluation_suite import BENCHMARKS, DECODERS, evaluation_job_id
from sam_rl.cli.experiment_suite import DEFAULT_REGISTRY, REPO_ROOT, load_registry, run_queue


def evaluation_ids_for_training_jobs(
    training_job_ids: tuple[str, ...], evaluation_plan: dict[str, Any]
) -> list[str]:
    """Return and validate the fixed six evaluation identities for each run."""
    planned = {job["job_id"]: job for job in evaluation_plan.get("jobs", [])}
    expected_pairs = {
        (benchmark_name, decoder)
        for benchmark_name, _source_dataset, _protocol, _manifest_key in BENCHMARKS
        for decoder in DECODERS
    }
    display_name_by_protocol = {
        protocol: benchmark_name
        for benchmark_name, _source_dataset, protocol, _manifest_key in BENCHMARKS
    }
    selected: list[str] = []
    for training_id in training_job_ids:
        matches = [job for job in planned.values() if job.get("training_job_id") == training_id]
        if not matches:
            raise ValueError(f"Evaluation plan has no jobs for training job {training_id}.")
        actual_pairs = {
            (display_name_by_protocol.get(str(job.get("benchmark"))), job.get("decoder"))
            for job in matches
        }
        if len(matches) != 6 or actual_pairs != expected_pairs:
            raise ValueError(
                f"Evaluation plan for {training_id} must contain exactly 3 benchmarks x 2 decoders."
            )
        for benchmark_name, _source_dataset, _protocol, _manifest_key in BENCHMARKS:
            for decoder in DECODERS:
                identifier = evaluation_job_id(
                    str(matches[0]["condition"]),
                    int(matches[0]["training_seed"]),
                    benchmark_name,
                    decoder,
                )
                job = planned.get(identifier)
                if job is None or job.get("training_job_id") != training_id:
                    raise ValueError(f"Evaluation plan is missing the expected job {identifier}.")
                if job.get("status") not in {"pending", "complete"}:
                    raise RuntimeError(
                        f"Evaluation job {identifier} has status {job.get('status')!r}; refusing to reset it."
                    )
                selected.append(identifier)
    if len(selected) != 6 * len(training_job_ids) or len(selected) != len(set(selected)):
        raise ValueError("Selected evaluation job IDs are not a unique six-per-training mapping.")
    return selected


def run_campaign(
    *,
    registry_path: str | Path,
    evaluation_plan_path: str | Path,
    training_job_ids: tuple[str, ...],
    physical_gpu: int,
    repo_root: str | Path = REPO_ROOT,
    log_dir: str | Path | None = None,
    dependency_poll_seconds: int = 20,
    dependency_timeout_seconds: int = 86_400,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    registry_file = resolve_path(root, registry_path)
    plan_file = resolve_path(root, evaluation_plan_path)
    if physical_gpu < 0:
        raise ValueError("physical_gpu must be a nonnegative device ID.")
    if not training_job_ids or len(training_job_ids) != len(set(training_job_ids)):
        raise ValueError("Pass a nonempty list of unique training job IDs.")
    registry = load_registry(registry_file)
    plan = json.loads(plan_file.read_text(encoding="utf-8"))
    plan_registry = resolve_path(root, plan["registry_path"])
    if plan_registry.resolve() != registry_file.resolve():
        raise ValueError("Evaluation plan points at a different host-local training registry.")

    for identifier in training_job_ids:
        record = registry.get("jobs", {}).get(identifier)
        if record is None:
            raise KeyError(f"Training registry is missing selected job {identifier}.")
        if record.get("status") in {"failed", "cancelled"}:
            raise RuntimeError(
                f"Training job {identifier} is {record['status']}; failed states are not reset."
            )
        if record.get("status") == "running":
            raise RuntimeError(f"Training job {identifier} is already running in another queue.")
        if record.get("status") not in {"planned", "completed"}:
            raise RuntimeError(
                f"Training job {identifier} has unexpected status {record.get('status')!r}."
            )
    expected_eval_ids = evaluation_ids_for_training_jobs(training_job_ids, plan)

    run_queue(
        registry_file,
        gpus=(int(physical_gpu),),
        job_ids=training_job_ids,
        wait_dependencies=True,
        dependency_poll_seconds=dependency_poll_seconds,
        dependency_timeout_seconds=dependency_timeout_seconds,
    )

    registry = load_registry(registry_file)
    incomplete = [
        identifier
        for identifier in training_job_ids
        if registry.get("jobs", {}).get(identifier, {}).get("status") != "completed"
    ]
    if incomplete:
        states = {
            identifier: registry.get("jobs", {}).get(identifier, {}).get("status", "missing")
            for identifier in incomplete
        }
        raise RuntimeError(f"Selected training jobs did not all complete: {states}.")
    log_root = Path(log_dir) if log_dir else plan_file.parent / "campaign_logs"
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = log_root / f"campaign-gpu-{physical_gpu}.log"
    command = [
        sys.executable,
        "-m",
        "sam_rl.cli.evaluation_suite",
        "--run-plan",
        str(plan_file),
        "--repo-root",
        str(root),
        "--device",
        "cuda:0",
    ]
    for identifier in expected_eval_ids:
        command.extend(["--job-id", identifier])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    with log_path.open("ab") as log:
        log.write(("\n[campaign] " + " ".join(command) + "\n").encode())
        log.flush()
        result = subprocess.run(
            command,
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Evaluation CLI failed with exit code {result.returncode}; see {log_path}."
        )

    updated_plan = json.loads(plan_file.read_text(encoding="utf-8"))
    eval_jobs = {job["job_id"]: job for job in updated_plan.get("jobs", [])}
    failed = {
        identifier: eval_jobs.get(identifier, {}).get("status", "missing")
        for identifier in expected_eval_ids
        if eval_jobs.get(identifier, {}).get("status") != "complete"
    }
    if failed:
        raise RuntimeError(f"Selected evaluation jobs did not all complete: {failed}.")
    return {
        "physical_gpu": int(physical_gpu),
        "training_job_ids": list(training_job_ids),
        "evaluation_job_ids": expected_eval_ids,
        "evaluation_log": str(log_path),
    }


def resolve_path(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--evaluation-plan", required=True)
    parser.add_argument("--jobs", nargs="+", required=True)
    parser.add_argument("--gpu", type=int, required=True, help="Physical GPU ID on this host.")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--log-dir")
    parser.add_argument("--dependency-poll-seconds", type=int, default=20)
    parser.add_argument("--dependency-timeout-seconds", type=int, default=86_400)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_campaign(
        registry_path=args.registry,
        evaluation_plan_path=args.evaluation_plan,
        training_job_ids=tuple(args.jobs),
        physical_gpu=args.gpu,
        repo_root=args.repo_root,
        log_dir=args.log_dir,
        dependency_poll_seconds=args.dependency_poll_seconds,
        dependency_timeout_seconds=args.dependency_timeout_seconds,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
