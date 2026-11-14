from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import sam_rl.cli.experiment_suite as suite
from sam_rl.cli.experiment_suite import (
    DEFAULT_SPEC,
    build_host_assignment,
    claim_job,
    dependency_state,
    generate_configs,
    host_seeds,
    load_registry,
    make_job_records,
    register_jobs,
    validate_jobs,
)
from sam_rl.cli.train_rl_finetune import ensure_rl_run_dir


def test_primary_matrix_has_expected_count_and_dependencies() -> None:
    jobs = make_job_records()
    assert len(jobs) == 65
    by_id = {job["job_id"]: job for job in jobs}
    assert len(by_id) == 65
    assert by_id["R0-s17"]["dependencies"] == ["B2-s17"]
    assert by_id["R5-s101"]["dependencies"] == ["B2-s101"]
    assert by_id["R8-s29"]["dependencies"] == ["B3-s29"]
    assert by_id["R2-s43"]["dependencies"] == []
    assert sum(job["kind"] == "bc" for job in jobs) == 20
    assert sum(job["kind"] == "ppo" for job in jobs) == 45


def test_generated_configs_encode_factorial_controls_and_unique_paths(tmp_path: Path) -> None:
    rows = generate_configs(
        DEFAULT_SPEC,
        output_dir=tmp_path / "configs",
        repo_root=tmp_path,
        seeds=(17,),
    )
    assert len(rows) == 13
    configs = {row["job_id"]: yaml.safe_load(Path(row["config_path"]).read_text()) for row in rows}
    assert configs["B0-s17"]["model"]["reserve_mask_channel"] is True
    assert configs["B0-s17"]["model"]["use_current_mask"] is False
    assert configs["B1-s17"]["model"]["use_current_mask"] is True
    assert configs["B0-s17"]["train"]["aux_rollout_iou"]["weight"] == 0.0
    assert configs["B2-s17"]["train"]["aux_rollout_iou"]["weight"] > 0
    assert configs["B2-s17"]["evaluation"]["method_name"] == "B2"
    assert configs["B2-s17"]["oracle"]["max_budget"] == 5
    cache_budget = 12 * 1024**3
    assert configs["B2-s17"]["models"]["sam_embedding_cpu_cache_bytes"] == cache_budget
    assert configs["B2-s17"]["models"]["sam_embedding_cache_dir"] == (
        "artifacts/sam_embedding_cache/fss1000_vitb_224"
    )
    assert configs["R5-s17"]["models"]["sam_embedding_cpu_cache_bytes"] == cache_budget
    assert (
        configs["R5-s17"]["models"]["train_sam_embedding_cache_dir"]
        == (configs["R5-s17"]["models"]["eval_sam_embedding_cache_dir"])
    )

    random_scratch = configs["R2-s17"]["rl"]
    assert random_scratch["init_mode"] == "random"
    assert random_scratch["init_checkpoint"] is None
    assert random_scratch["kl_reference_mode"] == "none"
    assert random_scratch["kl_reference_checkpoint"] is None
    random_anchored = configs["R6-s17"]["rl"]
    assert random_anchored["init_mode"] == "random"
    assert random_anchored["init_checkpoint"] is None
    assert random_anchored["kl_reference_mode"] == "bc"
    assert random_anchored["kl_reference_checkpoint"].endswith("B2_s17_seed17_N5/best.pt")
    assert configs["R5-s17"]["rl"]["init_checkpoint"].endswith("B2_s17_seed17_N5/best.pt")
    assert configs["R8-s17"]["rl"]["init_checkpoint"].endswith("B3_s17_seed17_N5/best.pt")
    assert configs["R8-s17"]["rl"]["kl_reference_checkpoint"].endswith("B3_s17_seed17_N5/best.pt")

    paths = [Path(row["config_path"]) for row in rows]
    assert len(paths) == len(set(paths))
    assert all(path.is_file() for path in paths)


def test_config_generation_rejects_overwrite_with_different_protocol(tmp_path: Path) -> None:
    output = tmp_path / "configs"
    generate_configs(DEFAULT_SPEC, output_dir=output, repo_root=tmp_path, seeds=(17,))
    one_config = output / "B2-s17.yaml"
    one_config.write_text(one_config.read_text() + "# changed\n")
    with pytest.raises(FileExistsError):
        generate_configs(DEFAULT_SPEC, output_dir=output, repo_root=tmp_path, seeds=(17,))


def test_registry_dependency_resolution_and_hash_validation(tmp_path: Path) -> None:
    jobs = generate_configs(
        DEFAULT_SPEC, output_dir=tmp_path / "configs", repo_root=tmp_path, seeds=(17,)
    )
    registry_path = tmp_path / "runs/registry.json"
    register_jobs(registry_path, jobs)
    registry = load_registry(registry_path)
    assert not validate_jobs(registry)
    r0 = registry["jobs"]["R0-s17"]
    ready, missing = dependency_state(r0, registry["jobs"])
    assert not ready and missing == ["B2-s17"]
    b2 = registry["jobs"]["B2-s17"]
    checkpoint = tmp_path / "b2.pt"
    checkpoint.write_bytes(b"checkpoint")
    b2["status"] = "completed"
    b2["checkpoint"] = str(checkpoint)
    ready, missing = dependency_state(r0, registry["jobs"])
    assert ready and not missing

    Path(r0["config_path"]).write_text("corrupt: true\n")
    registry_path.write_text(json.dumps(registry))
    assert any("hash mismatch" in error for error in validate_jobs(load_registry(registry_path)))


def test_pilot_has_separate_namespace_and_limited_work(tmp_path: Path) -> None:
    rows = generate_configs(
        DEFAULT_SPEC,
        output_dir=tmp_path / "pilot-configs",
        repo_root=tmp_path,
        pilot=True,
        conditions=("B2", "B3", "R3", "R5", "R8"),
        seeds=(17,),
        pilot_train_episodes=32,
        pilot_val_episodes=6,
        pilot_env_steps=800,
    )
    assert [row["job_id"] for row in rows] == [
        "pilot-B2-s17",
        "pilot-B3-s17",
        "pilot-R3-s17",
        "pilot-R5-s17",
        "pilot-R8-s17",
    ]
    bc_cfg = yaml.safe_load(Path(rows[0]["config_path"]).read_text())
    ppo_cfg = yaml.safe_load(Path(rows[3]["config_path"]).read_text())
    assert bc_cfg["dataset"]["max_train_episodes"] == 32
    assert bc_cfg["dataset"]["max_val_episodes"] == 6
    assert bc_cfg["train"]["epochs"] == 1
    assert ppo_cfg["rl"]["total_env_steps"] == 800
    assert ppo_cfg["validation"]["selection_episodes"] == 6
    assert ppo_cfg["validation"]["evaluate_every_updates"] == 5
    assert "/pilot/" in ppo_cfg["paths"]["run_root"]
    assert rows[2]["dependencies"] == []
    assert rows[3]["dependencies"] == ["pilot-B2-s17"]
    assert rows[4]["dependencies"] == ["pilot-B3-s17"]


def test_host_assignment_covers_each_job_once_and_keeps_seed_dependencies_local() -> None:
    plan = build_host_assignment()
    assert plan["total_jobs"] == 65
    assert plan["hosts"]["worker-a"]["job_count"] == 39
    assert plan["hosts"]["worker-b"]["job_count"] == 13
    assert set(plan["hosts"]["worker-b"]["gpu_ids"]) == {0, 1, 2}
    assert plan["hosts"]["worker-c"]["job_count"] == 13
    assert plan["hosts"]["worker-c"]["gpu_ids"] == [0]
    assert set(plan["hosts"]["worker-a"]["gpu_ids"]) == {0, 1, 2}
    assert len(plan["jobs"]) == len(set(plan["jobs"])) == 65
    first_cohort_ids = {job["job_id"] for job in suite.make_job_records(seeds=(17, 29, 43))}
    remaining_cohort_ids = {job["job_id"] for job in suite.make_job_records(seeds=(71, 101))}
    assert remaining_cohort_ids.isdisjoint(first_cohort_ids)
    assert remaining_cohort_ids | first_cohort_ids == set(plan["jobs"])
    for assignment in plan["jobs"].values():
        for dependency in assignment["dependencies"]:
            assert plan["jobs"][dependency]["host"] == assignment["host"]
            assert plan["jobs"][dependency]["seed"] == assignment["seed"]
    for host in plan["hosts"].values():
        scheduled = [identifier for ids in host["jobs_by_gpu"].values() for identifier in ids]
        assert len(scheduled) == len(set(scheduled)) == host["job_count"]


def test_host_cohorts_support_a_third_host_in_suite_spec(tmp_path: Path) -> None:
    spec = json.loads(DEFAULT_SPEC.read_text())
    spec["host_layout"] = {
        "worker-a": {"seeds": [17, 29, 43], "gpu_ids": [0, 1, 2]},
        "worker-b": {"seeds": [71], "gpu_ids": [0, 1, 2]},
        "worker-c": {"seeds": [101], "gpu_ids": [0]},
    }
    spec_path = tmp_path / "chair-v2.json"
    spec_path.write_text(json.dumps(spec))
    plan = build_host_assignment(spec_path)
    assert host_seeds("worker-a", spec_path) == (17, 29, 43)
    assert host_seeds("worker-c", spec_path) == (101,)
    assert plan["hosts"]["worker-c"]["gpu_ids"] == [0]
    assert plan["hosts"]["worker-c"]["job_count"] == 13
    assert {row["seed"] for row in plan["jobs"].values() if row["host"] == "worker-c"} == {101}


def test_host_layout_rejects_negative_or_duplicate_gpu_ids(tmp_path: Path) -> None:
    spec = json.loads(DEFAULT_SPEC.read_text())
    spec["host_layout"]["worker-a"]["gpu_ids"] = [0, 0]
    spec_path = tmp_path / "chair-v2.json"
    spec_path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="unique GPU ids"):
        suite.configured_host_layout(spec_path)

    spec["host_layout"]["worker-a"]["gpu_ids"] = [-1]
    spec_path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="nonnegative physical GPU ids"):
        suite.configured_host_layout(spec_path)


def test_unknown_cli_host_reports_configured_choices(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["experiment_suite", "generate", "--host", "worker-z"],
    )
    with pytest.raises(SystemExit, match=r"Unknown --host 'worker-z'.*worker-b"):
        suite.main()


def test_job_claim_is_atomic_and_only_accepts_planned_once(tmp_path: Path) -> None:
    rows = generate_configs(
        DEFAULT_SPEC,
        output_dir=tmp_path / "configs",
        repo_root=tmp_path,
        seeds=(17,),
        conditions=("B0",),
    )
    registry_path = tmp_path / "registry.json"
    register_jobs(registry_path, rows)
    kwargs = {
        "gpu": 0,
        "command": ["python", "train"],
        "log_path": tmp_path / "logs/B0-s17.log",
        "started": 1.0,
        "retry": False,
    }
    assert claim_job(registry_path, "B0-s17", **kwargs)
    assert not claim_job(registry_path, "B0-s17", **kwargs)
    running = load_registry(registry_path)["jobs"]["B0-s17"]
    assert running["status"] == "running"
    assert running["gpu"] == 0
    assert "hostname" in running["runner"]


def test_cpu_job_claim_keeps_gpu_null(tmp_path: Path) -> None:
    rows = generate_configs(
        DEFAULT_SPEC,
        output_dir=tmp_path / "configs",
        repo_root=tmp_path,
        seeds=(17,),
        conditions=("B0",),
    )
    registry_path = tmp_path / "registry.json"
    register_jobs(registry_path, rows)
    assert claim_job(
        registry_path,
        "B0-s17",
        gpu=None,
        command=["python", "train"],
        log_path=tmp_path / "logs/B0-s17.log",
        started=1.0,
        retry=False,
    )
    running = load_registry(registry_path)["jobs"]["B0-s17"]
    assert running["gpu"] is None
    assert running["runner"]["device"] == "cpu"


def test_ppo_artifact_record_matches_trainer_run_directory(tmp_path: Path) -> None:
    rows = generate_configs(
        DEFAULT_SPEC,
        output_dir=tmp_path / "configs",
        repo_root=tmp_path,
        seeds=(17,),
        conditions=("R5",),
    )
    job = rows[0]
    cfg = yaml.safe_load(Path(job["config_path"]).read_text())
    cfg["paths"]["run_root"] = str(tmp_path / "outputs")
    assert cfg["train"]["run_name"] == cfg["rl"]["run_name"]

    trainer_run_dir = ensure_rl_run_dir(cfg)
    scheduler_run_dir, checkpoint = suite._artifact_record(job, cfg)

    assert Path(scheduler_run_dir) == trainer_run_dir
    assert Path(checkpoint).parent == trainer_run_dir


def test_wait_for_dependencies_returns_after_checkpoint_publication(
    tmp_path: Path, monkeypatch
) -> None:
    rows = generate_configs(
        DEFAULT_SPEC,
        output_dir=tmp_path / "configs",
        repo_root=tmp_path,
        seeds=(17,),
        conditions=("B2", "R5"),
    )
    registry_path = tmp_path / "registry.json"
    register_jobs(registry_path, rows)
    checkpoint = tmp_path / "b2.pt"

    def finish_dependency(_seconds: int) -> None:
        checkpoint.write_bytes(b"completed checkpoint")

        def complete(job: dict) -> None:
            job["status"] = "completed"
            job["checkpoint"] = str(checkpoint)
            job["checkpoint_sha256"] = suite.sha256_file(checkpoint)

        suite.update_job(registry_path, "B2-s17", complete)

    monkeypatch.setattr(suite.time, "sleep", finish_dependency)
    suite.wait_for_dependencies(registry_path, "R5-s17", poll_seconds=20, timeout_seconds=60)
    ready, missing = dependency_state(
        load_registry(registry_path)["jobs"]["R5-s17"], load_registry(registry_path)["jobs"]
    )
    assert ready and not missing


def test_wait_for_dependencies_fails_fast_for_failed_parent(tmp_path: Path) -> None:
    rows = generate_configs(
        DEFAULT_SPEC,
        output_dir=tmp_path / "configs",
        repo_root=tmp_path,
        seeds=(17,),
        conditions=("B2", "R5"),
    )
    registry_path = tmp_path / "registry.json"
    register_jobs(registry_path, rows)
    suite.update_job(registry_path, "B2-s17", lambda job: job.__setitem__("status", "failed"))
    with pytest.raises(RuntimeError, match="B2-s17=failed"):
        suite.wait_for_dependencies(registry_path, "R5-s17", poll_seconds=20, timeout_seconds=60)


def test_wait_for_dependencies_has_a_finite_timeout(tmp_path: Path, monkeypatch) -> None:
    rows = generate_configs(
        DEFAULT_SPEC,
        output_dir=tmp_path / "configs",
        repo_root=tmp_path,
        seeds=(17,),
        conditions=("B2", "R5"),
    )
    registry_path = tmp_path / "registry.json"
    register_jobs(registry_path, rows)
    ticks = iter((0.0, 0.0, 2.0, 2.0))
    monkeypatch.setattr(suite.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(suite.time, "sleep", lambda _seconds: None)
    with pytest.raises(TimeoutError, match="Timed out waiting"):
        suite.wait_for_dependencies(registry_path, "R5-s17", poll_seconds=20, timeout_seconds=1)


def _mock_successful_training_process(registry_path: Path, monkeypatch, calls: list[dict]):
    def fake_popen(command, *, cwd, env, stdout, stderr):
        calls.append({"command": command, "cwd": cwd, "env": env, "stderr": stderr})
        job = load_registry(registry_path)["jobs"]["B0-s17"]
        cfg = suite.read_yaml(Path(job["config_path"]))
        run_dir, checkpoint = suite._artifact_record(job, cfg)
        Path(run_dir).mkdir(parents=True, exist_ok=True)
        Path(checkpoint).write_bytes(b"mock best checkpoint")
        return SimpleNamespace(pid=os.getpid(), wait=lambda: 0)

    monkeypatch.setattr(suite.subprocess, "Popen", fake_popen)


def _fresh_bc_registry(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setattr(suite, "REPO_ROOT", tmp_path)
    jobs = generate_configs(
        DEFAULT_SPEC,
        output_dir=tmp_path / "configs",
        repo_root=tmp_path,
        seeds=(17,),
        conditions=("B0",),
    )
    registry_path = tmp_path / "registry/registry.json"
    register_jobs(registry_path, jobs)
    return registry_path


def test_mocked_launch_creates_fresh_log_parent_and_records_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    registry_path = _fresh_bc_registry(tmp_path, monkeypatch)
    log_path = registry_path.parent / "logs/B0-s17.log"
    assert not log_path.parent.exists()
    calls: list[dict] = []
    _mock_successful_training_process(registry_path, monkeypatch, calls)

    records = suite.run_queue(registry_path, gpus=(0,), job_ids=("B0-s17",))

    assert records[0]["status"] == "completed"
    assert log_path.is_file()
    assert "[launcher]" in log_path.read_text()
    assert calls[0]["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    completed = load_registry(registry_path)["jobs"]["B0-s17"]
    assert completed["status"] == "completed"
    assert completed["checkpoint_sha256"] == suite.sha256_file(completed["checkpoint"])


def test_prelaunch_failure_can_retry_without_resume_checkpoint(tmp_path: Path, monkeypatch) -> None:
    registry_path = _fresh_bc_registry(tmp_path, monkeypatch)
    log_path = registry_path.parent / "logs/B0-s17.log"
    original_open = Path.open

    def fail_first_log_open(path, *args, **kwargs):
        if path == log_path:
            raise PermissionError("simulated log open failure")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_first_log_open)
    with pytest.raises(PermissionError, match="simulated log open failure"):
        suite.run_queue(registry_path, gpus=(0,), job_ids=("B0-s17",))
    failed = load_registry(registry_path)["jobs"]["B0-s17"]
    assert failed["status"] == "failed"
    assert failed["failure_phase"] == "prelaunch"
    assert failed["process_started"] is False

    monkeypatch.setattr(Path, "open", original_open)
    calls: list[dict] = []
    _mock_successful_training_process(registry_path, monkeypatch, calls)
    result = suite.run_queue(
        registry_path,
        gpus=(0,),
        job_ids=("B0-s17",),
        resume_failed=True,
    )
    assert result[0]["status"] == "completed"
    assert "--resume" not in calls[0]["command"]
