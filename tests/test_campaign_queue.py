from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import sam_rl.cli.campaign_queue as campaign
from sam_rl.cli.evaluation_suite import (
    BENCHMARKS,
    DECODERS,
    evaluation_job_id,
    generate_suite,
)
from sam_rl.cli.experiment_suite import generate_configs

REPO = Path(__file__).resolve().parents[1]
REAL_EVALUATION_MANIFESTS = (
    REPO / "artifacts/protocol/fss_manifests/fss_class_disjoint_canonical66_v2/test.json",
    REPO / "artifacts/protocol/fss_manifests/fss_canonical66_v2/test.json",
    REPO / "artifacts/protocol/kvasir_manifests/kvasir_seg_v2/test.json",
)


def make_registry_plan(tmp_path: Path, training_status: str = "planned"):
    registry_path = tmp_path / "training/registry.json"
    plan_path = tmp_path / "evaluation/evaluation_plan.json"
    registry_path.parent.mkdir(parents=True)
    plan_path.parent.mkdir(parents=True)
    registry_path.write_text(
        json.dumps(
            {
                "jobs": {
                    "B2-s17": {
                        "job_id": "B2-s17",
                        "condition": "B2",
                        "seed": 17,
                        "kind": "bc",
                        "status": training_status,
                    }
                }
            }
        )
    )
    eval_jobs = []
    for benchmark_name, dataset, protocol, _manifest in BENCHMARKS:
        for decoder in DECODERS:
            eval_jobs.append(
                {
                    "job_id": evaluation_job_id("B2", 17, benchmark_name, decoder),
                    "training_job_id": "B2-s17",
                    "condition": "B2",
                    "training_seed": 17,
                    "benchmark": protocol,
                    "dataset": dataset,
                    "decoder": decoder,
                    "status": "pending",
                    "csv": f"runs/{benchmark_name}_{decoder}.csv",
                }
            )
    plan_path.write_text(json.dumps({"registry_path": str(registry_path), "jobs": eval_jobs}))
    return registry_path, plan_path


def complete_training(registry_path: Path, *, gpus, job_ids, wait_dependencies, **_kwargs):
    assert gpus == (3,)
    assert job_ids == ("B2-s17",)
    assert wait_dependencies is True
    registry = json.loads(registry_path.read_text())
    registry["jobs"]["B2-s17"]["status"] = "completed"
    registry_path.write_text(json.dumps(registry))
    return [{"job_id": "B2-s17", "status": "completed"}]


def test_evaluation_id_selection_is_exactly_six_per_training_job(tmp_path: Path) -> None:
    _registry_path, plan_path = make_registry_plan(tmp_path)
    plan = json.loads(plan_path.read_text())
    selected = campaign.evaluation_ids_for_training_jobs(("B2-s17",), plan)
    assert len(selected) == 6
    assert len(set(selected)) == 6
    assert set(selected) == {job["job_id"] for job in plan["jobs"]}

    with pytest.raises(ValueError, match="exactly 3 benchmarks x 2 decoders"):
        campaign.evaluation_ids_for_training_jobs(("B2-s17",), {"jobs": plan["jobs"][:-1]})


@pytest.mark.skipif(
    not all(path.is_file() for path in REAL_EVALUATION_MANIFESTS),
    reason="Install the separate protocol bundle to test plans built from real manifests.",
)
def test_ids_from_real_generated_suite_for_bc_and_ppo(tmp_path: Path) -> None:
    repo = REPO
    config_dir = tmp_path / "training_configs"
    rows = generate_configs(
        output_dir=config_dir,
        repo_root=repo,
        conditions=("B2", "R3"),
        seeds=(17,),
    )
    registry_path = tmp_path / "training" / "registry.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(
        json.dumps({"jobs": {row["job_id"]: {**row, "status": "planned"} for row in rows}})
    )
    generated = generate_suite(
        spec_path=repo / "configs/experiments/chair_v2.json",
        registry_path=registry_path,
        repo_root=repo,
        output_root=tmp_path / "artifacts/evaluation",
        config_root=tmp_path / "configs/evaluation",
        conditions=("B2", "R3"),
        seeds=(17,),
    )

    for training_id in ("B2-s17", "R3-s17"):
        selected = campaign.evaluation_ids_for_training_jobs((training_id,), generated)
        assert len(selected) == 6
        records = [job for job in generated["jobs"] if job["training_job_id"] == training_id]
        assert {job["benchmark"] for job in records} == {entry[2] for entry in BENCHMARKS}
        assert set(selected) == {job["job_id"] for job in records}


def test_campaign_completes_training_then_dispatches_six_eval_ids_with_physical_gpu_env(
    tmp_path: Path, monkeypatch
) -> None:
    registry_path, plan_path = make_registry_plan(tmp_path)
    monkeypatch.setattr(campaign, "run_queue", complete_training)
    subprocess_calls = []

    def fake_subprocess_run(command, *, cwd, env, stdout, stderr, check):
        subprocess_calls.append(
            {"command": command, "cwd": cwd, "env": env.copy(), "stdout": stdout, "stderr": stderr}
        )
        plan = json.loads(plan_path.read_text())
        selected_ids = [
            command[index + 1] for index, item in enumerate(command[:-1]) if item == "--job-id"
        ]
        for job in plan["jobs"]:
            if job["job_id"] in selected_ids:
                job["status"] = "complete"
        plan_path.write_text(json.dumps(plan))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(campaign.subprocess, "run", fake_subprocess_run)
    result = campaign.run_campaign(
        registry_path=registry_path,
        evaluation_plan_path=plan_path,
        training_job_ids=("B2-s17",),
        physical_gpu=3,
        repo_root=tmp_path,
        log_dir=tmp_path / "campaign-logs",
    )

    assert len(result["evaluation_job_ids"]) == 6
    call = subprocess_calls[0]
    assert call["env"]["CUDA_VISIBLE_DEVICES"] == "3"
    assert call["command"][call["command"].index("--device") + 1] == "cuda:0"
    assert call["command"].count("--job-id") == 6
    assert Path(call["stdout"].name).name == "campaign-gpu-3.log"
    assert Path(call["stdout"].name).parent == tmp_path / "campaign-logs"
    assert "torch" not in vars(campaign)


def test_failed_training_is_not_reset_or_followed_by_evaluation(
    tmp_path: Path, monkeypatch
) -> None:
    registry_path, plan_path = make_registry_plan(tmp_path, training_status="failed")
    run_queue_called = False
    subprocess_called = False

    def forbidden_run_queue(*_args, **_kwargs):
        nonlocal run_queue_called
        run_queue_called = True

    def forbidden_subprocess(*_args, **_kwargs):
        nonlocal subprocess_called
        subprocess_called = True

    monkeypatch.setattr(campaign, "run_queue", forbidden_run_queue)
    monkeypatch.setattr(campaign.subprocess, "run", forbidden_subprocess)
    with pytest.raises(RuntimeError, match="B2-s17 is failed; failed states are not reset"):
        campaign.run_campaign(
            registry_path=registry_path,
            evaluation_plan_path=plan_path,
            training_job_ids=("B2-s17",),
            physical_gpu=3,
            repo_root=tmp_path,
        )
    assert not run_queue_called
    assert not subprocess_called
    assert json.loads(registry_path.read_text())["jobs"]["B2-s17"]["status"] == "failed"


def test_incomplete_training_and_failed_evaluation_fail_closed(tmp_path: Path, monkeypatch) -> None:
    registry_path, plan_path = make_registry_plan(tmp_path)
    monkeypatch.setattr(campaign, "run_queue", lambda *_args, **_kwargs: [])
    with pytest.raises(RuntimeError, match="did not all complete"):
        campaign.run_campaign(
            registry_path=registry_path,
            evaluation_plan_path=plan_path,
            training_job_ids=("B2-s17",),
            physical_gpu=3,
            repo_root=tmp_path,
            log_dir=tmp_path / "logs",
        )

    complete_training(registry_path, gpus=(3,), job_ids=("B2-s17",), wait_dependencies=True)

    def eval_failed(_command, **_kwargs):
        plan = json.loads(plan_path.read_text())
        for job in plan["jobs"]:
            job["status"] = "failed"
        plan_path.write_text(json.dumps(plan))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(campaign.subprocess, "run", eval_failed)
    with pytest.raises(RuntimeError, match="did not all complete"):
        campaign.run_campaign(
            registry_path=registry_path,
            evaluation_plan_path=plan_path,
            training_job_ids=("B2-s17",),
            physical_gpu=3,
            repo_root=tmp_path,
            log_dir=tmp_path / "logs",
        )
    assert all(job["status"] == "failed" for job in json.loads(plan_path.read_text())["jobs"])
