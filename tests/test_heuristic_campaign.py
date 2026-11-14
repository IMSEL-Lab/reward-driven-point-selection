from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from sam_rl.cli import heuristic_campaign as campaign
from sam_rl.episode_manifest import episode_identity, load_episode_manifest
from sam_rl.result_manifest import write_result_metadata


def test_generator_creates_validation_sweep_and_six_frozen_runs(tmp_path: Path) -> None:
    campaign_dir = tmp_path / "mask_aware_heuristic_v1"
    ledger = campaign.generate_campaign(campaign_dir)

    assert len(ledger["jobs"]) == 7
    assert ledger["evaluation_source_fingerprint"]["evaluation_source_sha256"]
    assert set(ledger["expected_model_weight_hashes"]) == {
        "dinov2_vitl14_pretrain.pth",
        "sam_vit_b_01ec64.pth",
        "sam_vit_h_4b8939.pth",
    }
    sweep = ledger["jobs"]["sweep_fss_val357_sam_vit_b"]
    assert sweep["phase"] == "validation"
    assert sweep["expected_episode_count"] == 357
    assert sweep["expected_methods"] == [
        "topk_argmax",
        "mask_aware_greedy",
        "sim_weighted_fps[alpha=0.25]",
        "sim_weighted_fps[alpha=0.5]",
        "sim_weighted_fps[alpha=0.75]",
    ]
    assert sweep["expected_seeds"] == {method: [0] for method in sweep["expected_methods"]}
    assert sweep["manifest_sha256"]
    assert sweep["config_sha256"]

    frozen = [job for job in ledger["jobs"].values() if job["phase"] == "test"]
    assert len(frozen) == 6
    assert {job["expected_episode_count"] for job in frozen} == {450, 660}
    assert {tuple(job["dependencies"]) for job in frozen} == {("sweep_fss_val357_sam_vit_b",)}
    assert len({job["output_csv"] for job in frozen}) == 6
    for job in frozen:
        config = json.loads(Path(job["config"]).read_text())
        assert config["models"]["sam_model_type"] in {"vit_b", "vit_h"}
        assert config["models"]["device"] == "cuda:0"
        assert config["evaluation"]["point_budgets"] == [5]
        assert config["evaluation"]["seeds"] == [0]
        assert config["evaluation"]["deterministic_seeds"] == [0]
        assert config["evaluation"]["prompt_snap_grid"] == 37
        assert config["campaign"]["precision"] == "fp32"
        assert config["campaign"]["test_tuning"] is False
        assert config["campaign"]["phase"] == "test"

    validation_config = json.loads(Path(sweep["config"]).read_text())
    assert validation_config["campaign"]["phase"] == "validation"


def test_generator_refuses_to_overwrite_existing_campaign(tmp_path: Path) -> None:
    campaign_dir = tmp_path / "existing"
    campaign_dir.mkdir()
    (campaign_dir / "operator.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        campaign.generate_campaign(campaign_dir)
    assert (campaign_dir / "operator.txt").read_text() == "keep"


def test_test_job_fails_closed_until_validation_job_completed(tmp_path: Path) -> None:
    ledger_path = tmp_path / "campaign" / "ledger.json"
    ledger = campaign.generate_campaign(ledger_path.parent)
    campaign._atomic_json(ledger_path, ledger)

    with pytest.raises(RuntimeError, match=r"dependency .* is not completed"):
        campaign.run_job(ledger_path, "frozen_fss_test450_sam_vit_b", 2)
    assert (
        campaign._load_ledger(ledger_path)["jobs"]["frozen_fss_test450_sam_vit_b"]["status"]
        == "planned"
    )


def test_mocked_sweep_launch_uses_requested_physical_gpu_and_marks_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_dir = tmp_path / "campaign"
    ledger = campaign.generate_campaign(campaign_dir)
    ledger_path = campaign_dir / "ledger.json"
    campaign._atomic_json(ledger_path, ledger)
    sweep = ledger["jobs"]["sweep_fss_val357_sam_vit_b"]
    frozen_path = Path(sweep["frozen_params"])
    launched: dict[str, object] = {}

    def fake_run(command, *, cwd, env, stdout, stderr, check=False, **kwargs):
        launched.update(command=command, cwd=cwd, gpu=env["CUDA_VISIBLE_DEVICES"])
        assert stdout is not None
        assert stderr == campaign.subprocess.STDOUT
        frozen_path.parent.mkdir(parents=True, exist_ok=True)
        frozen_path.write_text(
            json.dumps(
                {
                    "best_methods": [
                        {"family": "topk_argmax", "params": {}},
                        {"family": "mask_aware_greedy", "params": {}},
                        {"family": "sim_weighted_fps", "params": {"alpha": 0.5}},
                    ]
                }
            )
        )
        output = Path(sweep["output_csv"])
        output.parent.mkdir(parents=True, exist_ok=True)
        methods = sweep["expected_methods"]
        episodes = load_episode_manifest(
            sweep["manifest"],
            validate_paths=False,
            data_root=campaign.REPO_ROOT / "data/FSS-1000",
        )
        with output.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "method",
                    "N",
                    "image_id",
                    "episode_key",
                    "seed",
                    "iou",
                    "dice",
                    "boundary_iou",
                    "wallclock_ms",
                ],
            )
            writer.writeheader()
            for episode in episodes:
                for method in methods:
                    writer.writerow(
                        {
                            "method": method,
                            "N": 5,
                            "image_id": episode.image_id,
                            "episode_key": episode_identity(episode),
                            "seed": 0,
                            "iou": 0.5,
                            "dice": 0.5,
                            "boundary_iou": 0.5,
                            "wallclock_ms": 1,
                        }
                    )
        write_result_metadata(
            output,
            episodes=episodes,
            dataset="fss1000_val357",
            split="val",
            condition="mask_aware_heuristic_validation_v1",
            model_config={"fake": "mock-runner-test"},
            expected_result_rows=len(episodes) * len(methods),
            decoder="vit_b",
        )
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(campaign.subprocess, "run", fake_run)
    monkeypatch.setattr(campaign, "_write_summary", lambda job: None)
    fingerprint = {
        "dino_checkpoint": {
            "path": "/weights/dino.pth",
            "expected_sha256": "a",
            "actual_sha256": "a",
        },
        "sam_checkpoint": {
            "path": "/weights/sam.pth",
            "expected_sha256": "b",
            "actual_sha256": "b",
        },
    }
    monkeypatch.setattr(campaign, "_verify_checkpoint_weights", lambda job, ledger: fingerprint)

    assert campaign.run_job(ledger_path, "sweep_fss_val357_sam_vit_b", 3) == 0
    updated = campaign._load_ledger(ledger_path)["jobs"]["sweep_fss_val357_sam_vit_b"]
    assert updated["status"] == "completed"
    assert updated["gpu"] == 3
    assert updated["frozen_params_sha256"] == campaign._sha256(frozen_path)
    metadata = json.loads(Path(updated["metadata_json"]).read_text())
    assert metadata["checkpoint_fingerprints"] == fingerprint
    assert launched["gpu"] == "3"
    assert launched["cwd"] == campaign.REPO_ROOT
    launched_command = cast(list[str], launched["command"])
    assert isinstance(launched_command, list)
    assert "--mode" in launched_command
    assert launched_command[launched_command.index("--mode") + 1] == "sweep"


def test_nonzero_subprocess_marks_job_failed_without_retry_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    campaign_dir = tmp_path / "campaign"
    ledger = campaign.generate_campaign(campaign_dir)
    ledger_path = campaign_dir / "ledger.json"
    campaign._atomic_json(ledger_path, ledger)

    def fail_run(*args, **kwargs):
        return type("Completed", (), {"returncode": 19})()

    monkeypatch.setattr(campaign.subprocess, "run", fail_run)
    monkeypatch.setattr(campaign, "_verify_checkpoint_weights", lambda job, ledger: {})
    with pytest.raises(campaign.subprocess.CalledProcessError):
        campaign.run_job(ledger_path, "sweep_fss_val357_sam_vit_b", 0)
    job = campaign._load_ledger(ledger_path)["jobs"]["sweep_fss_val357_sam_vit_b"]
    assert job["status"] == "failed"
    with pytest.raises(RuntimeError, match="is failed"):
        campaign.run_job(ledger_path, "sweep_fss_val357_sam_vit_b", 0)


def test_weight_hash_verification_uses_tracked_manifest_without_requiring_weights_at_plan(
    tmp_path: Path,
) -> None:
    dino = tmp_path / "dinov2_vitl14_pretrain.pth"
    sam = tmp_path / "sam_vit_b_01ec64.pth"
    dino.write_bytes(b"dino")
    sam.write_bytes(b"sam")
    manifest = tmp_path / "weights.sha256"
    lines = [
        f"{hashlib.sha256(dino.read_bytes()).hexdigest()}  {dino.name}",
        f"{hashlib.sha256(sam.read_bytes()).hexdigest()}  {sam.name}",
        f"{'0' * 64}  sam_vit_h_4b8939.pth",
    ]
    manifest.write_text("\n".join(lines) + "\n")
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "models": {
                    "dino_checkpoint": str(dino),
                    "sam_checkpoint": str(sam),
                }
            }
        )
    )
    ledger = {
        "weights_manifest": str(manifest),
        "weights_manifest_sha256": campaign._sha256(manifest),
        "expected_model_weight_hashes": campaign._weight_hashes(manifest),
    }
    job = {"config": str(config)}

    result = campaign._verify_checkpoint_weights(job, ledger)
    assert result["dino_checkpoint"]["actual_sha256"] == hashlib.sha256(b"dino").hexdigest()
    assert result["sam_checkpoint"]["actual_sha256"] == hashlib.sha256(b"sam").hexdigest()

    sam.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        campaign._verify_checkpoint_weights(job, ledger)
