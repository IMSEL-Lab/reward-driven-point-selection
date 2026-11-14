from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from sam_rl.cli.analyze_experiments import _normalize_run
from sam_rl.cli.evaluation_suite import (
    BENCHMARKS,
    CONDITIONS,
    DECODERS,
    SEEDS,
    _checkpoint_for_record,
    _claim_job,
    _dispatch_integrity_error,
    _expected_csv_path,
    _finish_job,
    _sync_collection,
    evaluation_identities,
    generate_suite,
    sha256_file,
)
from sam_rl.cli.train_rl_finetune import ensure_rl_run_dir
from sam_rl.evaluation_provenance import (
    evaluation_config_sha256,
    evaluation_source_fingerprint,
)

REPO = Path(__file__).resolve().parents[1]


def training_config(condition: str, seed: int, root: Path) -> dict:
    base = {
        "seed": seed,
        "dataset": {
            "root": "data/FSS-1000",
            "data_root": "data/FSS-1000",
            "split_file": "artifacts/protocol/splits/fss_class_disjoint_canonical66_v2.json",
            "episode_manifests": {},
            "train_split": "train",
            "val_split": "val",
            "test_split": "test",
        },
        "models": {
            "device": "cuda",
            "image_size": 224,
            "precision": "fp32",
            "decoder_policy": "single",
            "dino_checkpoint": "artifacts/weights/dino.pth",
            "eval_sam_checkpoint": "artifacts/weights/sam_vit_b_01ec64.pth",
            "eval_sam_model_type": "vit_b",
            "oracle_sam_checkpoint": "artifacts/weights/sam_vit_b_01ec64.pth",
            "oracle_sam_model_type": "vit_b",
            "train_sam_checkpoint": "artifacts/weights/sam_vit_b_01ec64.pth",
            "train_sam_model_type": "vit_b",
        },
        "protocol": {"decoder_policy": "single"},
        "model": {"grid": 37, "use_current_mask": False},
        "evaluation": {"method_name": "placeholder", "eval_seed": 0},
        "train": {"run_name": f"{condition}_s{seed}", "budget": 5, "condition_id": condition},
        "paths": {"run_root": f"artifacts/training/{condition}/s{seed}"},
        "validation": {"selection_episodes": 357, "eval_max_episodes": 357},
    }
    if condition.startswith("R"):
        base["rl"] = {"run_name": f"{condition}_s{seed}", "budget": 5}
    else:
        base["models"]["sam_model_type"] = "vit_b"
    return base


class EvaluationSuiteTests(unittest.TestCase):
    def test_primary_plan_has_exact_390_unique_evaluation_identities(self):
        identities = evaluation_identities()
        self.assertEqual(len(identities), 390)
        self.assertEqual(len(set(identities)), 390)
        self.assertEqual(set(row[0] for row in identities), set(CONDITIONS))
        self.assertEqual(set(row[1] for row in identities), set(SEEDS))
        self.assertEqual(set(row[2] for row in identities), {item[0] for item in BENCHMARKS})
        self.assertEqual(set(row[3] for row in identities), set(DECODERS))
        self.assertEqual(len(evaluation_identities(seeds=(17, 29))), 156)

    def test_collection_accepts_pending_ledger_entries(self):
        run = _normalize_run(
            {
                "condition": "B0",
                "training_seed": 17,
                "eval_seed": 0,
                "dataset": "fss1000",
                "split": "test",
                "decoder": "vit_b",
                "protocol": "single|fss_test450",
                "status": "pending",
                "reason": "training incomplete",
            },
            Path("."),
        )
        self.assertEqual(run["status"], "pending")

    def test_collection_sync_preserves_expected_count_fields(self):
        plan = {
            "collection": {"runs": []},
            "jobs": [
                {
                    "condition": "B0",
                    "training_seed": 17,
                    "eval_seed": 0,
                    "dataset": "fss1000",
                    "split": "test",
                    "decoder": "vit_b",
                    "protocol": "single|fss_test450",
                    "status": "pending",
                    "expected_episode_count": 450,
                    "expected_result_rows": 450,
                }
            ],
        }

        _sync_collection(plan)

        run = plan["collection"]["runs"][0]
        self.assertEqual(run["expected_episode_count"], 450)
        self.assertEqual(run["expected_result_rows"], 450)

    def test_shared_plan_updates_preserve_other_runner_job_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan_path = root / "evaluation_plan.json"
            plan = {
                "collection": {"runs": []},
                "jobs": [
                    {
                        "job_id": job_id,
                        "condition": "B0",
                        "training_seed": 17,
                        "eval_seed": 0,
                        "dataset": "fss1000",
                        "split": "test",
                        "decoder": "vit_b",
                        "protocol": "single|fss-test",
                        "status": "pending",
                        "csv": f"{job_id}.csv",
                        "per_step_csv": f"{job_id}.per_step.csv",
                        "metadata": f"{job_id}.metadata.json",
                        "budget": 5,
                    }
                    for job_id in ("job-a", "job-b")
                ],
            }
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            self.assertEqual(_claim_job(plan_path, "job-a")["status"], "running")
            self.assertEqual(_claim_job(plan_path, "job-b")["status"], "running")
            self.assertIsNone(_claim_job(plan_path, "job-b"))
            _finish_job(plan_path, "job-a", status="complete", reason=None)
            current = json.loads(plan_path.read_text(encoding="utf-8"))
            statuses = {job["job_id"]: job["status"] for job in current["jobs"]}
            self.assertEqual(statuses, {"job-a": "complete", "job-b": "running"})
            self.assertEqual(current["collection"]["runs"][1]["status"], "pending")

    def test_dispatch_rechecks_selected_checkpoint_configs_and_manifest_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, content in (
                ("best.pt", b"selected best"),
                ("evaluation.yaml", b"evaluation: {}\n"),
                ("training.yaml", b"training config"),
                ("manifest.json", b"frozen manifest"),
            ):
                (root / name).write_bytes(content)
            (root / "sam_rl").mkdir()
            (root / "sam_rl" / "evaluation.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
            registry = {
                "jobs": {
                    "B0-s17": {
                        "status": "completed",
                        "checkpoint": "best.pt",
                        "checkpoint_sha256": sha256_file(root / "best.pt"),
                        "config_sha256": sha256_file(root / "training.yaml"),
                    }
                }
            }
            (root / "registry.json").write_text(json.dumps(registry), encoding="utf-8")
            (root / "evaluation_plan.json").write_text(
                json.dumps({"registry_path": "registry.json"}), encoding="utf-8"
            )
            job = {
                "training_job_id": "B0-s17",
                "checkpoint": "best.pt",
                "checkpoint_sha256": sha256_file(root / "best.pt"),
                "config": "evaluation.yaml",
                "config_sha256": sha256_file(root / "evaluation.yaml"),
                "evaluation_config_sha256": evaluation_config_sha256({"evaluation": {}}),
                "evaluation_source_sha256": evaluation_source_fingerprint(root)[
                    "evaluation_source_sha256"
                ],
                "training_config": "training.yaml",
                "training_config_sha256": sha256_file(root / "training.yaml"),
                "manifest": "manifest.json",
                "manifest_sha256": sha256_file(root / "manifest.json"),
            }
            plan_path = root / "evaluation_plan.json"
            self.assertIsNone(_dispatch_integrity_error(job, plan_path, root))
            source_file = root / "sam_rl" / "evaluation.py"
            source_file.write_text("VALUE = 2\n", encoding="utf-8")
            self.assertIn("source fingerprint", _dispatch_integrity_error(job, plan_path, root))
            source_file.write_text("VALUE = 1\n", encoding="utf-8")
            (root / "manifest.json").write_bytes(b"changed manifest")
            self.assertIn("manifest", _dispatch_integrity_error(job, plan_path, root))

    def test_ppo_paths_match_trainer_run_dir_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cfg = {
                "seed": 29,
                "paths": {"run_root": str(root / "runs")},
                "train": {"run_name": "authoritative-run-name", "budget": 5},
                "rl": {"run_name": "stale-rl-name", "condition_id": "R5", "budget": 5},
            }
            actual_run_dir = ensure_rl_run_dir(cfg)
            expected_checkpoint = _checkpoint_for_record({"kind": "ppo"}, cfg, root, "R5", 29)
            expected_csv = _expected_csv_path(cfg, "R5", 29)
            self.assertEqual(expected_checkpoint.parent, actual_run_dir)
            self.assertEqual(expected_csv.parent, actual_run_dir)
            self.assertEqual(expected_checkpoint.name, "best.pt")

    @unittest.skipUnless(
        (REPO / "artifacts/protocol/fss_manifests/fss_canonical66_v2/test.json").is_file(),
        "Install the protocol bundle to test a complete generated evaluation plan.",
    )
    def test_generated_outputs_are_isolated_from_training_and_each_other(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sam_rl").mkdir()
            (root / "sam_rl" / "evaluation.py").write_text("VALUE = 1\n", encoding="utf-8")
            (root / "pyproject.toml").write_text("[project]\nname='test'\n", encoding="utf-8")
            manifest_names = {
                "fss_test450": REPO
                / "artifacts/protocol/fss_manifests/fss_class_disjoint_canonical66_v2/test.json",
                "fss_canonical660": REPO
                / "artifacts/protocol/fss_manifests/fss_canonical66_v2/test.json",
                "kvasir_test450": REPO
                / "artifacts/protocol/kvasir_manifests/kvasir_seg_v2/test.json",
            }
            spec = json.loads((REPO / "configs/experiments/chair_v2.json").read_text())
            spec["evaluation_protocol"].update(
                {key: str(path) for key, path in manifest_names.items()}
            )
            spec_path = root / "chair_v2.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")

            records = {}
            for condition in CONDITIONS:
                for seed in SEEDS:
                    config_path = root / "training_configs" / f"{condition}-s{seed}.yaml"
                    config_path.parent.mkdir(parents=True, exist_ok=True)
                    config_path.write_text(
                        yaml.safe_dump(training_config(condition, seed, root)), encoding="utf-8"
                    )
                    records[f"{condition}-s{seed}"] = {
                        "job_id": f"{condition}-s{seed}",
                        "condition": condition,
                        "seed": seed,
                        "kind": "ppo" if condition.startswith("R") else "bc",
                        "status": "planned",
                        "config_path": str(config_path),
                        "config_sha256": "training-config-hash",
                    }
            registry_path = root / "registry.json"
            registry_path.write_text(json.dumps({"jobs": records}), encoding="utf-8")

            before_training_configs = {
                path: path.read_bytes() for path in root.glob("training_configs/*.yaml")
            }
            plan = generate_suite(
                spec_path=spec_path,
                registry_path=registry_path,
                repo_root=root,
                output_root=root / "artifacts/chair_v2",
                config_root=root / "configs/generated/evaluation",
            )
            jobs = plan["jobs"]
            self.assertEqual(len(jobs), 390)
            self.assertEqual(len({job["csv"] for job in jobs}), 390)
            self.assertEqual({job["status"] for job in jobs}, {"pending"})
            self.assertEqual(len(plan["collection"]["runs"]), 390)
            self.assertEqual(len({job["config_sha256"] for job in jobs}), 390)
            self.assertTrue(all(job["checkpoint_sha256"] is None for job in jobs))
            self.assertTrue(all(job["evaluation_config_sha256"] for job in jobs))
            self.assertTrue(all(job["evaluation_source_sha256"] for job in jobs))
            self.assertTrue(
                all(job["manifest_sha256"] and job["training_config_sha256"] for job in jobs)
            )

            cache_by_group = {}
            output_by_condition_seed = {}
            for job in jobs:
                cfg = yaml.safe_load((root / job["config"]).read_text(encoding="utf-8"))
                benchmark = cfg["experiment"]["benchmark"]
                self.assertEqual(cfg["dataset"]["episode_manifests"].keys(), {"test"})
                self.assertEqual(cfg["models"]["eval_sam_model_type"], job["decoder"])
                self.assertEqual(cfg["protocol"]["decoder_policy"], "single")
                self.assertEqual(cfg["models"]["precision"], "fp32")
                self.assertEqual(cfg["evaluation"]["final_mask_sample_count"], 10)
                self.assertFalse(cfg["experiment"]["kvasir_adaptation"])
                cache_key = (job["source_dataset"], job["decoder"])
                cache = cfg["models"]["sim_map_cache_dir"]
                self.assertEqual(cache_by_group.setdefault(cache_key, cache), cache)
                run_key = (job["condition"], job["training_seed"])
                output_root = cfg["paths"]["run_root"]
                output_by_condition_seed.setdefault(run_key, set()).add(output_root)
                expected_output_label = next(item[0] for item in BENCHMARKS if item[2] == benchmark)
                self.assertIn(expected_output_label, job["csv"])
                self.assertNotIn("/full/bc/", job["csv"])
                self.assertNotIn("/full/ppo/", job["csv"])
            for (condition, seed), roots in output_by_condition_seed.items():
                self.assertEqual(len(roots), 6, (condition, seed))
            self.assertEqual(
                {path: path.read_bytes() for path in root.glob("training_configs/*.yaml")},
                before_training_configs,
            )


if __name__ == "__main__":
    unittest.main()
