from __future__ import annotations

import hashlib
import importlib.machinery
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from sam_rl.cli.evaluation_suite import BENCHMARKS, CONDITIONS, DECODERS, SEEDS
from sam_rl.cli.merge_evaluation_collections import merge_collections
from sam_rl.evaluation_provenance import evaluation_config_sha256, evaluation_source_fingerprint
from sam_rl.result_manifest import manifest_hash


def _collection(seeds: tuple[int, ...], *, status="pending"):
    checkpoint_hash = "a" * 64
    config_hash = "b" * 64
    source_hash = "c" * 64
    runs = []
    for condition in CONDITIONS:
        for seed in seeds:
            for _label, dataset, protocol, _manifest in BENCHMARKS:
                for decoder in DECODERS:
                    stem = f"{condition}-s{seed}-{protocol}-{decoder}"
                    runs.append(
                        {
                            "condition": condition,
                            "training_seed": seed,
                            "eval_seed": 0,
                            "dataset": dataset,
                            "split": "test",
                            "decoder": decoder,
                            "protocol": f"single|{protocol}",
                            "status": status,
                            "reason": "not dispatched" if status != "complete" else None,
                            "csv": f"artifacts/evaluation_runs/{stem}.csv",
                            "per_step_csv": f"artifacts/evaluation_runs/{stem}.per_step.csv",
                            "metadata": f"artifacts/evaluation_runs/{stem}.metadata.json",
                            "expected_episode_count": 450,
                            "expected_result_rows": 450,
                            "mask_sample_count": 0,
                            "mask_sample_manifest": (
                                f"artifacts/evaluation_runs/{stem}.mask_samples/sample_manifest.json"
                            ),
                            "checkpoint_sha256": checkpoint_hash,
                            "evaluation_config_sha256": config_hash,
                            "evaluation_source_sha256": source_hash,
                        }
                    )
    return {
        "expected_conditions": list(CONDITIONS),
        "expected_training_seeds": list(seeds),
        "expected_eval_seeds": [0],
        "bootstrap_resamples": 5000,
        "bootstrap_seed": 731,
        "evaluation_source_sha256": source_hash,
        "episode_clusters": {"fss1000": {"episode": "class:one"}},
        "runs": runs,
    }


def _write_source(root: Path, name: str, seeds: tuple[int, ...]):
    path = root / name / "collection.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_collection(seeds)), encoding="utf-8")
    return path


class EvaluationCollectionMergeTests(unittest.TestCase):
    def _spec(self, root: Path, source_rows):
        spec_path = root / "merge_spec.json"
        spec_path.write_text(json.dumps({"sources": source_rows}), encoding="utf-8")
        return spec_path

    def test_merges_exact_390_matrix_and_rewrites_paths_from_declared_staging_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_source(root, "host-a", (17, 29, 43))
            _write_source(root, "host-b", (71, 101))
            spec = self._spec(
                root,
                [
                    {
                        "collection": "host-a/collection.json",
                        "source_root": "/remote/host-a/repo",
                        "staged_artifact_root": "artifacts/host-a",
                    },
                    {
                        "collection": "host-b/collection.json",
                        "source_root": "/remote/host-b/repo",
                        "staged_artifact_root": "artifacts/host-b",
                    },
                ],
            )
            output = root / "staged" / "collection.json"
            merged = merge_collections(spec, output, allow_incomplete=True)
            self.assertEqual(len(merged["runs"]), 390)
            self.assertEqual(merged["expected_training_seeds"], list(SEEDS))
            self.assertEqual(merged["episode_clusters"], {"fss1000": {"episode": "class:one"}})
            host_a_run = next(run for run in merged["runs"] if run["training_seed"] == 17)
            self.assertTrue(host_a_run["csv"].startswith("artifacts/host-a/"))
            self.assertTrue(host_a_run["mask_sample_manifest"].startswith("artifacts/host-a/"))
            self.assertEqual(host_a_run["source_root"], "/remote/host-a/repo")
            self.assertEqual(
                len(
                    {
                        (
                            r["condition"],
                            r["training_seed"],
                            r["eval_seed"],
                            r["dataset"],
                            r["split"],
                            r["decoder"],
                            r["protocol"],
                        )
                        for r in merged["runs"]
                    }
                ),
                390,
            )

    def test_incomplete_status_requires_explicit_flag_and_missing_matrix_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _write_source(root, "host-a", (17, 29, 43))
            other = _write_source(root, "host-b", (71, 101))
            spec = self._spec(
                root,
                [
                    {
                        "collection": str(source),
                        "source_root": "/host-a",
                        "staged_artifact_root": "host-a",
                    },
                    {
                        "collection": str(other),
                        "source_root": "/host-b",
                        "staged_artifact_root": "host-b",
                    },
                ],
            )
            with self.assertRaisesRegex(ValueError, "allow-incomplete"):
                merge_collections(spec, root / "out" / "collection.json")
            collection = json.loads(other.read_text(encoding="utf-8"))
            collection["runs"].pop()
            other.write_text(json.dumps(collection), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing"):
                merge_collections(spec, root / "out" / "collection.json", allow_incomplete=True)

    def test_complete_metadata_is_checked_after_path_rewrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_a = _collection((17, 29, 43))
            source_b = _collection((71, 101))
            first_run = source_a["runs"][0]
            first_run["status"] = "complete"
            first_run["expected_episode_keys"] = ["episode-a", "episode-b"]
            first_run["expected_episode_count"] = 2
            first_run["expected_result_rows"] = 2
            first_run["mask_sample_count"] = 1
            for name, payload in (("host-a", source_a), ("host-b", source_b)):
                folder = root / name
                folder.mkdir()
                (folder / "collection.json").write_text(json.dumps(payload), encoding="utf-8")
            spec = self._spec(
                root,
                [
                    {
                        "collection": "host-a/collection.json",
                        "source_root": "/host-a",
                        "staged_artifact_root": "artifacts/host-a",
                    },
                    {
                        "collection": "host-b/collection.json",
                        "source_root": "/host-b",
                        "staged_artifact_root": "artifacts/host-b",
                    },
                ],
            )
            output_base = root / "staged"
            staged = output_base / "artifacts/host-a/artifacts/evaluation_runs"
            staged.mkdir(parents=True)
            csv_path = staged / Path(first_run["csv"]).name
            csv_path.write_text("episode_key,iou\nepisode-a,0.5\nepisode-b,0.6\n", encoding="utf-8")
            metadata_path = staged / Path(first_run["metadata"]).name
            metadata = {
                "dataset": first_run["dataset"],
                "split": first_run["split"],
                "condition": first_run["condition"],
                "decoder": first_run["decoder"],
                "training_seed": first_run["training_seed"],
                "eval_seed": first_run["eval_seed"],
                "protocol": first_run["protocol"],
                "manifest_hash": manifest_hash(["episode-a", "episode-b"]),
                "expected_episode_keys": ["episode-a", "episode-b"],
                "expected_episode_count": 2,
                "expected_result_rows": 2,
                "checkpoint_sha256": first_run["checkpoint_sha256"],
                "evaluation_config_sha256": first_run["evaluation_config_sha256"],
                "evaluation_source_sha256": first_run["evaluation_source_sha256"],
            }
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            mask_dir = csv_path.with_suffix(".mask_samples")
            mask_dir.mkdir()
            mask_path = mask_dir / "mask_001.png"
            Image.new("L", (2, 2), color=0).save(mask_path, format="PNG")
            metadata.update(
                {
                    "model_config_hash": "1" * 64,
                    "model_hash": "2" * 64,
                }
            )
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            mask_manifest = {
                "sampling_rule": "first_n_in_ordered_manifest_v1",
                "sample_count": 1,
                "sampled_episode_keys": ["episode-a"],
                "result_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                "dataset": metadata["dataset"],
                "split": metadata["split"],
                "condition": metadata["condition"],
                "training_seed": metadata["training_seed"],
                "eval_seed": metadata["eval_seed"],
                "decoder": metadata["decoder"],
                "protocol": metadata["protocol"],
                "manifest_hash": metadata["manifest_hash"],
                "model_config_hash": metadata["model_config_hash"],
                "model_hash": metadata["model_hash"],
                "checkpoint_sha256": metadata["checkpoint_sha256"],
                "evaluation_config_sha256": metadata["evaluation_config_sha256"],
                "evaluation_source_sha256": metadata["evaluation_source_sha256"],
                "samples": [
                    {
                        "sample_index": 1,
                        "episode_key": "episode-a",
                        "mask_png": "mask_001.png",
                        "width": 2,
                        "height": 2,
                        "mask_sha256": hashlib.sha256(mask_path.read_bytes()).hexdigest(),
                    }
                ],
            }
            (mask_dir / "sample_manifest.json").write_text(
                json.dumps(mask_manifest), encoding="utf-8"
            )
            merged = merge_collections(spec, output_base / "collection.json", allow_incomplete=True)
            completed = next(run for run in merged["runs"] if run["status"] == "complete")
            self.assertEqual(completed["csv"], "artifacts/host-a/" + first_run["csv"])
            self.assertTrue(completed["mask_sample_manifest"].startswith("artifacts/host-a/"))

            # Older collections may omit both count fields. The validated
            # metadata episode keys remain the source of the count.
            legacy_collection = json.loads(
                (root / "host-a/collection.json").read_text(encoding="utf-8")
            )
            legacy_run = legacy_collection["runs"][0]
            legacy_run.pop("expected_episode_count")
            legacy_run.pop("expected_result_rows")
            (root / "host-a/collection.json").write_text(
                json.dumps(legacy_collection), encoding="utf-8"
            )
            legacy_merged = merge_collections(
                spec, output_base / "collection.json", allow_incomplete=True
            )
            legacy_completed = next(
                run for run in legacy_merged["runs"] if run["status"] == "complete"
            )
            self.assertEqual(legacy_completed["expected_episode_count"], 2)
            self.assertEqual(legacy_completed["expected_result_rows"], 2)
            preserved_source = json.loads(
                (root / "host-a/collection.json").read_text(encoding="utf-8")
            )["runs"][0]
            self.assertNotIn("expected_episode_count", preserved_source)
            self.assertNotIn("expected_result_rows", preserved_source)

            csv_path.write_text("episode_key,iou\nepisode-a,0.5\nepisode-a,0.6\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing, duplicate, or unexpected episodes"):
                merge_collections(spec, output_base / "collection.json", allow_incomplete=True)
            csv_path.write_text("episode_key,iou\nepisode-a,0.5\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing, duplicate, or unexpected episodes"):
                merge_collections(spec, output_base / "collection.json", allow_incomplete=True)
            csv_path.write_text("episode_key,iou\nepisode-a,0.5\nepisode-b,0.6\n", encoding="utf-8")

            metadata["expected_episode_count"] = 3
            metadata["expected_result_rows"] = 2
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected_episode_count"):
                merge_collections(spec, output_base / "collection.json", allow_incomplete=True)
            metadata["expected_episode_count"] = 2
            metadata["expected_result_rows"] = 3
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected_result_rows"):
                merge_collections(spec, output_base / "collection.json", allow_incomplete=True)

            metadata["expected_result_rows"] = 2
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
            legacy_run["expected_episode_count"] = 3
            (root / "host-a/collection.json").write_text(
                json.dumps(legacy_collection), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "expected_episode_count.*collection"):
                merge_collections(spec, output_base / "collection.json", allow_incomplete=True)

            legacy_run["expected_episode_count"] = 2
            (root / "host-a/collection.json").write_text(
                json.dumps(legacy_collection), encoding="utf-8"
            )
            legacy_run["expected_result_rows"] = 3
            (root / "host-a/collection.json").write_text(
                json.dumps(legacy_collection), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "expected_result_rows.*collection"):
                merge_collections(spec, output_base / "collection.json", allow_incomplete=True)

            mask_path.unlink()
            legacy_run["expected_result_rows"] = 2
            (root / "host-a/collection.json").write_text(
                json.dumps(legacy_collection), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Mask PNG is missing"):
                merge_collections(spec, output_base / "collection.json", allow_incomplete=True)

    def test_source_paths_cannot_escape_explicit_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = _collection((17, 29, 43))
            source["runs"][0]["csv"] = "../outside.csv"
            path = root / "source.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            second = _write_source(root, "host-b", (71, 101))
            spec = self._spec(
                root,
                [
                    {
                        "collection": "source.json",
                        "source_root": "/host-a",
                        "staged_artifact_root": "host-a",
                    },
                    {
                        "collection": str(second),
                        "source_root": "/host-b",
                        "staged_artifact_root": "host-b",
                    },
                ],
            )
            with self.assertRaisesRegex(ValueError, "escapes declared source_root"):
                merge_collections(spec, root / "merged" / "collection.json", allow_incomplete=True)

    def test_mixed_evaluator_fingerprints_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = _write_source(root, "host-a", (17, 29, 43))
            second = _write_source(root, "host-b", (71, 101))
            payload = json.loads(second.read_text(encoding="utf-8"))
            payload["evaluation_source_sha256"] = "d" * 64
            for run in payload["runs"]:
                run["evaluation_source_sha256"] = "d" * 64
            second.write_text(json.dumps(payload), encoding="utf-8")
            spec = self._spec(
                root,
                [
                    {
                        "collection": str(first),
                        "source_root": "/host-a",
                        "staged_artifact_root": "a",
                    },
                    {
                        "collection": str(second),
                        "source_root": "/host-b",
                        "staged_artifact_root": "b",
                    },
                ],
            )
            with self.assertRaisesRegex(ValueError, "fingerprint differs"):
                merge_collections(spec, root / "out" / "collection.json", allow_incomplete=True)

    def test_completed_sidecar_must_match_checkpoint_and_evaluation_config_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_a = _collection((17, 29, 43))
            source_b = _collection((71, 101))
            first_run = source_a["runs"][0]
            first_run["status"] = "complete"
            first_run["expected_episode_keys"] = ["episode-a", "episode-b"]
            first_run["expected_episode_count"] = 2
            first_run["expected_result_rows"] = 2
            for name, payload in (("host-a", source_a), ("host-b", source_b)):
                folder = root / name
                folder.mkdir()
                (folder / "collection.json").write_text(json.dumps(payload), encoding="utf-8")
            spec = self._spec(
                root,
                [
                    {
                        "collection": "host-a/collection.json",
                        "source_root": "/host-a",
                        "staged_artifact_root": "artifacts/host-a",
                    },
                    {
                        "collection": "host-b/collection.json",
                        "source_root": "/host-b",
                        "staged_artifact_root": "artifacts/host-b",
                    },
                ],
            )
            staged = root / "out" / "artifacts/host-a/artifacts/evaluation_runs"
            staged.mkdir(parents=True)
            csv_path = staged / Path(first_run["csv"]).name
            csv_path.write_text("episode_key,iou\nepisode-a,0.5\nepisode-b,0.6\n", encoding="utf-8")
            metadata = {
                "dataset": first_run["dataset"],
                "split": first_run["split"],
                "condition": first_run["condition"],
                "decoder": first_run["decoder"],
                "training_seed": first_run["training_seed"],
                "eval_seed": first_run["eval_seed"],
                "protocol": first_run["protocol"],
                "manifest_hash": manifest_hash(["episode-a", "episode-b"]),
                "expected_episode_keys": ["episode-a", "episode-b"],
                "expected_episode_count": 2,
                "expected_result_rows": 2,
                "checkpoint_sha256": "e" * 64,
                "evaluation_config_sha256": first_run["evaluation_config_sha256"],
                "evaluation_source_sha256": first_run["evaluation_source_sha256"],
            }
            csv_path.with_suffix(".metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "disagrees on checkpoint_sha256"):
                merge_collections(spec, root / "out" / "collection.json", allow_incomplete=True)
            metadata["checkpoint_sha256"] = first_run["checkpoint_sha256"]
            metadata["evaluation_config_sha256"] = "f" * 64
            csv_path.with_suffix(".metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "disagrees on evaluation_config_sha256"):
                merge_collections(spec, root / "out" / "collection.json", allow_incomplete=True)
            metadata["evaluation_config_sha256"] = first_run["evaluation_config_sha256"]
            metadata["evaluation_source_sha256"] = "e" * 64
            csv_path.with_suffix(".metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "disagrees on evaluation_source_sha256"):
                merge_collections(spec, root / "out" / "collection.json", allow_incomplete=True)


class EvaluationProvenanceTests(unittest.TestCase):
    def test_config_fingerprint_is_key_order_independent(self):
        self.assertEqual(
            evaluation_config_sha256({"a": 1, "b": 2}),
            evaluation_config_sha256({"b": 2, "a": 1}),
        )

    def test_source_fingerprint_is_path_independent_and_tracks_module_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = Path(temporary) / "one"
            second = Path(temporary) / "two"
            for root in (first, second):
                (root / "sam_rl").mkdir(parents=True)
                (root / "sam_rl" / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
                (root / "pyproject.toml").write_text(
                    '[project]\ndependencies = ["segment-anything @ '
                    "git+https://github.com/facebookresearch/segment-anything.git@"
                    'dca509fe793f601edb92606367a655c15ac00fdf"]\n',
                    encoding="utf-8",
                )
                (root / "uv.lock").write_text(
                    'source = { git = "https://github.com/facebookresearch/segment-anything", '
                    'rev = "dca509fe793f601edb92606367a655c15ac00fdf" }\n',
                    encoding="utf-8",
                )
            self.assertEqual(
                evaluation_source_fingerprint(first)["evaluation_source_sha256"],
                evaluation_source_fingerprint(second)["evaluation_source_sha256"],
            )
            (second / "sam_rl" / "helper.py").write_text("VALUE = 2\n", encoding="utf-8")
            self.assertNotEqual(
                evaluation_source_fingerprint(first)["evaluation_source_sha256"],
                evaluation_source_fingerprint(second)["evaluation_source_sha256"],
            )
            (second / "uv.lock").write_text(
                'source = { git = "https://github.com/facebookresearch/segment-anything", '
                'rev = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" }\n',
                encoding="utf-8",
            )
            self.assertNotEqual(
                evaluation_source_fingerprint(first)["evaluation_source_sha256"],
                evaluation_source_fingerprint(second)["evaluation_source_sha256"],
            )

    def test_source_fingerprint_tracks_installed_segment_anything_without_absolute_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "project"
            install_one = Path(temporary) / "one" / "segment_anything"
            install_two = Path(temporary) / "two" / "segment_anything"
            for install in (install_one, install_two):
                (root / "sam_rl").mkdir(parents=True, exist_ok=True)
                install.mkdir(parents=True)
                (root / "sam_rl" / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
                (root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
                (root / "uv.lock").write_text("revision = 'same-pin'\n", encoding="utf-8")
                (install / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
                (install / "predictor.py").write_text("VALUE = 1\n", encoding="utf-8")

            def module_spec(location):
                spec = importlib.machinery.ModuleSpec(
                    "segment_anything", loader=None, is_package=True
                )
                spec.submodule_search_locations = [str(location)]
                return spec

            with patch(
                "sam_rl.evaluation_provenance.importlib.util.find_spec",
                side_effect=[module_spec(install_one), module_spec(install_two)],
            ):
                first = evaluation_source_fingerprint(root)
                second = evaluation_source_fingerprint(root)
            self.assertEqual(first["evaluation_source_sha256"], second["evaluation_source_sha256"])
            (install_two / "predictor.py").write_text("VALUE = 2\n", encoding="utf-8")
            with patch(
                "sam_rl.evaluation_provenance.importlib.util.find_spec",
                side_effect=[module_spec(install_one), module_spec(install_two)],
            ):
                first = evaluation_source_fingerprint(root)
                second = evaluation_source_fingerprint(root)
            self.assertNotEqual(
                first["evaluation_source_sha256"], second["evaluation_source_sha256"]
            )
            self.assertIn(
                "external/segment_anything/segment_anything/predictor.py",
                first["evaluation_source_files"],
            )
            self.assertTrue(
                all(not Path(name).is_absolute() for name in first["evaluation_source_files"])
            )

    def test_source_fingerprint_records_missing_segment_anything_deterministically(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sam_rl").mkdir()
            (root / "sam_rl" / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
            with patch("sam_rl.evaluation_provenance.importlib.util.find_spec", return_value=None):
                first = evaluation_source_fingerprint(root)
            with patch("sam_rl.evaluation_provenance.importlib.util.find_spec", return_value=None):
                second = evaluation_source_fingerprint(root)
            self.assertEqual(first, second)
            self.assertIn("external/segment_anything/MISSING", first["evaluation_source_files"])


if __name__ == "__main__":
    unittest.main()
