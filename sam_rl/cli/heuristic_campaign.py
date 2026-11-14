"""Plan and safely run the mask-aware point-selection comparison."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sam_rl.episode_manifest import episode_identity, load_episode_manifest
from sam_rl.evaluation_provenance import evaluation_source_fingerprint
from sam_rl.point_selectors import build_method
from sam_rl.provenance import atomic_write_json
from sam_rl.result_manifest import validate_result_shards

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = REPO_ROOT / "configs/experiments/chair_v2.json"
DEFAULT_CAMPAIGN_SPEC = REPO_ROOT / "configs/experiments/mask_aware_heuristic_v1.json"
DEFAULT_CAMPAIGN = REPO_ROOT / "artifacts/experiments/mask_aware_heuristic_v1_review_v3"
ALPHA_GRID = (0.25, 0.5, 0.75)
METHOD_SPECS = [
    {"family": "topk_argmax"},
    {"family": "mask_aware_greedy"},
    {"family": "sim_weighted_fps", "search": {"alpha": list(ALPHA_GRID)}},
]
WEIGHT_MANIFEST = REPO_ROOT / "weights.sha256"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    atomic_write_json(path, payload)


@contextmanager
def _locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _manifest_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _episode_keys(manifest: Path, data_root: str, repo_root: Path) -> list[str]:
    root = Path(data_root)
    if not root.is_absolute():
        root = repo_root / root
    episodes = load_episode_manifest(manifest, validate_paths=False, data_root=root)
    return [episode_identity(episode) for episode in episodes]


def _weight_hashes(manifest_path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
        fields = raw_line.split()
        if len(fields) == 2:
            hashes[Path(fields[1]).name] = fields[0].lower()
    required = {
        "sam_vit_b_01ec64.pth",
        "sam_vit_h_4b8939.pth",
        "dinov2_vitl14_pretrain.pth",
    }
    missing = required - hashes.keys()
    if missing:
        raise ValueError(f"weights.sha256 lacks required checkpoints: {sorted(missing)}")
    return {name: hashes[name] for name in sorted(required)}


def _verify_checkpoint_weights(
    job: dict[str, Any], ledger: dict[str, Any]
) -> dict[str, dict[str, str]]:
    weight_manifest = Path(ledger["weights_manifest"])
    if _sha256(weight_manifest) != ledger["weights_manifest_sha256"]:
        raise ValueError("Tracked weights.sha256 changed after campaign generation.")
    declared = _weight_hashes(weight_manifest)
    if declared != ledger["expected_model_weight_hashes"]:
        raise ValueError("Checkpoint hashes in weights.sha256 changed after campaign generation.")
    config = json.loads(Path(job["config"]).read_text(encoding="utf-8"))
    fingerprints = {}
    for field in ("dino_checkpoint", "sam_checkpoint"):
        configured = Path(config["models"][field])
        checkpoint = configured if configured.is_absolute() else REPO_ROOT / configured
        expected = declared.get(checkpoint.name)
        if expected is None:
            raise ValueError(f"No tracked checkpoint hash for {checkpoint.name}.")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Required model checkpoint is not installed: {checkpoint}")
        actual = _sha256(checkpoint)
        if actual != expected:
            raise ValueError(f"Checkpoint checksum mismatch for {checkpoint}.")
        fingerprints[field] = {
            "path": str(checkpoint.resolve()),
            "expected_sha256": expected,
            "actual_sha256": actual,
        }
    return fingerprints


def _dataset_specs(spec: dict[str, Any]) -> list[dict[str, Any]]:
    fss_test = spec["evaluation_protocol"]["fss_test450"]
    canonical = spec["evaluation_protocol"]["fss_canonical660"]
    kvasir = spec["evaluation_protocol"]["kvasir_test450"]
    return [
        {
            "key": "fss_test450",
            "name": "fss1000_test450",
            "kind": "fss1000",
            "manifest": fss_test,
            "split": "test",
            "root": "data/FSS-1000",
            "expected_count": int(spec["test_episodes"]),
        },
        {
            "key": "fss_canonical660",
            "name": "fss1000_canonical660",
            "kind": "fss1000",
            "manifest": canonical,
            "split": "canonical66_test",
            "root": "data/FSS-1000",
            "expected_count": int(spec["canonical_test_episodes"]),
        },
        {
            "key": "kvasir_test450",
            "name": "kvasir_seg_test450",
            "kind": "kvasir_seg",
            "manifest": kvasir,
            "split": "test",
            "root": str(spec["kvasir"]["root"]),
            "expected_count": int(spec["kvasir"]["test_episodes"]),
        },
    ]


def _eval_config(
    *,
    dataset: dict[str, Any],
    decoder: str,
    checkpoint: str,
    output_dir: Path,
    manifest_path: Path,
    condition: str,
    phase: str,
    best_params_path: Path | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_key = dataset["key"]
    config: dict[str, Any] = {
        "condition": condition,
        "dataset": {
            "kind": dataset["kind"],
            "name": dataset["name"],
            "split": dataset["split"],
            "episode_manifest": str(manifest_path),
            "root": dataset["root"],
            "data_root": dataset["root"],
        },
        "models": {
            "device": "cuda:0",
            "image_size": 224,
            "dino_checkpoint": "artifacts/weights/dinov2_vitl14_pretrain.pth",
            "sam_checkpoint": checkpoint,
            "sam_model_type": decoder,
            "train_precision": "fp32",
            "eval_precision": "fp32",
            "decoder_policy": "single",
        },
        "evaluation": {
            "point_budgets": [5],
            "seeds": [0],
            "deterministic_seeds": [0],
            "output_csv": str(output_dir / "results.csv"),
            "overlay_dir": str(output_dir / "overlays"),
            "overlay_limit": 0,
            "overlay_budgets": [],
            "sim_map_cache_dir": f"artifacts/point_baselines/cache/{cache_key}_224",
            "prompt_snap_grid": 37,
        },
        "selection": {"metric": "iou"},
        "methods": copy.deepcopy(METHOD_SPECS),
        "campaign": {
            "version": "mask_aware_heuristic_v1",
            "phase": phase,
            "dataset_key": dataset["key"],
            "decoder": decoder,
            "expected_episodes": dataset["expected_count"],
            "evaluation_seed": 0,
            "point_budget": 5,
            "prompt_snap_grid": 37,
            "precision": "fp32",
            "frozen_params": str(best_params_path) if best_params_path else None,
            "test_tuning": False,
        },
    }
    if best_params_path:
        config["selection"]["best_params_json"] = str(best_params_path)
    return config


def generate_campaign(
    output_dir: str | Path,
    *,
    spec_path: str | Path = DEFAULT_SPEC,
    campaign_spec_path: str | Path = DEFAULT_CAMPAIGN_SPEC,
    repo_root: str | Path = REPO_ROOT,
) -> dict[str, Any]:
    """Create immutable sweep/test configs and their provenance ledger."""
    campaign_dir = Path(output_dir).resolve()
    root = Path(repo_root).resolve()
    source_spec = _manifest_path(str(spec_path), root)
    campaign_spec_file = _manifest_path(str(campaign_spec_path), root)
    if campaign_dir.exists() and any(campaign_dir.iterdir()):
        raise FileExistsError(f"Campaign directory already contains data: {campaign_dir}")
    source = json.loads(source_spec.read_text(encoding="utf-8"))
    campaign_spec = json.loads(campaign_spec_file.read_text(encoding="utf-8"))
    if source.get("suite") != "chair_v2":
        raise ValueError("Heuristic campaign requires the chair_v2 experiment spec.")
    if campaign_spec.get("campaign") != "mask_aware_heuristic_v1":
        raise ValueError("Unexpected heuristic campaign spec.")
    protocol = campaign_spec["protocol"]
    if (
        tuple(campaign_spec["methods"][2]["parameter_grid"]["alpha"]) != ALPHA_GRID
        or protocol["point_budget"] != 5
        or protocol["prompt_snap_grid"] != 37
        or protocol["evaluation_seed"] != 0
        or protocol["precision"] != "fp32"
        or protocol["decoder_policy"] != "single"
        or protocol["test_tuning"] is not False
    ):
        raise ValueError("Campaign spec does not match the frozen implementation protocol.")
    campaign_dir.mkdir(parents=True, exist_ok=True)
    config_dir = campaign_dir / "configs"
    frozen_path = campaign_dir / "validation" / "best_params.json"
    jobs: dict[str, dict[str, Any]] = {}
    all_sources = {
        str(source_spec): _sha256(source_spec),
        str(campaign_spec_file): _sha256(campaign_spec_file),
    }
    eval_source = evaluation_source_fingerprint(root)

    validation_dataset = {
        "key": "fss_val357",
        "name": "fss1000_val357",
        "kind": "fss1000",
        "manifest": source["fss_episode_manifests"]["val"],
        "split": "val",
        "root": "data/FSS-1000",
        "expected_count": int(source["validation_episodes"]),
    }
    declared_validation = campaign_spec["validation"]
    if (
        declared_validation["episodes"] != validation_dataset["expected_count"]
        or declared_validation["sam_model"] != "vit_b"
        or declared_validation["split"] != validation_dataset["split"]
    ):
        raise ValueError("Campaign validation declaration disagrees with chair_v2.")
    expected_benchmarks = {
        item["id"]: (item["manifest"], item["episodes"], item["dataset"])
        for item in campaign_spec["held_out_benchmarks"]
    }
    actual_benchmarks = {
        item["key"]: (item["manifest"], item["expected_count"], item["kind"])
        for item in _dataset_specs(source)
    }
    if expected_benchmarks != actual_benchmarks:
        raise ValueError("Campaign benchmark declarations disagree with chair_v2.")
    validation_manifest = _manifest_path(validation_dataset["manifest"], root)
    validation_keys = _episode_keys(validation_manifest, validation_dataset["root"], root)
    if len(validation_keys) != validation_dataset["expected_count"]:
        raise ValueError(
            f"Validation manifest has {len(validation_keys)} episodes; expected {validation_dataset['expected_count']}."
        )
    val_output = campaign_dir / "validation"
    val_config_path = config_dir / "fss_val357_sam_vit_b.json"
    val_config = _eval_config(
        dataset=validation_dataset,
        decoder="vit_b",
        checkpoint="artifacts/weights/sam_vit_b_01ec64.pth",
        output_dir=val_output,
        manifest_path=Path(validation_dataset["manifest"]),
        condition="mask_aware_heuristic_validation_v1",
        phase="validation",
        best_params_path=frozen_path,
    )
    # Sweep has no frozen parameters; the configured location is the sweep's output.
    val_config["selection"].pop("best_params_json", None)
    val_config["campaign"]["frozen_params"] = None
    _atomic_json(val_config_path, val_config)
    val_methods = [build_method("topk_argmax").method, build_method("mask_aware_greedy").method]
    val_methods += [
        build_method("sim_weighted_fps", {"alpha": alpha}).method for alpha in ALPHA_GRID
    ]
    val_job = _job_record(
        job_id="sweep_fss_val357_sam_vit_b",
        phase="validation",
        config_path=val_config_path,
        config=val_config,
        expected_methods=val_methods,
        expected_keys=validation_keys,
        manifest=validation_manifest,
        output_csv=Path(val_config["evaluation"]["output_csv"]),
        frozen_params=frozen_path,
        dependencies=[],
        source_hashes=all_sources,
    )
    jobs[val_job["job_id"]] = val_job

    for dataset in _dataset_specs(source):
        manifest_path = _manifest_path(str(dataset["manifest"]), root)
        keys = _episode_keys(manifest_path, dataset["root"], root)
        if len(keys) != dataset["expected_count"]:
            raise ValueError(
                f"{dataset['key']} manifest has {len(keys)} episodes; expected {dataset['expected_count']}."
            )
        for decoder in ("vit_b", "vit_h"):
            checkpoint = (
                "artifacts/weights/sam_vit_b_01ec64.pth"
                if decoder == "vit_b"
                else "artifacts/weights/sam_vit_h_4b8939.pth"
            )
            job_id = f"frozen_{dataset['key']}_sam_{decoder}"
            result_dir = campaign_dir / "test" / dataset["key"] / decoder
            config_path = config_dir / f"{dataset['key']}_sam_{decoder}.json"
            config = _eval_config(
                dataset=dataset,
                decoder=decoder,
                checkpoint=checkpoint,
                output_dir=result_dir,
                manifest_path=Path(dataset["manifest"]),
                condition=f"mask_aware_heuristic_{dataset['key']}_{decoder}",
                phase="test",
                best_params_path=frozen_path,
            )
            _atomic_json(config_path, config)
            jobs[job_id] = _job_record(
                job_id=job_id,
                phase="test",
                config_path=config_path,
                config=config,
                expected_methods=["topk_argmax", "mask_aware_greedy", "sim_weighted_fps"],
                expected_keys=keys,
                manifest=manifest_path,
                output_csv=Path(config["evaluation"]["output_csv"]),
                frozen_params=frozen_path,
                dependencies=[val_job["job_id"]],
                source_hashes=all_sources,
            )

    ledger = {
        "schema_version": 1,
        "campaign": "mask_aware_heuristic_v1",
        "created_from_spec": str(source_spec),
        "created_from_campaign_spec": str(campaign_spec_file),
        "source_spec_sha256": _sha256(source_spec),
        "protocol": {
            "methods": METHOD_SPECS,
            "sim_weighted_fps_alpha_grid": list(ALPHA_GRID),
            "point_budget": 5,
            "prompt_snap_grid": 37,
            "evaluation_seed": 0,
            "precision": "fp32",
            "decoder_policy": "single",
            "selection_dataset": "fss_val357",
            "test_tuning": False,
        },
        "source_hashes": all_sources,
        "evaluation_source_fingerprint": eval_source,
        "weights_manifest": str(WEIGHT_MANIFEST.resolve()),
        "weights_manifest_sha256": _sha256(WEIGHT_MANIFEST),
        "expected_model_weight_hashes": _weight_hashes(WEIGHT_MANIFEST),
        "jobs": jobs,
    }
    _atomic_json(campaign_dir / "ledger.json", ledger)
    return ledger


def _job_record(
    *,
    job_id: str,
    phase: str,
    config_path: Path,
    config: dict[str, Any],
    expected_methods: list[str],
    expected_keys: list[str],
    manifest: Path,
    output_csv: Path,
    frozen_params: Path,
    dependencies: list[str],
    source_hashes: dict[str, str],
) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "phase": phase,
        "status": "planned",
        "dependencies": dependencies,
        "config": str(config_path.resolve()),
        "config_sha256": _sha256(config_path),
        "manifest": str(manifest.resolve()),
        "manifest_sha256": _sha256(manifest),
        "expected_episode_count": len(expected_keys),
        "expected_episode_keys": expected_keys,
        "expected_methods": expected_methods,
        "expected_budgets": [5],
        "expected_seeds": {method: [0] for method in expected_methods},
        "source_hashes": source_hashes,
        "frozen_params": str(frozen_params.resolve()),
        "frozen_params_sha256": None,
        "output_csv": str(output_csv.resolve()),
        "metadata_json": str(output_csv.with_suffix(".metadata.json").resolve()),
        "summary_csv": str(output_csv.with_name("summary.csv").resolve()),
        "status_detail": None,
        "command": None,
        "gpu": None,
    }


def _load_ledger(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _strict_expected(job: dict[str, Any], mode: str) -> tuple[list[str], dict[str, list[int]]]:
    methods = list(job["expected_methods"])
    seeds = dict(job["expected_seeds"])
    if mode == "frozen":
        payload = json.loads(Path(job["frozen_params"]).read_text(encoding="utf-8"))
        methods = [
            build_method(str(item["family"]), dict(item.get("params", {}))).method
            for item in payload["best_methods"]
        ]
        seeds = {method: [0] for method in methods}
    return methods, seeds


def run_job(ledger_path: str | Path, job_id: str, gpu: int) -> int:
    """Run one planned job on an explicitly assigned physical GPU."""
    if int(gpu) < 0:
        raise ValueError("Physical GPU ID must be nonnegative.")
    ledger_path = Path(ledger_path).resolve()
    lock_path = ledger_path.with_suffix(ledger_path.suffix + ".lock")
    with _locked(lock_path):
        ledger = _load_ledger(ledger_path)
        if job_id not in ledger["jobs"]:
            raise KeyError(f"Unknown heuristic campaign job: {job_id}")
        job = ledger["jobs"][job_id]
        if job["status"] != "planned":
            raise RuntimeError(
                f"Job {job_id} is {job['status']}; planned jobs only are launchable."
            )
        for dep_id in job["dependencies"]:
            if ledger["jobs"][dep_id]["status"] != "completed":
                raise RuntimeError(f"Job {job_id} dependency {dep_id} is not completed.")
        config_path = Path(job["config"])
        if _sha256(config_path) != job["config_sha256"]:
            raise ValueError(f"Generated config changed after planning: {config_path}")
        if _sha256(Path(job["manifest"])) != job["manifest_sha256"]:
            raise ValueError(f"Episode manifest changed after planning: {job['manifest']}")
        for source_name, expected_hash in ledger["source_hashes"].items():
            source_path = Path(source_name)
            if _sha256(source_path) != expected_hash:
                raise ValueError(f"Campaign source changed after planning: {source_name}")
        current_fingerprint = evaluation_source_fingerprint(REPO_ROOT)
        if current_fingerprint != ledger["evaluation_source_fingerprint"]:
            raise ValueError("Evaluation source or dependency fingerprint changed after planning.")
        for path_key in ("output_csv", "metadata_json", "summary_csv"):
            if Path(job[path_key]).exists():
                raise FileExistsError(f"Refusing to overwrite campaign artifact {job[path_key]}")
        if job["phase"] == "validation" and Path(job["frozen_params"]).exists():
            raise FileExistsError(
                f"Refusing to overwrite frozen-parameter file {job['frozen_params']}"
            )
        if job["phase"] == "test" and not Path(job["frozen_params"]).is_file():
            raise FileNotFoundError(
                f"Missing validation-frozen parameter file {job['frozen_params']}"
            )
        if job["phase"] == "test":
            actual_frozen_hash = _sha256(Path(job["frozen_params"]))
            if job["frozen_params_sha256"] != actual_frozen_hash:
                raise ValueError(
                    "Frozen validation parameters changed after the sweep was completed."
                )
        checkpoint_fingerprints = _verify_checkpoint_weights(job, ledger)
        job["status"] = "running"
        job["gpu"] = int(gpu)
        job["checkpoint_fingerprints"] = checkpoint_fingerprints
        job["command"] = _command(job, config_path)
        job["status_detail"] = "started"
        _atomic_json(ledger_path, ledger)

    command = list(job["command"])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(int(gpu))
    log_path = ledger_path.parent / "logs" / f"{job_id}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("x", encoding="utf-8") as log:
            process = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, command)
        _attach_checkpoint_fingerprints(job)
        mode = "sweep" if job["phase"] == "validation" else "frozen"
        methods, seeds = _strict_expected(job, mode)
        validate_result_shards(
            [job["output_csv"]],
            expected_keys=job["expected_episode_keys"],
            expected_methods=methods,
            budgets=job["expected_budgets"],
            seeds=seeds,
            metadata_paths=[job["metadata_json"]],
        )
        _write_expected_manifest(job, methods, seeds)
        _write_summary(job)
        _finish_job(ledger_path, job_id, "completed", f"validated rows; log={log_path}")
        return 0
    except Exception as exc:
        _finish_job(ledger_path, job_id, "failed", f"{type(exc).__name__}: {exc}; log={log_path}")
        raise


def _command(job: dict[str, Any], config_path: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "sam_rl.cli.eval_point_baselines",
        "--config",
        str(config_path),
    ]
    if job["phase"] == "validation":
        command.extend(["--mode", "sweep", "--best-params-json", str(job["frozen_params"])])
    else:
        command.extend(["--mode", "frozen", "--frozen-params", str(job["frozen_params"])])
    return command


def _write_expected_manifest(
    job: dict[str, Any], methods: list[str], seeds: dict[str, list[int]]
) -> Path:
    path = Path(job["output_csv"]).with_name("expected_results.json")
    _atomic_json(
        path,
        {
            "expected_episode_keys": job["expected_episode_keys"],
            "methods": methods,
            "budgets": job["expected_budgets"],
            "seeds": seeds,
        },
    )
    return path


def _attach_checkpoint_fingerprints(job: dict[str, Any]) -> None:
    metadata_path = Path(job["metadata_json"])
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Evaluation did not produce result metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["checkpoint_fingerprints"] = job["checkpoint_fingerprints"]
    _atomic_json(metadata_path, metadata)


def _write_summary(job: dict[str, Any]) -> None:
    expected = Path(job["output_csv"]).with_name("expected_results.json")
    summary = Path(job["summary_csv"])
    if summary.exists():
        raise FileExistsError(summary)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "sam_rl.cli.summarize_method_results",
            "--inputs",
            str(job["output_csv"]),
            "--output",
            str(summary),
            "--expected-manifest",
            str(expected),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def _finish_job(ledger_path: Path, job_id: str, status: str, detail: str) -> None:
    with _locked(ledger_path.with_suffix(ledger_path.suffix + ".lock")):
        ledger = _load_ledger(ledger_path)
        job = ledger["jobs"][job_id]
        job["status"] = status
        job["status_detail"] = detail
        if job["phase"] == "validation" and status == "completed":
            frozen_hash = _sha256(Path(job["frozen_params"]))
            job["frozen_params_sha256"] = frozen_hash
            for dependent in ledger["jobs"].values():
                if job_id in dependent["dependencies"]:
                    dependent["frozen_params_sha256"] = frozen_hash
        _atomic_json(ledger_path, ledger)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    generate = sub.add_parser("generate", help="Create reviewable immutable configs and ledger.")
    generate.add_argument("--output-dir", default=str(DEFAULT_CAMPAIGN))
    generate.add_argument("--spec", default=str(DEFAULT_SPEC))
    generate.add_argument("--campaign-spec", default=str(DEFAULT_CAMPAIGN_SPEC))
    run = sub.add_parser("run", help="Run one planned job on an explicitly assigned GPU.")
    run.add_argument("--ledger", required=True)
    run.add_argument("--job-id", required=True)
    run.add_argument("--gpu", required=True, type=int, help="Physical GPU index on this host.")
    args = parser.parse_args()
    if args.action == "generate":
        ledger = generate_campaign(
            args.output_dir, spec_path=args.spec, campaign_spec_path=args.campaign_spec
        )
        print(f"Generated {len(ledger['jobs'])} planned jobs in {args.output_dir}")
    else:
        run_job(args.ledger, args.job_id, args.gpu)


if __name__ == "__main__":
    main()
