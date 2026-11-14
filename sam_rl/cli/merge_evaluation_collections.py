"""Merge staged, seed-partitioned chair_v2 evaluation collections."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, cast

from sam_rl.cli.evaluation_suite import BENCHMARKS, CONDITIONS, DECODERS, SEEDS
from sam_rl.mask_samples import mask_sample_dir_for_csv, validate_final_mask_samples
from sam_rl.provenance import atomic_write_json
from sam_rl.result_manifest import manifest_hash


def merge_collections(
    merge_spec_path: str | Path,
    merged_collection_path: str | Path,
    *,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Merge host ledgers, mapping each source repository tree into staged artifacts.

    Merge spec paths are explicit. ``source_root`` is the path base used by the
    source collection's relative run paths. ``staged_artifact_root`` is the
    corresponding directory under the merged collection's parent, where the
    root process has staged that source's artifacts.
    """
    spec_path = Path(merge_spec_path).resolve()
    output_path = Path(merged_collection_path).resolve()
    output_base = output_path.parent
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    sources_value = spec.get("sources")
    if not isinstance(sources_value, list) or not sources_value:
        raise ValueError("Merge spec must contain a nonempty sources list.")
    sources: list[Any] = sources_value

    ledgers: list[dict[str, Any]] = []
    common: dict[str, Any] | None = None
    seed_union: set[int] = set()
    identity_maps: dict[str, dict[str, Any]] = {
        "episode_clusters": {},
        "episode_image_ids": {},
    }
    evaluation_source_sha256: str | None = None
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(f"Merge source {index} must be an object.")
        source = cast(dict[str, Any], source)
        collection_file = _resolve(spec_path.parent, source.get("collection"), "collection")
        source_root = _resolve(spec_path.parent, source.get("source_root"), "source_root")
        artifact_root = _safe_relative(source.get("staged_artifact_root"), "staged_artifact_root")
        collection = json.loads(collection_file.read_text(encoding="utf-8"))
        header = _collection_header(collection)
        source_hash = header.pop("evaluation_source_sha256")
        if evaluation_source_sha256 is None:
            evaluation_source_sha256 = source_hash
        elif source_hash != evaluation_source_sha256:
            raise ValueError(
                f"Evaluation source fingerprint differs in {collection_file}; "
                "host collections must use identical evaluator source."
            )
        source_seeds = set(map(int, header.pop("expected_training_seeds")))
        if seed_union.intersection(source_seeds):
            raise ValueError(f"Overlapping training seed subsets in {collection_file}.")
        seed_union.update(source_seeds)
        for field in identity_maps:
            _merge_identity_map(identity_maps[field], collection.get(field, {}), field)
        if common is None:
            common = header
        elif header != common:
            raise ValueError(f"Collection header differs in {collection_file}.")
        for item in collection.get("runs", []):
            run = dict(item)
            if run.get("evaluation_source_sha256") != source_hash:
                raise ValueError(
                    f"Run evaluator source fingerprint disagrees with collection in "
                    f"{collection_file}: {run.get('evaluation_source_sha256')!r}."
                )
            if int(run.get("training_seed", -1)) not in source_seeds:
                raise ValueError(f"Run seed is not declared by {collection_file}: {run}.")
            for field in (
                "csv",
                "per_step_csv",
                "metadata",
                "config",
                "training_config",
                "checkpoint",
                "manifest",
                "mask_sample_manifest",
            ):
                if run.get(field):
                    relative = _source_relative(run[field], source_root, field)
                    run[field] = (artifact_root / relative).as_posix()
            run["source_collection"] = _relative_label(collection_file, spec_path.parent)
            run["source_root"] = str(source_root)
            ledgers.append(run)

    assert common is not None
    if seed_union != set(SEEDS):
        raise ValueError(
            f"Source seed subsets must cover exactly {list(SEEDS)}; found {sorted(seed_union)}."
        )
    expected = _expected_matrix()
    identities: dict[tuple[Any, ...], dict[str, Any]] = {}
    group_manifest: dict[tuple[str, str, str, str], tuple[str, tuple[str, ...]]] = {}
    for run in ledgers:
        identity = _run_identity(run)
        if identity not in expected:
            raise ValueError(f"Unexpected evaluation run identity {identity}.")
        if identity in identities:
            raise ValueError(f"Duplicate evaluation run identity across sources: {identity}.")
        if run.get("status") not in {"complete", "failed", "pending", "incomplete"}:
            raise ValueError(f"Invalid status for {identity}: {run.get('status')!r}.")
        if run["status"] != "complete" and not allow_incomplete:
            raise ValueError(
                f"Run {identity} has status {run['status']!r}; pass --allow-incomplete explicitly."
            )
        if run["status"] == "complete":
            _validate_completed_run(run, output_base, group_manifest)
        identities[identity] = run
    missing = expected - identities.keys()
    if missing:
        example = sorted(missing, key=repr)[:3]
        raise ValueError(
            f"Merged collections are missing {len(missing)} run identities, e.g. {example}."
        )
    if len(identities) != 390:
        raise ValueError(f"Expected exactly 390 evaluation entries, found {len(identities)}.")

    result = {
        **common,
        "evaluation_source_sha256": evaluation_source_sha256,
        "expected_training_seeds": list(SEEDS),
        "collection_version": 1,
        "path_base": ".",
        "runs": [identities[key] for key in sorted(identities, key=repr)],
    }
    result.update({field: value for field, value in identity_maps.items() if value})
    atomic_write_json(output_path, result)
    return result


def _collection_header(collection: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "expected_conditions",
        "expected_training_seeds",
        "expected_eval_seeds",
        "bootstrap_resamples",
        "bootstrap_seed",
        "evaluation_source_sha256",
    )
    missing = [field for field in fields if field not in collection]
    if missing:
        raise ValueError(f"Collection is missing required header fields {missing}.")
    header = {field: collection[field] for field in fields}
    if header["expected_conditions"] != list(CONDITIONS):
        raise ValueError("Each source collection must declare all 13 expected conditions.")
    if header["expected_eval_seeds"] != [0]:
        raise ValueError("chair_v2 evaluations require the fixed evaluator seed [0].")
    source_hash = header["evaluation_source_sha256"]
    if (
        not isinstance(source_hash, str)
        or len(source_hash) != 64
        or any(character not in "0123456789abcdef" for character in source_hash)
    ):
        raise ValueError("Collection evaluation_source_sha256 must be a lowercase SHA-256 digest.")
    if (
        not isinstance(header["expected_training_seeds"], list)
        or not header["expected_training_seeds"]
    ):
        raise ValueError("Each source collection must declare its training seed subset.")
    if len(set(header["expected_training_seeds"])) != len(header["expected_training_seeds"]):
        raise ValueError("A source collection declares duplicate training seeds.")
    if not set(header["expected_training_seeds"]).issubset(SEEDS):
        raise ValueError("A source collection declares unsupported training seeds.")
    return header


def _expected_matrix() -> set[tuple[Any, ...]]:
    return {
        (condition, seed, 0, dataset, "test", decoder, f"single|{protocol}")
        for condition in CONDITIONS
        for seed in SEEDS
        for _label, dataset, protocol, _manifest in BENCHMARKS
        for decoder in DECODERS
    }


def _run_identity(run: dict[str, Any]) -> tuple[Any, ...]:
    fields = ("condition", "training_seed", "eval_seed", "dataset", "split", "decoder", "protocol")
    missing = [field for field in fields if field not in run]
    if missing:
        raise ValueError(f"Evaluation run is missing identity fields {missing}.")
    return (
        str(run["condition"]),
        int(run["training_seed"]),
        int(run["eval_seed"]),
        str(run["dataset"]),
        str(run["split"]),
        str(run["decoder"]),
        str(run["protocol"]),
    )


def _merge_identity_map(target: dict[str, Any], incoming: Any, field: str) -> None:
    if not incoming:
        return
    if not isinstance(incoming, dict):
        raise ValueError(f"Collection field {field} must be an object.")
    for key, value in incoming.items():
        if key not in target:
            target[key] = value
        elif isinstance(target[key], dict) and isinstance(value, dict):
            for item_key, item_value in value.items():
                if item_key in target[key] and target[key][item_key] != item_value:
                    raise ValueError(f"Conflicting {field} value for {key}/{item_key}.")
                target[key][item_key] = item_value
        elif target[key] != value:
            raise ValueError(f"Conflicting {field} value for {key}.")


def _validate_completed_run(run, output_base: Path, group_manifest) -> None:
    csv_path = _resolve(output_base, run.get("csv"), "csv")
    metadata_path = _resolve(output_base, run.get("metadata"), "metadata")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Completed run CSV is missing after staging: {csv_path}.")
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Completed run metadata is missing after staging: {metadata_path}."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    identity = _run_identity(run)
    required = (
        "dataset",
        "split",
        "condition",
        "decoder",
        "training_seed",
        "eval_seed",
        "manifest_hash",
        "expected_episode_keys",
        "expected_episode_count",
        "expected_result_rows",
        "protocol",
        "checkpoint_sha256",
        "evaluation_config_sha256",
        "evaluation_source_sha256",
    )
    missing = [field for field in required if field not in metadata]
    if missing:
        raise ValueError(f"Completed run metadata {metadata_path} lacks fields {missing}.")
    expected_values = {
        "dataset": identity[3],
        "split": identity[4],
        "condition": identity[0],
        "decoder": identity[5],
        "training_seed": identity[1],
        "eval_seed": identity[2],
        "protocol": identity[6],
    }
    for field, value in expected_values.items():
        if metadata[field] != value:
            raise ValueError(f"Metadata {metadata_path} disagrees on {field}.")
    for field in ("checkpoint_sha256", "evaluation_config_sha256", "evaluation_source_sha256"):
        expected_hash = run.get(field)
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            raise ValueError(f"Completed run {identity} has invalid {field}.")
        if metadata[field] != expected_hash:
            raise ValueError(f"Metadata {metadata_path} disagrees on {field}.")
    keys = tuple(map(str, metadata["expected_episode_keys"]))
    if not keys or len(keys) != len(set(keys)) or manifest_hash(keys) != metadata["manifest_hash"]:
        raise ValueError(f"Metadata {metadata_path} has invalid episode identities.")
    expected_count = len(keys)
    for field in ("expected_episode_count", "expected_result_rows"):
        count = metadata[field]
        if type(count) is not int or count != expected_count:
            raise ValueError(f"Metadata {metadata_path} {field} disagrees with episode identities.")
        if field in run and (type(run[field]) is not int or run[field] != expected_count):
            raise ValueError(f"Metadata {metadata_path} {field} disagrees with collection.")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        csv_keys = [row.get("episode_key") for row in csv.DictReader(handle)]
    if (
        len(csv_keys) != expected_count
        or len(set(csv_keys)) != expected_count
        or set(csv_keys) != set(keys)
    ):
        raise ValueError(
            f"Completed run CSV {csv_path} has missing, duplicate, or unexpected episodes."
        )
    # Older suite collections omitted these fields. Restore them only in the
    # merged record after metadata and actual CSV identities agree.
    run.setdefault("expected_episode_count", expected_count)
    run.setdefault("expected_result_rows", expected_count)
    sample_count = run.get("mask_sample_count", 0)
    try:
        sample_count = int(sample_count)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Completed run {identity} has invalid mask_sample_count.") from exc
    if sample_count < 0:
        raise ValueError(f"Completed run {identity} has negative mask_sample_count.")
    sample_manifest_value = run.get("mask_sample_manifest")
    if sample_count > 0:
        sample_manifest_path = _resolve(output_base, sample_manifest_value, "mask_sample_manifest")
        expected_manifest_path = mask_sample_dir_for_csv(csv_path) / "sample_manifest.json"
        if sample_manifest_path.resolve() != expected_manifest_path.resolve():
            raise ValueError(
                f"Mask sample manifest for {identity} must be staged beside its result CSV."
            )
        validate_final_mask_samples(
            sample_manifest_path,
            metadata_path=metadata_path,
            expected_episode_keys=list(keys),
            sample_count=sample_count,
        )
    group = (identity[3], identity[4], identity[5], identity[6])
    current = (str(metadata["manifest_hash"]), keys)
    prior = group_manifest.setdefault(group, current)
    if prior != current:
        raise ValueError(f"Completed runs have inconsistent manifest metadata for {group}.")


def _safe_relative(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Merge source requires a nonempty {field}.")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} must be a safe relative path, got {value!r}.")
    return path


def _source_relative(value: str, source_root: Path, field: str) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (source_root / path).resolve()
    try:
        return resolved.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(
            f"{field} path {value!r} escapes declared source_root {source_root}."
        ) from exc


def _resolve(base: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Expected nonempty {field} path.")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _relative_label(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merge-spec", required=True)
    parser.add_argument("--merged-collection", required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = merge_collections(
        args.merge_spec,
        args.merged_collection,
        allow_incomplete=args.allow_incomplete,
    )
    print(f"Wrote {len(result['runs'])} merged runs to {Path(args.merged_collection).resolve()}.")


if __name__ == "__main__":
    main()
