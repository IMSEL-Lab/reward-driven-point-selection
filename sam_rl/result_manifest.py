"""Stable episode identities and strict validation for distributed result shards."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from sam_rl.episode_manifest import EpisodeSpec, episode_identity
from sam_rl.provenance import atomic_write_json, file_fingerprint, stable_hash

METRICS = ("iou", "dice", "boundary_iou", "wallclock_ms")


def episode_key(episode: EpisodeSpec) -> str:
    """Return the canonical path-independent episode ID."""
    return episode_identity(episode)


def manifest_hash(keys: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(keys).encode()).hexdigest()


def state_dict_hash(state: Mapping[str, object]) -> str:
    """Hash a torch-like state dict without device-specific storage metadata."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if not all(hasattr(value, attr) for attr in ("detach", "cpu", "numpy")):
            raise TypeError(f"State value {name!r} is not a tensor-like value.")
        tensor = cast(Any, value).detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def write_result_metadata(
    csv_path: str | Path,
    *,
    episodes: Sequence[EpisodeSpec],
    dataset: str,
    split: str,
    condition: str,
    model_config: Mapping[str, object],
    expected_result_rows: int,
    shard_episodes: Sequence[EpisodeSpec] | None = None,
    training_seed: int | None = None,
    eval_seed: int | None = None,
    decoder: str | None = None,
    model_state: Mapping[str, object] | None = None,
    checkpoint_path: str | Path | None = None,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Write the standard path-independent sidecar for one complete result partition."""
    if not episodes or not dataset or not split or not condition:
        raise ValueError("Episodes, dataset, split, and condition must be nonempty.")
    if expected_result_rows < 1:
        raise ValueError("expected_result_rows must be positive.")
    keys = [episode_identity(episode) for episode in episodes]
    shard_keys = [episode_identity(episode) for episode in (shard_episodes or episodes)]
    if len(set(keys)) != len(keys) or len(set(shard_keys)) != len(shard_keys):
        raise ValueError("Result metadata episode identities must be unique.")
    config = dict(model_config)
    if not config:
        raise ValueError("model_config must contain identifying settings.")
    payload: dict[str, object] = {
        "dataset": str(dataset),
        "split": str(split),
        "condition": str(condition),
        "model_config_hash": stable_hash(config),
        "model_config": config,
        "manifest_hash": manifest_hash(keys),
        "expected_episode_keys": keys,
        "expected_episode_count": len(keys),
        "shard_episode_keys": shard_keys,
        "episode_count": len(shard_keys),
        "expected_result_rows": int(expected_result_rows),
    }
    if model_state is not None and checkpoint_path is not None:
        raise ValueError("Pass model_state or checkpoint_path, not both.")
    if model_state is not None:
        payload["model_hash"] = state_dict_hash(model_state)
    elif checkpoint_path is not None:
        payload["model_hash"] = file_fingerprint(checkpoint_path)["sha256"]
    if training_seed is not None:
        payload["training_seed"] = int(training_seed)
    if eval_seed is not None:
        payload["eval_seed"] = int(eval_seed)
    if decoder is not None:
        payload["decoder"] = str(decoder)
    if extra:
        reserved = set(payload).intersection(extra)
        if reserved:
            raise ValueError(f"Extra metadata cannot override reserved fields {sorted(reserved)}.")
        payload.update(dict(extra))
    sidecar = Path(csv_path).with_suffix(".metadata.json")
    atomic_write_json(sidecar, payload)
    return payload


def validate_result_shards(
    csv_paths: Iterable[str | Path],
    *,
    expected_keys: Iterable[str],
    expected_methods: Iterable[str],
    budgets: Iterable[int],
    seeds: Iterable[int] | Mapping[str, Iterable[int]],
    training_seeds: Iterable[int] | None = None,
    eval_seeds: Iterable[int] | None = None,
    metadata_paths: Iterable[str | Path] | None = None,
    allow_legacy: bool = False,
) -> list[dict[str, str]]:
    """Require exact method, budget, episode, and seed coverage across shards."""
    expected_order = list(expected_keys)
    expected_keyset = set(expected_order)
    if len(expected_order) != len(expected_keyset):
        raise ValueError("Expected episode identity list contains duplicates.")
    methods, budget_set = set(expected_methods), set(map(int, budgets))
    if isinstance(seeds, Mapping):
        seed_values = cast(Mapping[str, Iterable[int]], seeds)
        seed_map = {method: set(map(int, values)) for method, values in seed_values.items()}
    else:
        seed_map = {method: set(map(int, seeds)) for method in methods}
    paired_seed_contract = training_seeds is not None or eval_seeds is not None
    train_seed_set = set(map(int, training_seeds or ()))
    eval_seed_set = set(map(int, eval_seeds or ()))
    if paired_seed_contract and (not train_seed_set or not eval_seed_set):
        raise ValueError("Both training_seeds and eval_seeds are required for split seed fields.")
    if (
        not expected_keyset
        or not methods
        or not budget_set
        or (not paired_seed_contract and any(not values for values in seed_map.values()))
    ):
        raise ValueError("Expected identity, method, budget, and seed sets must be nonempty.")
    paths = [Path(p) for p in csv_paths]
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("Duplicate result shard paths were supplied.")
    metadata = (
        [Path(p) for p in metadata_paths]
        if metadata_paths is not None
        else [p.with_suffix(".metadata.json") for p in paths]
    )
    if len(metadata) != len(paths):
        raise ValueError("Each CSV shard must have one metadata file.")
    docs: list[dict] = []
    if not allow_legacy:
        for p in metadata:
            if not p.exists():
                raise ValueError(f"Missing shard metadata: {p}")
        docs = [json.loads(p.read_text(encoding="utf-8")) for p in metadata]
        metadata_groups: dict[tuple, dict] = {}
        for doc in docs:
            required = ("dataset", "split", "condition", "model_config_hash", "manifest_hash")
            missing_fields = [field for field in required if not doc.get(field)]
            if missing_fields:
                raise ValueError(
                    f"Shard metadata {doc.get('output_csv', '')} lacks {missing_fields}."
                )
            if doc.get("manifest_hash") != manifest_hash(expected_order):
                raise ValueError("Shard manifest hash does not match expected episode identities.")
            group = (
                doc["dataset"],
                doc["split"],
                doc["condition"],
                doc.get("training_seed"),
                doc.get("eval_seed"),
                doc.get("decoder"),
                doc["model_config_hash"],
            )
            prior = metadata_groups.setdefault(group, doc)
            if doc.get("model_hash") != prior.get("model_hash"):
                raise ValueError("Incompatible model_hash among shards for the same run partition.")
        if len({(doc["dataset"], doc["split"], doc["manifest_hash"]) for doc in docs}) > 1:
            raise ValueError("Incompatible dataset, split, or episode manifest across shards.")
    seen: set[tuple] = set()
    rows: list[dict[str, str]] = []
    for path_index, path in enumerate(paths):
        if not path.exists():
            raise FileNotFoundError(path)
        shard_rows: list[dict[str, str]] = []
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                method, budget = str(row["method"]), int(row["N"])
                key = str(row.get("episode_key") or row.get("image_id", ""))
                training_seed = int(row["training_seed"]) if paired_seed_contract else None
                eval_seed = int(row["eval_seed"]) if paired_seed_contract else int(row["seed"])
                if (
                    method not in methods
                    or budget not in budget_set
                    or key not in expected_keyset
                    or (
                        (training_seed not in train_seed_set or eval_seed not in eval_seed_set)
                        if paired_seed_contract
                        else method not in seed_map or eval_seed not in seed_map[method]
                    )
                ):
                    raise ValueError(
                        f"Unexpected result identity in {path}: {method}/{budget}/{key}/{training_seed}/{eval_seed}"
                    )
                for metric in METRICS:
                    value = float(row[metric])
                    if not math.isfinite(value):
                        raise ValueError(f"Nonfinite {metric} in {path} for {key}.")
                identity = (method, budget, key, training_seed, eval_seed)
                if identity in seen:
                    raise ValueError(f"Duplicate result identity {identity}.")
                seen.add(identity)
                rows.append(row)
                shard_rows.append(row)
        if docs:
            doc = docs[path_index]
            if "expected_result_rows" in doc and int(doc["expected_result_rows"]) != len(
                shard_rows
            ):
                raise ValueError(
                    f"Partial shard {path}: expected {doc['expected_result_rows']} rows, found {len(shard_rows)}."
                )
            if "shard_episode_keys" in doc:
                actual_shard_keys = {
                    str(row.get("episode_key") or row.get("image_id", "")) for row in shard_rows
                }
                if actual_shard_keys != set(map(str, doc["shard_episode_keys"])):
                    raise ValueError(f"Shard episode identities disagree with metadata for {path}.")
    if paired_seed_contract:
        expected = {
            (m, n, e, train_seed, eval_seed)
            for m in methods
            for n in budget_set
            for e in expected_keyset
            for train_seed in train_seed_set
            for eval_seed in eval_seed_set
        }
    else:
        expected = {
            (m, n, e, None, seed)
            for m in methods
            for n in budget_set
            for e in expected_keyset
            for seed in seed_map[m]
        }
    missing = expected - seen
    if missing:
        raise ValueError(
            f"Missing {len(missing)} expected result rows; first: {sorted(missing)[:3]}"
        )
    if seen != expected:
        raise ValueError("Result shard identity set differs from the expected set.")
    return rows
