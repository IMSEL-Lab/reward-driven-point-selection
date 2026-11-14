from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EpisodeSpec:
    dataset: str
    split: str
    class_name: str
    support_id: str
    query_id: str
    support_image_path: Path
    support_mask_path: Path
    query_image_path: Path
    query_mask_path: Path

    @property
    def image_id(self) -> str:
        return f"{self.class_name}_s{self.support_id}_q{self.query_id}"

    @property
    def stable_id(self) -> str:
        return episode_identity(self)


def safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text)).strip("_")


def load_episode_manifest(
    manifest_path: str | Path,
    *,
    max_episodes: int | None = None,
    validate_paths: bool = True,
    data_root: str | Path | None = None,
) -> list[EpisodeSpec]:
    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes_raw = payload["episodes"] if isinstance(payload, dict) else payload
    episodes: list[EpisodeSpec] = []
    for item in episodes_raw:
        episode = episode_from_record(
            item, default_split=str(payload.get("split", "")) if isinstance(payload, dict) else ""
        )
        if data_root is not None:
            episode = resolve_episode_paths(episode, Path(data_root))
        if validate_paths:
            validate_episode_paths(episode)
        episodes.append(episode)
        if max_episodes is not None and len(episodes) >= int(max_episodes):
            break
    return episodes


def write_episode_manifest(
    manifest_path: str | Path,
    episodes: Iterable[EpisodeSpec],
    *,
    metadata: dict | None = None,
) -> None:
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    episode_rows = [episode_to_record(episode) for episode in episodes]
    payload: dict[str, object] = {"episodes": episode_rows}
    if metadata:
        payload.update(dict(metadata))
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def episode_from_record(record: dict, *, default_split: str = "") -> EpisodeSpec:
    support_id = str(record.get("support_id", "support"))
    query_id = str(record.get("query_id", "query"))
    class_name = str(record.get("class_name", record.get("category", "object")))
    split = str(record.get("split", default_split))
    return EpisodeSpec(
        dataset=str(record.get("dataset", "custom")),
        split=split,
        class_name=class_name,
        support_id=support_id,
        query_id=query_id,
        support_image_path=Path(record["support_image_path"]),
        support_mask_path=Path(record["support_mask_path"]),
        query_image_path=Path(record["query_image_path"]),
        query_mask_path=Path(record["query_mask_path"]),
    )


def episode_to_record(episode: EpisodeSpec) -> dict[str, str]:
    return {
        "dataset": str(episode.dataset),
        "split": str(episode.split),
        "class_name": str(episode.class_name),
        "support_id": str(episode.support_id),
        "query_id": str(episode.query_id),
        "image_id": str(episode.image_id),
        "episode_id": episode_identity(episode),
        "support_image_path": str(episode.support_image_path),
        "support_mask_path": str(episode.support_mask_path),
        "query_image_path": str(episode.query_image_path),
        "query_mask_path": str(episode.query_mask_path),
    }


def validate_episode_paths(episode: EpisodeSpec) -> None:
    for path in (
        episode.support_image_path,
        episode.support_mask_path,
        episode.query_image_path,
        episode.query_mask_path,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing episode asset: {path}")


def resolve_episode_paths(episode: EpisodeSpec, data_root: Path) -> EpisodeSpec:
    """Resolve relative manifest asset paths under an explicit dataset root.

    Absolute legacy manifest paths remain absolute. Relative paths are rooted at
    ``data_root``; ``"."`` is consequently a valid relocatable root placeholder.
    """
    from dataclasses import replace

    def resolve(path: Path) -> Path:
        return path if path.is_absolute() else data_root / path

    return replace(
        episode,
        support_image_path=resolve(episode.support_image_path),
        support_mask_path=resolve(episode.support_mask_path),
        query_image_path=resolve(episode.query_image_path),
        query_mask_path=resolve(episode.query_mask_path),
    )


def episode_manifest_hash(manifest_path: str | Path) -> str:
    """SHA-256 of the exact manifest file used by an experiment."""
    return hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()


def episode_identity(episode: EpisodeSpec) -> str:
    """Stable ID over protocol identities, independent of machine-local paths."""
    identity = [
        episode.dataset,
        episode.split,
        episode.class_name,
        episode.support_id,
        episode.query_id,
    ]
    digest = hashlib.sha256(json.dumps(identity, separators=(",", ":")).encode()).hexdigest()
    return f"{episode.dataset}:{digest[:24]}"


def validate_manifest_set(
    manifest_paths: dict[str, str | Path],
    *,
    expected_counts: dict[str, int] | None = None,
    expected_query_ids: dict[str, set[str]] | None = None,
    support_bank_ids: set[str] | None = None,
    supports_per_query: int | None = None,
) -> dict[str, Any]:
    """Validate immutable split manifests without requiring their assets locally."""
    report: dict[str, Any] = {"splits": {}, "manifest_hashes": {}}
    seen_queries: dict[tuple[str, str, str], str] = {}
    for split, path_value in manifest_paths.items():
        path = Path(path_value)
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw = payload.get("episodes", payload) if isinstance(payload, dict) else payload
        if not isinstance(raw, list):
            raise ValueError(f"Manifest {path} must contain an episodes list")
        if (
            expected_counts is not None
            and split in expected_counts
            and len(raw) != expected_counts[split]
        ):
            raise ValueError(
                f"{split}: expected {expected_counts[split]} episodes, found {len(raw)}"
            )
        pairs: set[tuple[str, str, str]] = set()
        queries: set[str] = set()
        query_counts: dict[str, int] = {}
        for row in raw:
            support_id = str(row["support_id"])
            query_id = str(row["query_id"])
            class_name = str(row.get("class_name", row.get("category", "object")))
            if row.get("episode_id"):
                stored_id = str(row["episode_id"])
                expected_id = episode_identity(episode_from_record(row, default_split=split))
                if stored_id != expected_id:
                    raise ValueError(f"{split}: stored episode_id does not match its identities")
            if support_id == query_id:
                raise ValueError(f"{split}: support equals query for {query_id}")
            if support_bank_ids is not None and support_id not in support_bank_ids:
                raise ValueError(f"{split}: support {support_id} is outside the declared bank")
            pair = (class_name, support_id, query_id)
            if pair in pairs:
                raise ValueError(f"{split}: duplicate support/query pair {pair}")
            pairs.add(pair)
            queries.add(query_id)
            query_counts[query_id] = query_counts.get(query_id, 0) + 1
        if supports_per_query is not None:
            wrong = {
                query: count for query, count in query_counts.items() if count != supports_per_query
            }
            if wrong:
                raise ValueError(
                    f"{split}: expected {supports_per_query} supports per query; "
                    f"found {list(wrong.items())[:8]}"
                )
        expected = None if expected_query_ids is None else expected_query_ids.get(split)
        if expected is not None and queries != expected:
            raise ValueError(
                f"{split}: query ID mismatch; missing={sorted(expected - queries)[:8]}, "
                f"unexpected={sorted(queries - expected)[:8]}"
            )
        for query_id in queries:
            row_dataset = str(row.get("dataset", "custom"))
            row_class = str(row.get("class_name", row.get("category", "object")))
            query_key = (row_dataset, row_class, query_id)
            previous = seen_queries.setdefault(query_key, split)
            if previous != split:
                raise ValueError(f"query {query_id} appears in both {previous} and {split}")
        report["splits"][split] = {
            "episode_count": len(raw),
            "unique_query_count": len(queries),
            "unique_pair_count": len(pairs),
        }
        report["manifest_hashes"][split] = episode_manifest_hash(path)
    return report
