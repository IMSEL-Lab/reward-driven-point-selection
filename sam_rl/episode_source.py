from __future__ import annotations

import json
from pathlib import Path

from sam_rl.episode_manifest import EpisodeSpec, load_episode_manifest, validate_episode_paths


def build_fss_episodes(
    dataset_root: str | Path,
    split_file: str | Path,
    split_name: str,
    pair_schedule: list[list[int | str]] | list[tuple[int | str, int | str]],
    max_episodes: int | None = None,
) -> list[EpisodeSpec]:
    split_path = Path(split_file)
    payload = json.loads(split_path.read_text(encoding="utf-8"))
    if split_name not in payload:
        raise KeyError(f"Unknown split '{split_name}' in {split_path}")
    dataset_root = Path(dataset_root)
    episodes: list[EpisodeSpec] = []
    for class_name in payload[split_name]:
        class_dir = dataset_root / str(class_name)
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing FSS-1000 class directory: {class_dir}")
        for support_id, query_id in pair_schedule:
            episode = EpisodeSpec(
                dataset="fss1000",
                split=str(split_name),
                class_name=str(class_name),
                support_id=str(support_id),
                query_id=str(query_id),
                support_image_path=class_dir / f"{support_id}.jpg",
                support_mask_path=class_dir / f"{support_id}.png",
                query_image_path=class_dir / f"{query_id}.jpg",
                query_mask_path=class_dir / f"{query_id}.png",
            )
            validate_episode_paths(episode)
            episodes.append(episode)
            if max_episodes is not None and len(episodes) >= int(max_episodes):
                return episodes
    return episodes


def build_episodes_from_dataset_cfg(
    dataset_cfg: dict,
    split_name: str,
    *,
    max_episodes: int | None = None,
) -> list[EpisodeSpec]:
    if dataset_cfg.get("episode_manifest"):
        return load_episode_manifest(
            dataset_cfg["episode_manifest"],
            max_episodes=max_episodes,
            data_root=dataset_cfg.get("data_root", dataset_cfg.get("root")),
        )
    manifests = dataset_cfg.get("episode_manifests")
    if manifests:
        manifest_path = manifests[str(split_name)]
        return load_episode_manifest(
            manifest_path,
            max_episodes=max_episodes,
            data_root=dataset_cfg.get("data_root", dataset_cfg.get("root")),
        )
    manifest_pattern = dataset_cfg.get("episode_manifest_pattern")
    if manifest_pattern:
        return load_episode_manifest(
            str(manifest_pattern).format(split=split_name),
            max_episodes=max_episodes,
            data_root=dataset_cfg.get("data_root", dataset_cfg.get("root")),
        )
    return build_fss_episodes(
        dataset_root=dataset_cfg["root"],
        split_file=dataset_cfg["split_file"],
        split_name=split_name,
        pair_schedule=list(dataset_cfg["pair_schedule"]),
        max_episodes=max_episodes,
    )
