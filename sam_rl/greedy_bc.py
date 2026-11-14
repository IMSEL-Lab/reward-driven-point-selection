from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from sam_rl._types import OracleExample, RolloutResult
from sam_rl.baseline_eval import (
    DINO_REVISION,
    CsvResultWriter,
    DinoSimilarityCache,
    FewShotEpisode,
    boundary_iou_score,
    dice_score,
    iou_score,
    load_dino_model,
    load_episode_arrays,
    load_sam_predictor,
    predict_mask_from_points,
    save_overlay_artifacts,
    write_json,
)
from sam_rl.episode_manifest import episode_identity
from sam_rl.episode_source import build_episodes_from_dataset_cfg
from sam_rl.mask_samples import mask_sample_dir_for_csv, write_final_mask_samples
from sam_rl.protocol import (
    autocast_context,
    multimask_for_prefix,
    normalize_decoder_policy,
    resolve_precision,
)
from sam_rl.provenance import (
    atomic_write_json as write_atomic_json,
)
from sam_rl.provenance import (
    atomic_write_npz,
    cache_metadata,
    file_fingerprint,
)
from sam_rl.result_manifest import write_result_metadata
from sam_rl.sam_cache import SamImageEmbeddingCache, set_predictor_image_embedding


@dataclass(frozen=True)
class OracleHeatmapExample:
    episode: FewShotEpisode
    step_index: int
    budget: int
    target_idx: int
    prefix_indices: tuple[int, ...]


class OracleHeatmapDataset(Dataset):
    def __init__(
        self,
        episodes: Sequence[FewShotEpisode],
        oracle_dir: str | Path,
        sim_cache: DinoSimilarityCache,
        image_size: int,
        grid: int,
        budget: int,
        query_feature_cache: DinoQueryFeatureCache | None = None,
        use_query_rgb: bool = False,
        predictor=None,
        use_current_mask: bool = False,
        mask_cache_dir: str | Path | None = None,
        decoder_policy: str = "single",
        decoder_checkpoint: str | None = None,
        sam_embedding_cache: SamImageEmbeddingCache | None = None,
    ) -> None:
        self.oracle_dir = Path(oracle_dir)
        self.sim_cache = sim_cache
        self.image_size = int(image_size)
        self.grid = int(grid)
        self.budget = int(budget)
        self.query_feature_cache = query_feature_cache
        self.use_query_rgb = bool(use_query_rgb)
        self.predictor = predictor
        self.use_current_mask = bool(use_current_mask)
        self.mask_cache_dir = Path(mask_cache_dir) if mask_cache_dir is not None else None
        self.decoder_policy = normalize_decoder_policy(decoder_policy)
        self.decoder_checkpoint = str(decoder_checkpoint or "")
        self.sam_embedding_cache = sam_embedding_cache
        self._prefix_mask_cache: dict[tuple[str, tuple[int, ...]], torch.Tensor] = {}
        self.examples: list[OracleHeatmapExample] = []
        self._feature_cache: dict[str, dict[str, torch.Tensor]] = {}
        for episode in episodes:
            payload = load_oracle_payload(self.oracle_dir, episode)
            steps = list(payload.get("steps", []))
            if len(steps) < self.budget:
                raise ValueError(
                    f"Oracle cache for {episode.image_id} only has {len(steps)} steps; "
                    f"need at least {self.budget}."
                )
            prefix: list[int] = []
            for step_idx in range(self.budget):
                step = steps[step_idx]
                target_idx = int(step["location_idx"])
                self.examples.append(
                    OracleHeatmapExample(
                        episode=episode,
                        step_index=step_idx + 1,
                        budget=self.budget,
                        target_idx=target_idx,
                        prefix_indices=tuple(prefix),
                    )
                )
                prefix.append(target_idx)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> OracleExample:
        example = self.examples[index]
        features = self._episode_features(example.episode)
        click_map = build_click_map(self.grid, example.prefix_indices)
        obs = {
            "sim_map": features["sim_map"].clone(),
            "click_map": click_map,
            "step_index": torch.tensor([float(example.step_index)], dtype=torch.float32),
            "budget": torch.tensor([float(example.budget)], dtype=torch.float32),
        }
        if getattr(self, "use_current_mask", False):
            obs["current_mask"] = self._prefix_mask(example)
        if "query_rgb" in features:
            obs["query_rgb"] = features["query_rgb"].clone()
        if "query_dino" in features:
            obs["query_dino"] = features["query_dino"].clone()
        return {
            "obs": obs,
            "target_idx": torch.tensor(example.target_idx, dtype=torch.long),
            "example_index": torch.tensor(index, dtype=torch.long),
        }

    def _episode_features(self, episode: FewShotEpisode) -> dict[str, torch.Tensor]:
        cache_key = episode.image_id
        cached = self._feature_cache.get(cache_key)
        if cached is not None:
            return cached
        cached = {
            "sim_map": scalar_map_to_grid_tensor(self.sim_cache.load(episode), self.grid),
        }
        if self.use_query_rgb:
            _, _, query_rgb, _ = load_episode_arrays(episode, image_size=self.image_size)
            cached["query_rgb"] = rgb_to_grid_tensor(query_rgb, self.grid)
        if self.query_feature_cache is not None:
            cached["query_dino"] = torch.from_numpy(
                np.asarray(self.query_feature_cache.load(episode), dtype=np.float32).copy()
            ).contiguous()
        self._feature_cache[cache_key] = cached
        return cached

    def _prefix_mask(self, example: OracleHeatmapExample) -> torch.Tensor:
        key = (example.episode.image_id, example.prefix_indices)
        if not example.prefix_indices or self.predictor is None:
            return torch.zeros(1, self.grid, self.grid, dtype=torch.float32)
        cached = self._prefix_mask_cache.get(key)
        if cached is not None:
            return cached.clone()
        metadata = cache_metadata(
            kind="causal_prefix_mask",
            episode=example.episode,
            config={
                "prefix_indices": list(example.prefix_indices),
                "grid": self.grid,
                "image_size": self.image_size,
                "decoder_policy": self.decoder_policy,
                "coordinate_policy": "rounded_pixel_centers_v1",
                "mask_resize": "bilinear_align_corners_false",
                "decoder_checkpoint": (
                    file_fingerprint(self.decoder_checkpoint)
                    if self.decoder_checkpoint and Path(self.decoder_checkpoint).is_file()
                    else self.decoder_checkpoint
                ),
            },
        )
        path = None
        if self.mask_cache_dir is not None:
            path = (
                self.mask_cache_dir
                / metadata["cache_key"][:24]
                / f"{safe_slug(example.episode.image_id)}.npz"
            )
            if path.is_file():
                try:
                    with np.load(path, allow_pickle=False) as saved:
                        if json.loads(str(saved["metadata_json"].item())) == metadata:
                            tensor = torch.from_numpy(
                                np.asarray(saved["mask"], dtype=np.float32).copy()
                            )
                            self._prefix_mask_cache[key] = tensor
                            return tensor.clone()
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    pass
        _, _, image, _ = load_episode_arrays(example.episode, image_size=self.image_size)
        points = []
        h, w = image.shape[:2]
        for idx in example.prefix_indices:
            gx, gy = location_idx_to_grid_xy(idx, self.grid)
            px, py = rounded_grid_xy_to_pixel_xy(
                gx, gy, image_width=w, image_height=h, grid=self.grid
            )
            points.append((px, py))
        set_cached_predictor_image(
            self.predictor,
            self.sam_embedding_cache,
            example.episode,
            image,
        )
        mask = predict_mask_from_points(
            self.predictor, np.asarray(points, dtype=np.float32), decoder_policy=self.decoder_policy
        )
        tensor = scalar_map_to_grid_tensor(np.asarray(mask, dtype=np.float32), self.grid)
        self._prefix_mask_cache[key] = tensor
        if path is not None:
            atomic_write_npz(
                path,
                mask=tensor.numpy(),
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
        return tensor.clone()


class GreedyClickHeatmapPolicy(nn.Module):
    def __init__(
        self,
        grid: int = 37,
        hidden_channels: int = 64,
        depth: int = 4,
        dropout: float = 0.1,
        use_query_rgb: bool = False,
        use_query_dino_features: bool = False,
        dino_feature_channels: int = 1024,
        dino_proj_channels: int = 32,
        use_current_mask: bool = False,
        reserve_mask_channel: bool = False,
    ) -> None:
        super().__init__()
        self.grid = int(grid)
        self.use_query_rgb = bool(use_query_rgb)
        self.use_query_dino_features = bool(use_query_dino_features)
        self.dino_feature_channels = int(dino_feature_channels)
        self.dino_proj_channels = int(dino_proj_channels)
        self.use_current_mask = bool(use_current_mask)
        self.reserve_mask_channel = bool(reserve_mask_channel or use_current_mask)
        extra_channels = 3 if self.use_query_rgb else 0
        if self.use_query_dino_features:
            self.dino_proj = nn.Sequential(
                nn.Conv2d(self.dino_feature_channels, self.dino_proj_channels, kernel_size=1),
                nn.GroupNorm(
                    num_groups=max(1, self.dino_proj_channels // 8),
                    num_channels=self.dino_proj_channels,
                ),
                nn.SiLU(inplace=True),
            )
            extra_channels += self.dino_proj_channels
        else:
            self.dino_proj = None
        self.bc_input_channels = 7 + extra_channels
        in_channels = self.bc_input_channels + (1 if self.reserve_mask_channel else 0)
        layers: list[nn.Module] = []
        channels = in_channels
        for block_idx in range(max(depth, 1)):
            layers.append(nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1))
            layers.append(
                nn.GroupNorm(num_groups=max(1, hidden_channels // 16), num_channels=hidden_channels)
            )
            layers.append(nn.SiLU(inplace=True))
            if block_idx > 0:
                layers.append(nn.Dropout2d(dropout))
            channels = hidden_channels
        self.encoder = nn.Sequential(*layers)
        self.skip = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        if self.reserve_mask_channel:
            with torch.no_grad():
                self.encoder[0].weight[:, self.bc_input_channels :, :, :] = 0.0
                self.skip.weight[:, self.bc_input_channels :, :] = 0.0
        coords = torch.linspace(-1.0, 1.0, self.grid)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("_coord_x", xx.unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("_coord_y", yy.unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        sim_map = ensure_single_channel(obs["sim_map"])
        click_map = ensure_single_channel(obs["click_map"])
        step_index = ensure_scalar(obs["step_index"]).float()
        budget = ensure_scalar(obs["budget"]).float().clamp_min(1.0)
        batch_size = sim_map.shape[0]
        step_plane = (
            (step_index / budget).view(batch_size, 1, 1, 1).expand(-1, -1, self.grid, self.grid)
        )
        budget_plane = (
            (budget / 10.0).view(batch_size, 1, 1, 1).expand(-1, -1, self.grid, self.grid)
        )
        click_count = click_map.flatten(2).sum(dim=-1, keepdim=True).view(batch_size, 1, 1, 1)
        click_plane = (click_count / budget.view(batch_size, 1, 1, 1)).expand(
            -1, -1, self.grid, self.grid
        )
        parts = [
            sim_map,
            click_map,
            step_plane,
            budget_plane,
            click_plane,
            self._coord_x.expand(batch_size, -1, -1, -1),
            self._coord_y.expand(batch_size, -1, -1, -1),
        ]
        if self.reserve_mask_channel:
            current_mask = obs.get("current_mask")
            if current_mask is None:
                current_mask = torch.zeros(
                    batch_size, 1, self.grid, self.grid, device=sim_map.device, dtype=sim_map.dtype
                )
            current_mask = ensure_single_channel(current_mask).float()
            parts.append(current_mask if self.use_current_mask else torch.zeros_like(current_mask))
        if self.dino_proj is not None:
            dino_features = self.dino_proj(ensure_image(obs["query_dino"]).float())
            if dino_features.shape[-2:] != (self.grid, self.grid):
                dino_features = F.interpolate(
                    dino_features,
                    size=(self.grid, self.grid),
                    mode="bilinear",
                    align_corners=False,
                )
            parts.insert(0, dino_features)
        if self.use_query_rgb:
            parts.insert(0, ensure_image(obs["query_rgb"]))
        stacked = torch.cat(parts, dim=1)
        features = self.encoder(stacked) + self.skip(stacked)
        return self.head(features).flatten(1)


def ensure_image(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 3:
        return tensor.unsqueeze(0)
    return tensor


def ensure_single_channel(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.dim() == 3:
        return tensor.unsqueeze(1)
    return tensor


def ensure_scalar(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 0:
        return tensor.view(1, 1)
    if tensor.dim() == 1:
        return tensor.unsqueeze(-1)
    return tensor


def collate_oracle_heatmap_batch(
    batch: list[OracleExample],
) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    obs_keys = batch[0]["obs"].keys()
    obs = {key: torch.stack([sample["obs"][key] for sample in batch], dim=0) for key in obs_keys}
    target_idx = torch.stack([sample["target_idx"] for sample in batch], dim=0)
    example_index = torch.stack([sample["example_index"] for sample in batch], dim=0)
    return {
        "obs": obs,
        "target_idx": target_idx.long(),
        "example_index": example_index.long(),
    }


def build_policy(cfg: dict) -> GreedyClickHeatmapPolicy:
    return GreedyClickHeatmapPolicy(
        grid=int(cfg.get("grid", 37)),
        hidden_channels=int(cfg.get("hidden_channels", 64)),
        depth=int(cfg.get("depth", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_query_rgb=bool(cfg.get("use_query_rgb", False)),
        use_query_dino_features=bool(cfg.get("use_query_dino_features", False)),
        dino_feature_channels=int(cfg.get("dino_feature_channels", 1024)),
        dino_proj_channels=int(cfg.get("dino_proj_channels", 32)),
        use_current_mask=bool(cfg.get("use_current_mask", False)),
        reserve_mask_channel=bool(cfg.get("reserve_mask_channel", False)),
    )


def heatmap_loss(
    logits: torch.Tensor,
    target_idx: torch.Tensor,
    loss_cfg: dict | None = None,
    invalid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    loss_cfg = loss_cfg or {}
    kind = str(loss_cfg.get("kind", "cross_entropy"))
    if kind == "cross_entropy":
        return F.cross_entropy(logits, target_idx)
    if kind == "focal":
        ce = F.cross_entropy(logits, target_idx, reduction="none")
        pt = torch.exp(-ce)
        alpha = float(loss_cfg.get("alpha", 1.0))
        gamma = float(loss_cfg.get("gamma", 2.0))
        return (alpha * torch.pow(1.0 - pt, gamma) * ce).mean()
    grid = infer_grid_size(logits)
    sigma = float(loss_cfg.get("sigma", 2.0))
    if kind == "gaussian_kl":
        target = gaussian_target_map(
            target_idx, grid=grid, sigma=sigma, device=logits.device, dtype=logits.dtype
        )
        if invalid_mask is not None:
            target = target.masked_fill(invalid_mask.view(target.shape[0], grid, grid), 0.0)
        target = target.flatten(1)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean")
    if kind == "gaussian_bce":
        target = gaussian_target_map(
            target_idx, grid=grid, sigma=sigma, device=logits.device, dtype=logits.dtype
        ).flatten(1)
        if invalid_mask is not None:
            target = target.masked_fill(invalid_mask, 0.0)
        return masked_binary_cross_entropy_with_logits(logits, target, invalid_mask=invalid_mask)
    raise ValueError(f"Unsupported heatmap loss kind: {kind}")


def rgb_to_grid_tensor(rgb: np.ndarray, grid: int) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(rgb, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
    return (
        F.interpolate(
            tensor.unsqueeze(0),
            size=(int(grid), int(grid)),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .contiguous()
    )


def scalar_map_to_grid_tensor(values: np.ndarray, grid: int) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32).copy()).unsqueeze(0).unsqueeze(0)
    return (
        F.interpolate(
            tensor,
            size=(int(grid), int(grid)),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .contiguous()
    )


def build_click_map(grid: int, prefix_indices: Sequence[int]) -> torch.Tensor:
    click_map = torch.zeros(1, int(grid), int(grid), dtype=torch.float32)
    if prefix_indices:
        click_map.view(-1)[list(prefix_indices)] = 1.0
    return click_map


class DinoQueryFeatureCache:
    def __init__(
        self,
        cache_dir: str | Path,
        dino_model: torch.nn.Module,
        dino_checkpoint: str | Path,
        device: str,
        image_size: int,
        precision: str = "fp32",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dino_model = dino_model
        self.dino_checkpoint = Path(dino_checkpoint)
        self.device = str(device)
        self.image_size = int(image_size)
        self.requested_precision = str(precision).lower()
        self.precision = resolve_dino_precision(self.requested_precision, self.device)
        self.storage_dtype = "float32" if self.precision == "fp32" else "float16"
        if self.image_size % 14 != 0:
            raise ValueError(f"DINO image_size must be divisible by 14, got {self.image_size}.")

    def path_for(self, episode: FewShotEpisode) -> Path:
        provenance = self._metadata(episode)
        return (
            self.cache_dir
            / provenance["cache_key"][:24]
            / f"{episode.split}_{safe_slug(episode.image_id)}.npz"
        )

    def _metadata(self, episode: FewShotEpisode) -> dict:
        return cache_metadata(
            kind="dino_query_feature",
            episode=episode,
            config={
                "checkpoint": file_fingerprint(self.dino_checkpoint),
                "image_size": self.image_size,
                "preprocessing": "imagenet_rgb_resize_normalize_patch14",
                "encoder_revision": DINO_REVISION,
                "compute_precision": self.precision,
                "requested_precision": self.requested_precision,
                "storage_dtype": self.storage_dtype,
            },
        )

    def ensure(self, episode: FewShotEpisode) -> Path:
        path = self.path_for(episode)
        metadata = self._metadata(episode)
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as payload:
                    if json.loads(str(payload["metadata_json"].item())) == metadata:
                        return path
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                pass
        _, _, query_rgb, _ = load_episode_arrays(episode, image_size=self.image_size)
        query_dino = compute_dino_query_feature_map(
            self.dino_model, self.device, query_rgb, precision=self.precision
        )
        from sam_rl.provenance import atomic_write_npz

        atomic_write_npz(
            path,
            query_dino=np.asarray(query_dino, dtype=self.storage_dtype),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        return path

    def load(self, episode: FewShotEpisode) -> np.ndarray:
        with np.load(self.ensure(episode), allow_pickle=False) as payload:
            return np.asarray(payload["query_dino"], dtype=np.float32)


@torch.inference_mode()
def compute_dino_query_feature_map(
    dino_model: torch.nn.Module,
    device: str,
    query_rgb: np.ndarray,
    precision: str = "fp32",
) -> np.ndarray:
    query_tensor = dino_preprocess_rgb(query_rgb).unsqueeze(0).to(device)
    with autocast_context(device, precision):
        features = dino_model.forward_features(query_tensor)["x_norm_patchtokens"].float()
    features = F.normalize(features, dim=-1).cpu()[0]
    patch_count, channels = features.shape
    patch_grid = round(float(patch_count) ** 0.5)
    if patch_grid * patch_grid != patch_count:
        raise ValueError(f"Expected square DINO patch grid, got {patch_count} tokens.")
    return features.reshape(patch_grid, patch_grid, channels).permute(2, 0, 1).contiguous().numpy()


def resolve_dino_precision(precision: str, device: str) -> str:
    requested = str(precision).lower()
    if requested == "legacy":
        requested = "bf16" if torch.device(device).type == "cuda" else "fp32"
    return resolve_precision(device, requested)


def dino_preprocess_rgb(rgb: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(rgb, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def build_fss_split_episodes(
    cfg: dict, split_name: str, max_episodes: int | None = None
) -> list[FewShotEpisode]:
    episodes = build_episodes_from_dataset_cfg(
        cfg["dataset"],
        str(split_name),
        max_episodes=max_episodes,
    )
    return [coerce_episode_to_few_shot(episode) for episode in episodes]


def coerce_episode_to_few_shot(episode: FewShotEpisode) -> FewShotEpisode:
    if isinstance(episode, FewShotEpisode):
        return episode
    return FewShotEpisode(
        dataset=str(episode.dataset),
        split=str(episode.split),
        class_name=str(episode.class_name),
        support_id=str(episode.support_id),
        query_id=str(episode.query_id),
        support_image_path=Path(episode.support_image_path),
        support_mask_path=Path(episode.support_mask_path),
        query_image_path=Path(episode.query_image_path),
        query_mask_path=Path(episode.query_mask_path),
    )


def build_similarity_cache(
    cfg: dict, device: str, dino_model: torch.nn.Module | None = None
) -> DinoSimilarityCache:
    model_cfg = cfg["models"]
    if dino_model is None:
        dino_model = load_dino_model(model_cfg["dino_checkpoint"], device=device)
    return DinoSimilarityCache(
        cache_dir=model_cfg["sim_map_cache_dir"],
        dino_model=dino_model,
        dino_checkpoint=model_cfg["dino_checkpoint"],
        device=device,
        image_size=int(model_cfg["image_size"]),
        precision=resolve_dino_precision(
            str(model_cfg.get("dino_precision", model_cfg.get("precision", "fp32"))), device
        ),
    )


def build_query_feature_cache(
    cfg: dict, device: str, dino_model: torch.nn.Module | None = None
) -> DinoQueryFeatureCache | None:
    model_cfg = cfg["models"]
    cache_dir = model_cfg.get("dino_query_cache_dir")
    if not cache_dir:
        return None
    if dino_model is None:
        dino_model = load_dino_model(model_cfg["dino_checkpoint"], device=device)
    return DinoQueryFeatureCache(
        cache_dir=cache_dir,
        dino_model=dino_model,
        dino_checkpoint=model_cfg["dino_checkpoint"],
        device=device,
        image_size=int(model_cfg["image_size"]),
        precision=str(model_cfg.get("dino_precision", model_cfg.get("precision", "fp32"))),
    )


def build_bc_feature_caches(
    cfg: dict, device: str
) -> tuple[DinoSimilarityCache, DinoQueryFeatureCache | None]:
    if bool(cfg.get("model", {}).get("use_query_dino_features", False)):
        dino_model = load_dino_model(cfg["models"]["dino_checkpoint"], device=device)
        return (
            build_similarity_cache(cfg, device=device, dino_model=dino_model),
            build_query_feature_cache(cfg, device=device, dino_model=dino_model),
        )
    return build_similarity_cache(cfg, device=device), None


def ensure_query_feature_caches(
    episodes: Sequence[FewShotEpisode],
    query_feature_cache: DinoQueryFeatureCache | None,
    log_every: int = 100,
) -> None:
    if query_feature_cache is None:
        return
    total = 0
    for episode in episodes:
        query_feature_cache.ensure(episode)
        total += 1
        if total == 1 or total % max(1, int(log_every)) == 0:
            print(f"[dino-query] cached {total} episodes (latest: {episode.image_id})", flush=True)


def resolve_sam_spec(cfg: dict, purpose: str = "eval") -> tuple[str, str]:
    model_cfg = cfg["models"]
    purpose = str(purpose)
    if purpose == "oracle":
        checkpoint = model_cfg.get("oracle_sam_checkpoint", model_cfg.get("sam_checkpoint"))
        model_type = model_cfg.get("oracle_sam_model_type", model_cfg.get("sam_model_type"))
    elif purpose == "train":
        checkpoint = model_cfg.get(
            "train_sam_checkpoint",
            model_cfg.get("oracle_sam_checkpoint", model_cfg.get("sam_checkpoint")),
        )
        model_type = model_cfg.get(
            "train_sam_model_type",
            model_cfg.get("oracle_sam_model_type", model_cfg.get("sam_model_type")),
        )
    else:
        checkpoint = model_cfg.get("eval_sam_checkpoint", model_cfg.get("sam_checkpoint"))
        model_type = model_cfg.get("eval_sam_model_type", model_cfg.get("sam_model_type"))
    if checkpoint is None or model_type is None:
        raise ValueError(f"Missing SAM spec for purpose={purpose}.")
    return str(checkpoint), str(model_type)


def build_sam_predictor(cfg: dict, device: str, purpose: str = "eval"):
    checkpoint, model_type = resolve_sam_spec(cfg, purpose=purpose)
    return load_sam_predictor(
        checkpoint=checkpoint,
        model_type=model_type,
        device=device,
    )


def build_sam_embedding_cache(
    cfg: dict, predictor, device: str, purpose: str = "eval"
) -> SamImageEmbeddingCache | None:
    models = cfg["models"]
    cache_dir = models.get(
        f"{purpose}_sam_embedding_cache_dir", models.get("sam_embedding_cache_dir")
    )
    if not cache_dir:
        return None
    checkpoint, model_type = resolve_sam_spec(cfg, purpose=purpose)
    precision = str(models.get(f"{purpose}_precision", models.get("precision", "fp32")))
    return SamImageEmbeddingCache(
        cache_dir=cache_dir,
        predictor=predictor,
        sam_checkpoint=checkpoint,
        sam_model_type=model_type,
        image_size=int(models["image_size"]),
        precision=precision,
        cpu_cache_max_bytes=int(models.get("sam_embedding_cpu_cache_bytes", 0)),
    )


def set_cached_predictor_image(
    predictor,
    embedding_cache: SamImageEmbeddingCache | None,
    episode: FewShotEpisode,
    image: np.ndarray,
) -> None:
    if embedding_cache is None:
        predictor.set_image(image, image_format="RGB")
        return
    record = embedding_cache.load(episode, device="cpu")
    set_predictor_image_embedding(predictor, record)


def oracle_cache_path(oracle_dir: str | Path, episode: FewShotEpisode) -> Path:
    oracle_dir = Path(oracle_dir)
    oracle_dir.mkdir(parents=True, exist_ok=True)
    return oracle_dir / f"{episode.split}_{safe_slug(episode.image_id)}.json"


def load_oracle_payload(oracle_dir: str | Path, episode: FewShotEpisode) -> dict:
    path = oracle_cache_path(oracle_dir, episode)
    if not path.exists():
        raise FileNotFoundError(f"Missing oracle cache for {episode.image_id}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_oracle_caches(
    episodes: Sequence[FewShotEpisode],
    oracle_dir: str | Path,
    predictor,
    sim_cache: DinoSimilarityCache,
    grid: int,
    max_budget: int,
    candidate_batch_size: int,
    candidate_pool_size: int | None = None,
    use_bfloat16: bool = False,
    sam_checkpoint: str | None = None,
    sam_model_type: str | None = None,
    overwrite: bool = False,
    shard_index: int = 0,
    num_shards: int = 1,
    log_every: int = 25,
    decoder_policy: str = "single",
    sam_embedding_cache: SamImageEmbeddingCache | None = None,
) -> None:
    total = 0
    shard_index = int(shard_index)
    num_shards = int(num_shards)
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("num_shards must be positive and shard_index must be in range.")
    manifest_path = (
        Path(oracle_dir) / "shards" / f"shard-{shard_index:04d}-of-{num_shards:04d}.json"
    )
    ordered_ids = [episode_identity(ep) for ep in episodes]
    shard_ids = [
        image_id for index, image_id in enumerate(ordered_ids) if index % num_shards == shard_index
    ]
    manifest = {
        "schema_version": 1,
        "split": str(episodes[0].split) if episodes else "",
        "shard_index": shard_index,
        "num_shards": num_shards,
        "ordered_episode_ids": ordered_ids,
        "expected_episode_ids": shard_ids,
        "status": "running",
    }
    write_atomic_json(manifest_path, manifest)
    for episode_index, episode in enumerate(episodes):
        if (episode_index % num_shards) != shard_index:
            continue
        ensure_oracle_cache(
            episode=episode,
            oracle_dir=oracle_dir,
            predictor=predictor,
            sim_cache=sim_cache,
            grid=grid,
            max_budget=max_budget,
            candidate_batch_size=candidate_batch_size,
            candidate_pool_size=candidate_pool_size,
            use_bfloat16=use_bfloat16,
            sam_checkpoint=sam_checkpoint,
            sam_model_type=sam_model_type,
            overwrite=overwrite,
            decoder_policy=decoder_policy,
            sam_embedding_cache=sam_embedding_cache,
        )
        total += 1
        if total == 1 or total % max(1, int(log_every)) == 0:
            print(
                f"[oracle] cached {total} episodes in shard {shard_index + 1}/{num_shards} "
                f"(latest: {episode.image_id})",
                flush=True,
            )
    manifest["status"] = "complete"
    manifest["completed_episode_ids"] = shard_ids
    write_atomic_json(manifest_path, manifest)


@torch.inference_mode()
def ensure_oracle_cache(
    episode: FewShotEpisode,
    oracle_dir: str | Path,
    predictor,
    sim_cache: DinoSimilarityCache,
    grid: int,
    max_budget: int,
    candidate_batch_size: int,
    candidate_pool_size: int | None = None,
    use_bfloat16: bool = False,
    sam_checkpoint: str | None = None,
    sam_model_type: str | None = None,
    overwrite: bool = False,
    decoder_policy: str = "single",
    sam_embedding_cache: SamImageEmbeddingCache | None = None,
) -> Path:
    grid = int(grid)
    max_budget = int(max_budget)
    decoder_policy = normalize_decoder_policy(decoder_policy)
    path = oracle_cache_path(oracle_dir, episode)
    ckpt_fingerprint = (
        file_fingerprint(sam_checkpoint)
        if sam_checkpoint and Path(sam_checkpoint).is_file()
        else {}
    )
    expected_provenance = cache_metadata(
        kind="greedy_oracle",
        episode=episode,
        config={
            "grid": grid,
            "max_budget": max_budget,
            "candidate_pool_size": candidate_pool_size,
            "decoder_policy": decoder_policy,
            "coordinate_policy": "rounded_pixel_centers_v1",
            "mask_resize": "bilinear_align_corners_false",
            "sam_model_type": sam_model_type,
            "sam_checkpoint": ckpt_fingerprint,
            "similarity_cache": sim_cache._metadata(episode),
            "image_size": int(sim_cache.image_size),
            "precision": "bf16" if use_bfloat16 else "fp32",
            "candidate_domain": "all_unused_grid_cells"
            if candidate_pool_size is None
            else "top_similarity_pool",
        },
    )
    if path.exists() and not overwrite:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("provenance") == expected_provenance
            and payload.get("episode_key") == episode_identity(episode)
            and int(payload.get("grid", -1)) == grid
            and str(payload.get("sam_prompt_mode", "")) == "one_shot_positive_points"
            and str(payload.get("oracle_sam_checkpoint", "")) == str(sam_checkpoint or "")
            and str(payload.get("oracle_sam_model_type", "")) == str(sam_model_type or "")
            and len(payload.get("steps", [])) >= max_budget
        ):
            return path

    _, _, query_rgb, query_mask = load_episode_arrays(episode, image_size=int(sim_cache.image_size))
    sim_map = sim_cache.load(episode)
    sim_grid = scalar_map_to_grid_tensor(sim_map, grid).squeeze(0).numpy()
    target_mask = torch.from_numpy(np.asarray(query_mask, dtype=bool).copy()).to(
        device=predictor.device
    )
    autocast_enabled = bool(use_bfloat16) and str(predictor.device).startswith("cuda")
    if sam_embedding_cache is None:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            predictor.set_image(query_rgb, image_format="RGB")
    else:
        set_cached_predictor_image(predictor, sam_embedding_cache, episode, query_rgb)
    query_h, query_w = query_mask.shape
    prefix_indices: list[int] = []
    prefix_points_px: list[tuple[float, float]] = []
    steps: list[dict[str, object]] = []

    for step_index in range(max_budget):
        candidate_indices = candidate_index_pool(
            sim_grid=sim_grid,
            chosen_indices=prefix_indices,
            candidate_pool_size=candidate_pool_size,
        )
        if not candidate_indices:
            break
        step_start = time.perf_counter()
        start_batch_size = (
            len(candidate_indices) if int(candidate_batch_size) <= 0 else int(candidate_batch_size)
        )
        working_batch_size = max(1, min(start_batch_size, len(candidate_indices)))
        best_idx = -1
        best_mask: np.ndarray | None = None
        best_iou = -1.0
        best_sim = -1.0

        while True:
            try:
                best_idx = -1
                best_mask = None
                best_iou = -1.0
                best_sim = -1.0
                candidate_batches = chunked(candidate_indices, working_batch_size)
                for batch_indices in candidate_batches:
                    batch_coords = build_candidate_coord_batch(
                        candidate_indices=batch_indices,
                        prefix_points_px=prefix_points_px,
                        grid=grid,
                        image_width=query_w,
                        image_height=query_h,
                        device=predictor.device,
                    )
                    batch_labels = torch.ones(
                        batch_coords.shape[:2], dtype=torch.int, device=predictor.device
                    )
                    transformed_coords = predictor.transform.apply_coords_torch(
                        batch_coords, predictor.original_size
                    )
                    with torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
                    ):
                        masks, scores, _low_res = predictor.predict_torch(
                            point_coords=transformed_coords,
                            point_labels=batch_labels,
                            boxes=None,
                            mask_input=None,
                            multimask_output=multimask_for_prefix(
                                decoder_policy, len(prefix_points_px) + 1
                            ),
                            return_logits=False,
                        )
                    if multimask_for_prefix(decoder_policy, len(prefix_points_px) + 1):
                        best_mask_choice = scores.argmax(dim=1)
                        gather = torch.arange(masks.shape[0], device=masks.device)
                        selected_masks = masks[gather, best_mask_choice]
                    else:
                        selected_masks = masks[:, 0]
                    batch_ious = batch_iou(selected_masks, target_mask)
                    for local_index, location_idx in enumerate(batch_indices):
                        iou_value = float(batch_ious[local_index].item())
                        sim_value = float(sim_grid.reshape(-1)[int(location_idx)])
                        if not is_better_candidate(
                            iou_value, sim_value, int(location_idx), best_iou, best_sim, best_idx
                        ):
                            continue
                        best_iou = iou_value
                        best_idx = int(location_idx)
                        best_sim = sim_value
                        best_mask = (
                            selected_masks[local_index].detach().to("cpu").numpy().astype(bool)
                        )
                break
            except torch.OutOfMemoryError:
                if working_batch_size <= 1:
                    raise
                next_batch_size = max(1, working_batch_size // 2)
                if next_batch_size == working_batch_size:
                    raise
                print(
                    f"[oracle] OOM at candidate batch_size={working_batch_size}; retrying with {next_batch_size}",
                    flush=True,
                )
                working_batch_size = next_batch_size
                if str(predictor.device).startswith("cuda"):
                    torch.cuda.empty_cache()

        if best_idx < 0 or best_mask is None:
            raise RuntimeError(
                f"Failed to find greedy oracle click for {episode.image_id} step {step_index + 1}."
            )
        gx, gy = location_idx_to_grid_xy(best_idx, grid)
        px, py = rounded_grid_xy_to_pixel_xy(
            gx, gy, image_width=query_w, image_height=query_h, grid=grid
        )
        prefix_indices.append(best_idx)
        prefix_points_px.append((px, py))
        steps.append(
            {
                "step_index": step_index + 1,
                "location_idx": best_idx,
                "grid_x": gx,
                "grid_y": gy,
                "point_x": round(float(px), 3),
                "point_y": round(float(py), 3),
                "iou": round(float(iou_score(best_mask, query_mask)), 6),
                "dice": round(float(dice_score(best_mask, query_mask)), 6),
                "boundary_iou": round(float(boundary_iou_score(best_mask, query_mask)), 6),
                "sim_value": round(best_sim, 6),
                "wallclock_ms": round((time.perf_counter() - step_start) * 1000.0, 3),
            }
        )

    payload = {
        "dataset": episode.dataset,
        "split": episode.split,
        "class_name": episode.class_name,
        "image_id": episode.image_id,
        "episode_key": episode_identity(episode),
        "grid": grid,
        "image_size": [int(query_h), int(query_w)],
        "max_budget": int(max_budget),
        "candidate_pool_size": None if candidate_pool_size is None else int(candidate_pool_size),
        "oracle_sam_checkpoint": str(sam_checkpoint or ""),
        "oracle_sam_model_type": str(sam_model_type or ""),
        "sam_prompt_mode": "one_shot_positive_points",
        "decoder_policy": decoder_policy,
        "coordinate_policy": "rounded_pixel_centers_v1",
        "mask_resize": "bilinear_align_corners_false",
        "provenance": expected_provenance,
        "steps": steps,
    }
    atomic_write_json(path, payload)
    return path


def evaluate_policy(
    model: nn.Module,
    episodes: Sequence[FewShotEpisode],
    predictor,
    sim_cache: DinoSimilarityCache,
    query_feature_cache: DinoQueryFeatureCache | None,
    budget: int,
    grid: int,
    image_size: int,
    device: str,
    output_csv: str | Path | None = None,
    overlay_dir: str | Path | None = None,
    overlay_limit: int = 0,
    method_name: str = "greedy_bc",
    decoder_policy: str = "single",
    training_seed: int = 0,
    eval_seed: int = 0,
    result_metadata: dict | None = None,
    sam_embedding_cache: SamImageEmbeddingCache | None = None,
    final_mask_sample_count: int = 0,
) -> dict[str, float]:
    if output_csv is not None and result_metadata is None:
        raise ValueError("result_metadata is required when writing a BC result CSV.")
    if final_mask_sample_count < 0:
        raise ValueError("final_mask_sample_count must be nonnegative.")
    if final_mask_sample_count and output_csv is None:
        raise ValueError("Final mask samples require a result CSV and metadata sidecar.")
    model.eval()
    writer = CsvResultWriter(output_csv) if output_csv is not None else None
    overlay_count = 0
    per_step_rows: list[dict[str, object]] = []
    final_mask_entries: list[tuple[FewShotEpisode, np.ndarray]] = []
    sums = {
        "mean_iou": 0.0,
        "mean_dice": 0.0,
        "mean_boundary_iou": 0.0,
        "mean_wallclock_ms": 0.0,
    }
    for episode in episodes:
        step_ious: list[float] = []
        result = rollout_episode(
            model=model,
            episode=episode,
            predictor=predictor,
            sim_cache=sim_cache,
            query_feature_cache=query_feature_cache,
            budget=budget,
            grid=grid,
            image_size=image_size,
            device=device,
            decoder_policy=decoder_policy,
            per_step_ious=step_ious,
            sam_embedding_cache=sam_embedding_cache,
        )
        if len(final_mask_entries) < int(final_mask_sample_count):
            final_mask_entries.append((episode, np.asarray(result["pred_mask"]).copy()))
        points_xy = np.asarray(result["points_xy"], dtype=np.int32)
        for step_index, step_iou in enumerate(step_ious, start=1):
            per_step_rows.append(
                {
                    "episode_key": episode_identity(episode),
                    "training_seed": int(training_seed),
                    "eval_seed": int(eval_seed),
                    "step": step_index,
                    "iou": float(step_iou),
                    "point_x": int(points_xy[step_index - 1, 0]),
                    "point_y": int(points_xy[step_index - 1, 1]),
                    "point_label": 1,
                }
            )
        if writer is not None:
            writer.write(
                build_bc_result_row(
                    method=method_name,
                    budget=budget,
                    episode=episode,
                    training_seed=training_seed,
                    eval_seed=eval_seed,
                    result=result,
                )
            )
        if overlay_dir is not None and overlay_count < int(overlay_limit):
            save_overlay_artifacts(
                output_dir=overlay_dir,
                episode=episode,
                method=method_name,
                n_points=budget,
                seed=0,
                query_rgb=result["query_rgb"],
                sim_map=result["sim_map"],
                pred_mask=result["pred_mask"],
                target_mask=result["target_mask"],
                points=result["points_xy"],
                metrics={
                    "iou": result["iou"],
                    "dice": result["dice"],
                    "boundary_iou": result["boundary_iou"],
                    "wallclock_ms": result["wallclock_ms"],
                },
            )
            overlay_count += 1
        sums["mean_iou"] += float(result["iou"])
        sums["mean_dice"] += float(result["dice"])
        sums["mean_boundary_iou"] += float(result["boundary_iou"])
        sums["mean_wallclock_ms"] += float(result["wallclock_ms"])
    if writer is not None:
        writer.close()
    denom = max(len(episodes), 1)
    summary = {key: value / denom for key, value in sums.items()}
    summary["episodes"] = float(len(episodes))
    if output_csv is not None:
        assert result_metadata is not None
        _write_per_step_csv(Path(output_csv).with_suffix(".per_step.csv"), per_step_rows)
        metadata_extra = {
            "budget": int(budget),
            "protocol": decoder_policy,
            "grid": int(grid),
            "image_size": int(image_size),
            "method": method_name,
            **dict(result_metadata.get("extra", {})),
            **summary,
        }
        write_result_metadata(
            output_csv,
            episodes=episodes,
            dataset=str(result_metadata["dataset"]),
            split=str(result_metadata["split"]),
            condition=str(result_metadata["condition"]),
            model_config=result_metadata["model_config"],
            expected_result_rows=len(episodes),
            training_seed=int(training_seed),
            eval_seed=int(eval_seed),
            decoder=str(result_metadata.get("decoder", decoder_policy)),
            model_state=model.state_dict(),
            extra=metadata_extra,
        )
        write_result_metadata(
            Path(output_csv).with_suffix(".per_step.csv"),
            episodes=episodes,
            dataset=str(result_metadata["dataset"]),
            split=str(result_metadata["split"]),
            condition=str(result_metadata["condition"]),
            model_config=result_metadata["model_config"],
            expected_result_rows=len(episodes) * int(budget),
            training_seed=int(training_seed),
            eval_seed=int(eval_seed),
            decoder=str(result_metadata.get("decoder", decoder_policy)),
            model_state=model.state_dict(),
            extra=metadata_extra,
        )
        if final_mask_sample_count:
            write_final_mask_samples(
                output_dir=mask_sample_dir_for_csv(output_csv),
                episodes=episodes,
                mask_entries=final_mask_entries,
                sample_count=final_mask_sample_count,
                result_csv=output_csv,
            )
    return summary


def build_bc_result_row(
    *,
    method: str,
    budget: int,
    episode: FewShotEpisode,
    training_seed: int,
    eval_seed: int,
    result: Mapping[str, object],
) -> dict[str, object]:
    return {
        "method": method,
        "N": int(budget),
        "image_id": episode.image_id,
        "episode_key": episode_identity(episode),
        "seed": int(training_seed),
        "training_seed": int(training_seed),
        "eval_seed": int(eval_seed),
        "iou": result["iou"],
        "dice": result["dice"],
        "boundary_iou": result["boundary_iou"],
        "wallclock_ms": result["wallclock_ms"],
    }


def _write_per_step_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode_key",
        "training_seed",
        "eval_seed",
        "step",
        "iou",
        "point_x",
        "point_y",
        "point_label",
    ]
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


@torch.inference_mode()
def rollout_episode(
    model: nn.Module,
    episode: FewShotEpisode,
    predictor,
    sim_cache: DinoSimilarityCache,
    query_feature_cache: DinoQueryFeatureCache | None,
    budget: int,
    grid: int,
    image_size: int,
    device: str,
    decoder_policy: str = "single",
    per_step_ious: list[float] | None = None,
    sam_embedding_cache: SamImageEmbeddingCache | None = None,
) -> RolloutResult:
    _, _, query_rgb, query_mask = load_episode_arrays(episode, image_size=image_size)
    sim_map = sim_cache.load(episode)
    query_dino = query_feature_cache.load(episode) if query_feature_cache is not None else None
    start = time.perf_counter()
    prefix_masks: list[np.ndarray] = []
    points_xy = select_points_with_model(
        model=model,
        sim_map=sim_map,
        budget=budget,
        device=device,
        query_rgb=query_rgb if getattr(model, "use_query_rgb", False) else None,
        query_dino=query_dino,
        predictor=predictor,
        query_image=query_rgb,
        decoder_policy=decoder_policy,
        prefix_masks=prefix_masks,
        episode=episode,
        sam_embedding_cache=sam_embedding_cache,
    )
    if not prefix_masks:
        raise RuntimeError("BC evaluation did not decode any click prefix.")
    pred_mask = prefix_masks[-1]
    if per_step_ious is not None:
        per_step_ious.extend(prefix_iou_scores(prefix_masks, query_mask))

    wallclock_ms = (time.perf_counter() - start) * 1000.0
    return {
        "query_rgb": query_rgb,
        "sim_map": sim_map,
        "pred_mask": pred_mask,
        "target_mask": query_mask,
        "points_xy": points_xy,
        "iou": float(iou_score(pred_mask, query_mask)),
        "dice": float(dice_score(pred_mask, query_mask)),
        "boundary_iou": float(boundary_iou_score(pred_mask, query_mask)),
        "wallclock_ms": float(wallclock_ms),
    }


def prefix_iou_scores(prefix_masks: Sequence[np.ndarray], target_mask: np.ndarray) -> list[float]:
    return [float(iou_score(mask, target_mask)) for mask in prefix_masks]


def evaluate_loss(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    loss_cfg: dict | None = None,
) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    for batch in loader:
        obs = {key: value.to(device) for key, value in batch["obs"].items()}
        target_idx = batch["target_idx"].to(device)
        logits, invalid_mask = mask_clicked_logits(model(obs), obs["click_map"])
        loss = heatmap_loss(logits, target_idx, loss_cfg=loss_cfg, invalid_mask=invalid_mask)
        total_loss += float(loss.item()) * int(target_idx.shape[0])
        total_examples += int(target_idx.shape[0])
    return total_loss / max(total_examples, 1)


def warm_start_similarity(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epochs: int,
) -> list[dict[str, float | int]]:
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(epochs) + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch in train_loader:
            obs = {key: value.to(device) for key, value in batch["obs"].items()}
            logits, invalid_mask = mask_clicked_logits(model(obs), obs["click_map"])
            target = obs["sim_map"].flatten(1).float().clamp(0.0, 1.0)
            loss = masked_binary_cross_entropy_with_logits(
                logits, target, invalid_mask=invalid_mask
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_size = int(target.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
        mean_loss = total_loss / max(total_examples, 1)
        entry = {
            "warm_start_epoch": epoch,
            "warm_start_loss": mean_loss,
        }
        history.append(entry)
        print(f"[warm-start] epoch {epoch}/{epochs} loss={mean_loss:.4f}", flush=True)
    return history


def train_click_policy(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    val_episodes: Sequence[FewShotEpisode],
    predictor,
    sim_cache: DinoSimilarityCache,
    query_feature_cache: DinoQueryFeatureCache | None,
    cfg: dict,
    budget: int,
    device: str,
    run_dir: str | Path,
    resume_path: str | Path | None = None,
    sam_embedding_cache: SamImageEmbeddingCache | None = None,
) -> dict[str, object]:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = cfg.get("train", {})
    val_cfg = cfg.get("validation", {})
    cfg.get("evaluation", {})
    model_cfg = cfg.get("model", {})
    model_cfg = dict(model_cfg)
    model_cfg.setdefault("use_current_mask", False)
    model_cfg.setdefault("reserve_mask_channel", bool(model_cfg.get("use_current_mask", False)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 3e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(train_cfg.get("epochs", 12))
    max_grad_norm = float(train_cfg.get("max_grad_norm", 1.0))
    evaluate_every = max(1, int(val_cfg.get("evaluate_every_epochs", 1)))
    best_metric = float("-inf")
    best_epoch = 0
    history: list[dict[str, float | int]] = []
    start_epoch = 1
    best_checkpoint_path = run_dir / "best.pt"
    last_checkpoint_path = run_dir / "last.pt"
    best_val_summary: dict[str, float] | None = None
    warm_start_epochs = int(train_cfg.get("warm_start_epochs", 0))
    aux_cfg = dict(train_cfg.get("aux_rollout_iou", {}))
    aux_cfg.setdefault("decoder_policy", cfg.get("protocol", {}).get("decoder_policy", "single"))
    train_dataset = getattr(train_loader, "dataset", None)
    config_identity = training_config_identity(cfg)

    if resume_path is not None:
        state = torch.load(resume_path, map_location=device, weights_only=False)
        if state.get("config_identity", config_identity) != config_identity:
            raise ValueError("Resume checkpoint scientific configuration does not match this run.")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        start_epoch = int(state["epoch"]) + 1
        best_metric = float(state.get("best_metric", best_metric))
        best_epoch = int(state.get("best_epoch", 0))
        history = list(state.get("history", []))
        selected_entry = next(
            (entry for entry in history if int(entry.get("epoch", -1)) == best_epoch), None
        )
        if selected_entry is not None:
            best_val_summary = {
                "mean_iou": float(selected_entry.get("val_rollout_iou", 0.0)),
                "mean_dice": float(selected_entry.get("val_rollout_dice", 0.0)),
                "mean_boundary_iou": float(selected_entry.get("val_rollout_boundary_iou", 0.0)),
            }
        if "python_rng_state" in state:
            random.setstate(state["python_rng_state"])
            np.random.set_state(state["numpy_rng_state"])
            torch.set_rng_state(state["torch_rng_state"].cpu())
            if torch.cuda.is_available() and state.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(
                    [rng_state.cpu() for rng_state in state["cuda_rng_state"]]
                )

    if resume_path is None and warm_start_epochs > 0:
        history.extend(
            warm_start_similarity(
                model, train_loader, optimizer, device=device, epochs=warm_start_epochs
            )
        )
        write_json(run_dir / "history.json", {"budget": int(budget), "history": history})

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_supervised_loss = 0.0
        epoch_aux_loss = 0.0
        epoch_examples = 0
        for batch in train_loader:
            obs = {key: value.to(device) for key, value in batch["obs"].items()}
            target_idx = batch["target_idx"].to(device)
            logits, invalid_mask = mask_clicked_logits(model(obs), obs["click_map"])
            supervised_loss = heatmap_loss(
                logits, target_idx, loss_cfg=train_cfg.get("loss"), invalid_mask=invalid_mask
            )
            aux_loss = logits.new_zeros(())
            aux_weight = float(aux_cfg.get("weight", 0.0))
            if aux_weight > 0.0 and isinstance(train_dataset, OracleHeatmapDataset):
                aux_loss = rollout_iou_aux_loss(
                    masked_logits=logits,
                    batch=batch,
                    dataset=train_dataset,
                    predictor=predictor,
                    image_size=int(cfg["models"]["image_size"]),
                    grid=int(model_cfg.get("grid", 37)),
                    aux_cfg=aux_cfg,
                    sam_embedding_cache=sam_embedding_cache,
                )
            loss = supervised_loss + (aux_weight * aux_loss)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            batch_size = int(target_idx.shape[0])
            epoch_loss += float(loss.item()) * batch_size
            epoch_supervised_loss += float(supervised_loss.item()) * batch_size
            epoch_aux_loss += float(aux_loss.item()) * batch_size
            epoch_examples += batch_size

        train_loss = epoch_loss / max(epoch_examples, 1)
        entry: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_supervised_loss": epoch_supervised_loss / max(epoch_examples, 1),
            "train_aux_loss": epoch_aux_loss / max(epoch_examples, 1),
        }
        if val_loader is not None:
            entry["val_loss"] = evaluate_loss(
                model, val_loader, device=device, loss_cfg=train_cfg.get("loss")
            )
        if epoch % evaluate_every == 0 and val_episodes and predictor is not None:
            eval_episodes = list(val_episodes)
            max_eval_episodes = val_cfg.get("eval_max_episodes")
            if max_eval_episodes is not None:
                eval_episodes = eval_episodes[: int(max_eval_episodes)]
            val_summary = evaluate_policy(
                model=model,
                episodes=eval_episodes,
                predictor=predictor,
                sim_cache=sim_cache,
                query_feature_cache=query_feature_cache,
                budget=budget,
                grid=int(model_cfg.get("grid", 37)),
                image_size=int(cfg["models"]["image_size"]),
                device=device,
                output_csv=None,
                overlay_dir=None,
                overlay_limit=0,
                decoder_policy=str(cfg.get("protocol", {}).get("decoder_policy", "single")),
                training_seed=int(cfg.get("seed", 0)),
                eval_seed=int(cfg.get("evaluation", {}).get("eval_seed", 0)),
            )
            entry.update(
                {
                    "val_rollout_iou": float(val_summary["mean_iou"]),
                    "val_rollout_dice": float(val_summary["mean_dice"]),
                    "val_rollout_boundary_iou": float(val_summary["mean_boundary_iou"]),
                }
            )
            metric_name = str(val_cfg.get("checkpoint_metric", "mean_iou"))
            metric_key = (
                f"mean_{metric_name}"
                if not str(metric_name).startswith("mean_")
                else str(metric_name)
            )
            current_metric = float(val_summary.get(metric_key, val_summary["mean_iou"]))
            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch
                best_val_summary = dict(val_summary)
                atomic_torch_save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_config": model_cfg,
                        "budget": int(budget),
                        "epoch": int(epoch),
                        "best_metric": float(best_metric),
                        "config_identity": config_identity,
                        "training_seed": int(cfg.get("seed", 0)),
                        "train_config": train_cfg,
                    },
                    best_checkpoint_path,
                )
        history.append(entry)
        write_json(run_dir / "history.json", {"budget": int(budget), "history": history})
        atomic_torch_save(
            {
                "model_state_dict": model.state_dict(),
                "model_config": model_cfg,
                "optimizer_state_dict": optimizer.state_dict(),
                "budget": int(budget),
                "epoch": int(epoch),
                "best_metric": float(best_metric),
                "best_epoch": int(best_epoch),
                "config_identity": config_identity,
                "training_seed": int(cfg.get("seed", 0)),
                "train_config": train_cfg,
                "history": history,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
            },
            last_checkpoint_path,
        )
        print(
            f"[train] N={budget} epoch {epoch}/{epochs} train_loss={train_loss:.4f}"
            + (f" val_iou={entry['val_rollout_iou']:.4f}" if "val_rollout_iou" in entry else ""),
            flush=True,
        )

    return {
        "best_checkpoint": str(best_checkpoint_path),
        "last_checkpoint": str(last_checkpoint_path),
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
        "best_val_summary": best_val_summary or {},
    }


def load_click_policy(checkpoint_path: str | Path, device: str) -> tuple[nn.Module, dict]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_policy(payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, payload


def atomic_torch_save(payload: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def training_config_identity(cfg: dict) -> str:
    identity = json.loads(json.dumps(cfg, sort_keys=True, default=str))
    identity.pop("paths", None)
    train = identity.get("train", {})
    if isinstance(train, dict):
        train.pop("epochs", None)
        train.pop("device", None)
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


class BudgetedGreedyBCSelector:
    def __init__(self, checkpoints_by_budget: dict[int, str | Path], device: str = "cpu") -> None:
        self.checkpoints_by_budget = {
            int(key): Path(value) for key, value in checkpoints_by_budget.items()
        }
        self.device = str(device)
        self._models: dict[int, nn.Module] = {}

    def select_points(
        self, sim_map: np.ndarray, n_points: int, rng: np.random.Generator
    ) -> np.ndarray:
        del rng
        budget = int(n_points)
        if budget not in self.checkpoints_by_budget:
            raise KeyError(f"No checkpoint configured for budget N={budget}.")
        model = self._models.get(budget)
        if model is None:
            model, _ = load_click_policy(self.checkpoints_by_budget[budget], device=self.device)
            self._models[budget] = model
        return select_points_with_model(
            model=model, sim_map=sim_map, budget=budget, device=self.device
        )


@torch.inference_mode()
def select_points_with_model(
    model: nn.Module,
    sim_map: np.ndarray,
    budget: int,
    device: str,
    query_rgb: np.ndarray | None = None,
    query_dino: np.ndarray | torch.Tensor | None = None,
    predictor=None,
    query_image: np.ndarray | None = None,
    decoder_policy: str = "single",
    prefix_masks: list[np.ndarray] | None = None,
    episode: FewShotEpisode | None = None,
    sam_embedding_cache: SamImageEmbeddingCache | None = None,
) -> np.ndarray:
    grid = int(getattr(model, "grid", 37))
    budget = int(budget)
    sim_grid = scalar_map_to_grid_tensor(sim_map, grid).unsqueeze(0).to(device)
    click_map = torch.zeros(1, 1, grid, grid, dtype=torch.float32, device=device)
    current_mask = torch.zeros_like(click_map)
    selected_indices: list[int] = []
    query_rgb_tensor = None
    if getattr(model, "use_query_rgb", False):
        if query_rgb is None:
            raise ValueError("Model requires query_rgb, but no query_rgb was provided.")
        query_rgb_tensor = rgb_to_grid_tensor(query_rgb, grid).unsqueeze(0).to(device)
    query_dino_tensor = None
    if getattr(model, "use_query_dino_features", False):
        if query_dino is None:
            raise ValueError("Model requires query_dino features, but none were provided.")
        if isinstance(query_dino, torch.Tensor):
            query_dino_tensor = query_dino.unsqueeze(0) if query_dino.dim() == 3 else query_dino
            query_dino_tensor = query_dino_tensor.to(device=device, dtype=torch.float32)
        else:
            query_dino_tensor = (
                torch.from_numpy(np.asarray(query_dino, dtype=np.float32).copy())
                .unsqueeze(0)
                .to(device)
            )
    if predictor is not None and query_image is not None:
        if sam_embedding_cache is not None and episode is not None:
            set_cached_predictor_image(predictor, sam_embedding_cache, episode, query_image)
        else:
            predictor.set_image(query_image, image_format="RGB")
    for step_index in range(1, budget + 1):
        obs = {
            "sim_map": sim_grid,
            "click_map": click_map,
            "step_index": torch.tensor([[float(step_index)]], dtype=torch.float32, device=device),
            "budget": torch.tensor([[float(budget)]], dtype=torch.float32, device=device),
        }
        if getattr(model, "reserve_mask_channel", False):
            obs["current_mask"] = current_mask
        if query_rgb_tensor is not None:
            obs["query_rgb"] = query_rgb_tensor
        if query_dino_tensor is not None:
            obs["query_dino"] = query_dino_tensor
        logits, _ = mask_clicked_logits(model(obs), click_map)
        next_idx = int(torch.argmax(logits, dim=-1).item())
        selected_indices.append(next_idx)
        click_map.view(-1)[next_idx] = 1.0
        if predictor is not None and query_image is not None:
            prefix_points = []
            h, w = np.asarray(query_image).shape[:2]
            for idx in selected_indices:
                gx, gy = location_idx_to_grid_xy(idx, grid)
                prefix_points.append(rounded_grid_xy_to_pixel_xy(gx, gy, w, h, grid))
            decoded = predict_mask_from_points(
                predictor,
                np.asarray(prefix_points, dtype=np.float32),
                decoder_policy=decoder_policy,
            )
            if prefix_masks is not None:
                prefix_masks.append(decoded)
            if getattr(model, "use_current_mask", False):
                current_mask = (
                    scalar_map_to_grid_tensor(np.asarray(decoded, dtype=np.float32), grid)
                    .unsqueeze(0)
                    .to(device)
                )
    height, width = np.asarray(sim_map).shape[:2]
    points = []
    for location_idx in selected_indices:
        gx, gy = location_idx_to_grid_xy(location_idx, grid)
        px, py = rounded_grid_xy_to_pixel_xy(
            gx, gy, image_width=width, image_height=height, grid=grid
        )
        points.append((int(px), int(py)))
    return np.asarray(points, dtype=np.int32)


def safe_slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(text)).strip("_")


def location_idx_to_grid_xy(location_idx: int, grid: int) -> tuple[int, int]:
    return int(location_idx) % int(grid), int(location_idx) // int(grid)


def grid_xy_to_pixel_xy(
    grid_x: int, grid_y: int, image_width: int, image_height: int, grid: int
) -> tuple[float, float]:
    px = ((float(grid_x) + 0.5) / float(grid)) * float(image_width)
    py = ((float(grid_y) + 0.5) / float(grid)) * float(image_height)
    return float(np.clip(px, 0.0, image_width - 1.0)), float(np.clip(py, 0.0, image_height - 1.0))


def rounded_grid_xy_to_pixel_xy(
    grid_x: int, grid_y: int, image_width: int, image_height: int, grid: int
) -> tuple[float, float]:
    """Match PPO's rounded, clamped integer-center prompt coordinates."""
    px, py = grid_xy_to_pixel_xy(grid_x, grid_y, image_width, image_height, grid)
    return float(round(px)), float(round(py))


def candidate_index_pool(
    sim_grid: np.ndarray,
    chosen_indices: Sequence[int],
    candidate_pool_size: int | None,
) -> list[int]:
    flat = np.asarray(sim_grid, dtype=np.float32).reshape(-1)
    keep = np.ones(flat.shape[0], dtype=bool)
    if chosen_indices:
        keep[list(chosen_indices)] = False
    available = np.flatnonzero(keep)
    if candidate_pool_size is not None and int(candidate_pool_size) < int(available.shape[0]):
        candidate_pool_size = int(candidate_pool_size)
        top_local = np.argpartition(flat[available], -candidate_pool_size)[-candidate_pool_size:]
        available = available[top_local]
    order = np.argsort(flat[available])[::-1]
    return [int(idx) for idx in available[order]]


def build_candidate_coord_batch(
    candidate_indices: Sequence[int],
    prefix_points_px: Sequence[tuple[float, float]],
    grid: int,
    image_width: int,
    image_height: int,
    device: torch.device,
) -> torch.Tensor:
    batch = torch.empty(
        (len(candidate_indices), len(prefix_points_px) + 1, 2), dtype=torch.float32, device=device
    )
    if prefix_points_px:
        prefix = torch.tensor(prefix_points_px, dtype=torch.float32, device=device)
        batch[:, :-1, :] = prefix.unsqueeze(0).expand(len(candidate_indices), -1, -1)
    coords = []
    for location_idx in candidate_indices:
        gx, gy = location_idx_to_grid_xy(int(location_idx), grid)
        coords.append(
            rounded_grid_xy_to_pixel_xy(
                gx, gy, image_width=image_width, image_height=image_height, grid=grid
            )
        )
    batch[:, -1, :] = torch.tensor(coords, dtype=torch.float32, device=device)
    return batch


def batch_iou(pred_masks: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    pred = pred_masks.bool()
    target = target_mask.bool().unsqueeze(0)
    inter = torch.logical_and(pred, target).flatten(1).sum(dim=1).float()
    union = torch.logical_or(pred, target).flatten(1).sum(dim=1).float()
    return torch.where(union > 0.0, inter / union, torch.ones_like(inter))


def is_better_candidate(
    iou_value: float,
    sim_value: float,
    location_idx: int,
    best_iou: float,
    best_sim: float,
    best_idx: int,
) -> bool:
    if iou_value > best_iou + 1e-9:
        return True
    if iou_value < best_iou - 1e-9:
        return False
    if sim_value > best_sim + 1e-9:
        return True
    if sim_value < best_sim - 1e-9:
        return False
    return int(location_idx) < int(best_idx) or int(best_idx) < 0


def chunked(items: Sequence[int], chunk_size: int) -> Iterable[list[int]]:
    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(items), chunk_size):
        yield [int(value) for value in items[start : start + chunk_size]]


def atomic_write_json(path: str | Path, payload: dict) -> None:
    write_atomic_json(path, payload)


def infer_grid_size(logits: torch.Tensor) -> int:
    grid = round(float(logits.shape[-1]) ** 0.5)
    if grid * grid != int(logits.shape[-1]):
        raise ValueError(f"Expected square heatmap logits, got {tuple(logits.shape)}.")
    return grid


def gaussian_target_map(
    target_idx: torch.Tensor,
    grid: int,
    sigma: float,
    device: torch.device | str,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = torch.arange(int(grid), device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    gx = (target_idx.long() % int(grid)).to(dtype=dtype).view(-1, 1, 1)
    gy = (
        torch.div(target_idx.long(), int(grid), rounding_mode="floor")
        .to(dtype=dtype)
        .view(-1, 1, 1)
    )
    sigma = max(float(sigma), 1e-6)
    dist2 = torch.square(xx.unsqueeze(0) - gx) + torch.square(yy.unsqueeze(0) - gy)
    return torch.exp(-0.5 * dist2 / (sigma * sigma))


def mask_clicked_logits(
    logits: torch.Tensor, click_map: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    invalid_mask = ensure_single_channel(click_map).flatten(1) > 0.5
    if invalid_mask.any():
        logits = logits.masked_fill(invalid_mask, -1e9)
    return logits, invalid_mask


def masked_binary_cross_entropy_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    invalid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if invalid_mask is None or not invalid_mask.any():
        return F.binary_cross_entropy_with_logits(logits, target)
    valid_mask = ~invalid_mask
    if not valid_mask.any():
        return logits.new_zeros(())
    return F.binary_cross_entropy_with_logits(logits[valid_mask], target[valid_mask])


def rollout_iou_aux_loss(
    masked_logits: torch.Tensor,
    batch: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    dataset: OracleHeatmapDataset,
    predictor,
    image_size: int,
    grid: int,
    aux_cfg: dict,
    sam_embedding_cache: SamImageEmbeddingCache | None = None,
) -> torch.Tensor:
    if not bool(aux_cfg.get("enabled", False)):
        return masked_logits.new_zeros(())
    subset_size = min(int(aux_cfg.get("subset_size", 0)), int(masked_logits.shape[0]))
    if subset_size <= 0:
        return masked_logits.new_zeros(())
    topk = max(1, int(aux_cfg.get("topk", 4)))
    temperature = max(float(aux_cfg.get("temperature", 1.0)), 1e-6)
    use_bfloat16 = bool(aux_cfg.get("use_bfloat16", True))
    example_index = batch.get("example_index")
    if not isinstance(example_index, torch.Tensor):
        return masked_logits.new_zeros(())
    selected_rows = torch.randperm(int(masked_logits.shape[0]), device=masked_logits.device)[
        :subset_size
    ].tolist()
    grouped_examples: dict[str, list[tuple[int, OracleHeatmapExample]]] = defaultdict(list)
    for row_idx in selected_rows:
        example = dataset.examples[int(example_index[row_idx].item())]
        grouped_examples[example.episode.image_id].append((int(row_idx), example))

    losses: list[torch.Tensor] = []
    autocast_enabled = bool(use_bfloat16) and str(predictor.device).startswith("cuda")
    for items in grouped_examples.values():
        episode = items[0][1].episode
        _, _, query_rgb, query_mask = load_episode_arrays(episode, image_size=int(image_size))
        target_mask = torch.from_numpy(np.asarray(query_mask, dtype=bool).copy()).to(
            device=predictor.device
        )
        with torch.no_grad():
            if sam_embedding_cache is None:
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
                ):
                    predictor.set_image(query_rgb, image_format="RGB")
            else:
                set_cached_predictor_image(predictor, sam_embedding_cache, episode, query_rgb)
        for row_idx, example in items:
            remaining = int(masked_logits.shape[-1]) - len(example.prefix_indices)
            candidate_count = min(int(topk), max(remaining, 0))
            if candidate_count <= 0:
                continue
            candidate_logits, candidate_indices = torch.topk(
                masked_logits[row_idx], k=candidate_count, dim=-1
            )
            prefix_points_px = [
                rounded_grid_xy_to_pixel_xy(
                    *location_idx_to_grid_xy(location_idx, int(grid)),
                    image_width=query_mask.shape[1],
                    image_height=query_mask.shape[0],
                    grid=int(grid),
                )
                for location_idx in example.prefix_indices
            ]
            with torch.no_grad():
                batch_coords = build_candidate_coord_batch(
                    candidate_indices=candidate_indices.tolist(),
                    prefix_points_px=prefix_points_px,
                    grid=int(grid),
                    image_width=query_mask.shape[1],
                    image_height=query_mask.shape[0],
                    device=predictor.device,
                )
                batch_labels = torch.ones(
                    batch_coords.shape[:2], dtype=torch.int, device=predictor.device
                )
                transformed_coords = predictor.transform.apply_coords_torch(
                    batch_coords, predictor.original_size
                )
                with torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled
                ):
                    masks, scores, _ = predictor.predict_torch(
                        point_coords=transformed_coords,
                        point_labels=batch_labels,
                        boxes=None,
                        mask_input=None,
                        multimask_output=multimask_for_prefix(
                            aux_cfg.get("decoder_policy", "single"),
                            len(prefix_points_px) + 1,
                        ),
                        return_logits=False,
                    )
                if multimask_for_prefix(
                    aux_cfg.get("decoder_policy", "single"), len(prefix_points_px) + 1
                ):
                    best_mask_choice = scores.argmax(dim=1)
                    gather = torch.arange(masks.shape[0], device=masks.device)
                    selected_masks = masks[gather, best_mask_choice]
                else:
                    selected_masks = masks[:, 0]
                candidate_ious = batch_iou(selected_masks, target_mask).to(
                    device=masked_logits.device, dtype=masked_logits.dtype
                )
            probs = torch.softmax(candidate_logits / temperature, dim=-1)
            losses.append(1.0 - torch.sum(probs * candidate_ious))
    if not losses:
        return masked_logits.new_zeros(())
    return torch.stack(losses).mean()
