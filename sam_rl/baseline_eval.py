from __future__ import annotations

import csv
import json
import random
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from sam_rl.episode_manifest import EpisodeSpec, safe_slug
from sam_rl.episode_source import build_fss_episodes as _build_fss_episodes
from sam_rl.protocol import (
    autocast_context,
    multimask_for_prefix,
    normalize_decoder_policy,
    resolve_precision,
    select_sam_mask,
)
from sam_rl.provenance import atomic_write_npz, cache_metadata, cache_path, file_fingerprint

GARNET = np.array([0x73, 0x00, 0x0A], dtype=np.uint8)
BLACK = np.array([0x00, 0x00, 0x00], dtype=np.uint8)
WHITE = np.array([0xFF, 0xFF, 0xFF], dtype=np.uint8)
NEUTRAL = np.array([0xA2, 0xA2, 0xA2], dtype=np.uint8)
ATLANTIC = np.array([0x1F, 0x41, 0x4D], dtype=np.uint8)
ROSE = np.array([0xCC, 0x2E, 0x40], dtype=np.uint8)
LIME = np.array([0xCE, 0xD3, 0x18], dtype=np.uint8)


FewShotEpisode = EpisodeSpec
DINO_REVISION = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"


class CsvResultWriter:
    fieldnames: ClassVar[list[str]] = [
        "method",
        "N",
        "image_id",
        "episode_key",
        "seed",
        "training_seed",
        "eval_seed",
        "iou",
        "dice",
        "boundary_iou",
        "wallclock_ms",
    ]

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8", newline="")
        self.writer = csv.DictWriter(self.handle, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.handle.flush()

    def write(self, row: dict[str, object]) -> None:
        self.writer.writerow(row)
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class DinoSimilarityCache:
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
        self.precision = resolve_precision(self.device, precision)
        if self.image_size % 14 != 0:
            raise ValueError(f"DINO image_size must be divisible by 14, got {self.image_size}.")

    def path_for(self, episode: FewShotEpisode) -> Path:
        metadata = self._metadata(episode)
        return cache_path(self.cache_dir, "dino_similarity", episode, metadata)

    def _metadata(self, episode: FewShotEpisode) -> dict:
        return cache_metadata(
            kind="dino_similarity",
            episode=episode,
            config={
                "encoder_checkpoint": file_fingerprint(self.dino_checkpoint),
                "encoder": "dinov2_vitl14",
                "image_size": self.image_size,
                "preprocessing": "rgb_imagenet_normalize_resize_bicubic",
                "precision": self.precision,
                "encoder_revision": DINO_REVISION,
            },
            schema_version=2,
        )

    def ensure(self, episode: FewShotEpisode) -> Path:
        path = self.path_for(episode)
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as payload:
                    metadata = json.loads(str(payload["metadata_json"].item()))
                    similarity = np.asarray(payload["sim_map"], dtype=np.float32)
                expected = self._metadata(episode)
                if metadata == expected and similarity.ndim == 2 and np.isfinite(similarity).all():
                    return path
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                pass
            path.unlink(missing_ok=True)
        support_rgb, support_mask, query_rgb, _ = load_episode_arrays(
            episode, image_size=self.image_size
        )
        sim_map = compute_similarity_map(
            dino_model=self.dino_model,
            device=self.device,
            support_rgb=support_rgb,
            support_mask=support_mask,
            query_rgb=query_rgb,
            image_size=self.image_size,
            precision=self.precision,
        )
        atomic_write_npz(
            path,
            sim_map=sim_map.astype(np.float32),
            metadata_json=np.asarray(json.dumps(self._metadata(episode), sort_keys=True)),
        )
        return path

    def load(self, episode: FewShotEpisode) -> np.ndarray:
        payload = np.load(self.ensure(episode), allow_pickle=False)
        return np.asarray(payload["sim_map"], dtype=np.float32)


def build_fss_episodes(
    dataset_root: str | Path,
    split_file: str | Path,
    split_name: str,
    pair_schedule: list[list[int | str]] | list[tuple[int | str, int | str]],
    max_episodes: int | None = None,
) -> list[FewShotEpisode]:
    return _build_fss_episodes(
        dataset_root=dataset_root,
        split_file=split_file,
        split_name=split_name,
        pair_schedule=pair_schedule,
        max_episodes=max_episodes,
    )


def load_episode_arrays(
    episode: FewShotEpisode,
    image_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    support_rgb = _load_rgb(episode.support_image_path, image_size=image_size)
    support_mask = _load_mask(episode.support_mask_path, image_size=image_size)
    query_rgb = _load_rgb(episode.query_image_path, image_size=image_size)
    query_mask = _load_mask(episode.query_mask_path, image_size=image_size)
    return support_rgb, support_mask, query_rgb, query_mask


def load_dino_model(checkpoint: str | Path, device: str) -> torch.nn.Module:
    with _preserve_frozen_loader_rng():
        model = torch.hub.load(
            f"facebookresearch/dinov2:{DINO_REVISION}",
            "dinov2_vitl14",
            pretrained=False,
            trust_repo=True,
            skip_validation=True,
        )
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.to(device=device)
        model.eval()
    return model


def load_sam_predictor(checkpoint: str | Path, model_type: str, device: str):
    with _preserve_frozen_loader_rng():
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ModuleNotFoundError as exc:
            if exc.name in {"torchvision", "segment_anything"}:
                raise ModuleNotFoundError(
                    "SAM evaluation requires the pinned segment-anything dependency and torchvision. "
                    "Install the research extra with `uv sync --extra research`."
                ) from exc
            raise

        sam = sam_model_registry[str(model_type)](checkpoint=str(checkpoint))
        sam.to(device=device)
        predictor = SamPredictor(sam)
    return predictor


@contextmanager
def _preserve_frozen_loader_rng():
    """Keep frozen model construction/loading out of learner RNG streams.

    DINO and SAM factories initialize on CPU and load CPU checkpoints before
    moving weights to the requested device. Fork only the CPU Torch stream so
    this does not initialize or touch other visible CUDA devices. Preserve the
    Python and NumPy streams because lazy factories may also use global RNGs.
    """
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


@torch.inference_mode()
def compute_similarity_map(
    dino_model: torch.nn.Module,
    device: str,
    support_rgb: np.ndarray,
    support_mask: np.ndarray,
    query_rgb: np.ndarray,
    image_size: int,
    precision: str = "fp32",
) -> np.ndarray:
    support_tensor = _preprocess_rgb(support_rgb)
    query_tensor = _preprocess_rgb(query_rgb)
    batch = torch.stack([support_tensor, query_tensor], dim=0).to(device)
    effective_precision = resolve_precision(device, precision)
    with autocast_context(device, effective_precision):
        feature_dict = cast(Any, dino_model).forward_features(batch)
        features = feature_dict["x_norm_patchtokens"].float()
    features = F.normalize(features, dim=-1).cpu()
    patch_grid = image_size // 14
    weights = _mask_patch_weights(support_mask, patch_grid)
    if float(weights.sum()) <= 0.0:
        weights = torch.ones_like(weights)
    prototype = (features[0] * weights[:, None]).sum(dim=0, keepdim=True) / weights.sum()
    prototype = F.normalize(prototype, dim=-1)[0]
    similarity = torch.matmul(features[1], prototype).reshape(patch_grid, patch_grid)
    similarity = similarity.clamp(-1.0, 1.0)
    similarity = (similarity + 1.0) / 2.0
    sim_map = (
        F.interpolate(
            similarity.unsqueeze(0).unsqueeze(0),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .squeeze(0)
        .numpy()
    )
    return np.asarray(sim_map, dtype=np.float32)


def predict_mask_from_points(
    predictor,
    points: np.ndarray,
    *,
    decoder_policy: str = "single",
    precision: str = "fp32",
) -> np.ndarray:
    coords = np.asarray(points, dtype=np.float32)
    labels = np.ones(coords.shape[0], dtype=np.int32)
    policy = normalize_decoder_policy(decoder_policy)
    multimask = multimask_for_prefix(policy, int(coords.shape[0]))
    device = getattr(predictor, "device", "cpu")
    effective_precision = resolve_precision(device, precision)
    with torch.inference_mode(), autocast_context(device, effective_precision):
        masks, scores, _ = predictor.predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=multimask,
        )
    return select_sam_mask(masks, scores, multimask)


def snap_points_to_grid(
    points: np.ndarray, grid: int, image_width: int, image_height: int
) -> np.ndarray:
    coords = np.asarray(points, dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected [N,2] points, got shape {coords.shape}")
    snapped = np.empty_like(coords, dtype=np.int32)
    for index, (x, y) in enumerate(coords.tolist()):
        gx = _pixel_to_grid_index(float(x), image_width, int(grid))
        gy = _pixel_to_grid_index(float(y), image_height, int(grid))
        px = _grid_index_to_pixel_center(gx, image_width, int(grid))
        py = _grid_index_to_pixel_center(gy, image_height, int(grid))
        snapped[index, 0] = round(px)
        snapped[index, 1] = round(py)
    return snapped


def iou_score(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    pred = np.asarray(pred_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    inter = int(np.logical_and(pred, target).sum())
    union = int(np.logical_or(pred, target).sum())
    return float(inter / union) if union else 1.0


def dice_score(pred_mask: np.ndarray, target_mask: np.ndarray) -> float:
    pred = np.asarray(pred_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    inter = int(np.logical_and(pred, target).sum())
    denom = int(pred.sum() + target.sum())
    return float((2 * inter) / denom) if denom else 1.0


def boundary_iou_score(
    pred_mask: np.ndarray, target_mask: np.ndarray, dilation_ratio: float = 0.02
) -> float:
    pred = np.asarray(pred_mask, dtype=bool)
    target = np.asarray(target_mask, dtype=bool)
    radius = max(1, round(float(np.hypot(*pred.shape)) * dilation_ratio))
    pred_band = boundary_band(pred, radius)
    target_band = boundary_band(target, radius)
    inter = int(np.logical_and(pred_band, target_band).sum())
    union = int(np.logical_or(pred_band, target_band).sum())
    return float(inter / union) if union else 1.0


def boundary_band(mask: np.ndarray, radius: int) -> np.ndarray:
    binary = np.asarray(mask, dtype=bool)
    dilated = _binary_dilate(binary, radius)
    eroded = _binary_erode(binary, radius)
    return np.logical_xor(dilated, eroded)


def save_overlay_artifacts(
    output_dir: str | Path,
    episode: FewShotEpisode,
    method: str,
    n_points: int,
    seed: int,
    query_rgb: np.ndarray,
    sim_map: np.ndarray,
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    points: np.ndarray,
    metrics: dict[str, float],
) -> Path:
    output_dir = Path(output_dir)
    sample_dir = (
        output_dir
        / safe_slug(method)
        / f"N{int(n_points)}"
        / f"{safe_slug(episode.image_id)}_seed{int(seed)}"
    )
    sample_dir.mkdir(parents=True, exist_ok=True)

    Image.fromarray(query_rgb).save(sample_dir / "query.png")
    _save_sim_map(sim_map, points, sample_dir / "sim_map.png")
    _save_overlay(query_rgb, pred_mask, target_mask, points, sample_dir / "overlay.png")
    metadata = {
        "dataset": episode.dataset,
        "split": episode.split,
        "class_name": episode.class_name,
        "image_id": episode.image_id,
        "method": method,
        "N": int(n_points),
        "seed": int(seed),
        "points_xy": np.asarray(points, dtype=int).tolist(),
        **{key: float(value) for key, value in metrics.items()},
    }
    (sample_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return sample_dir


def write_json(path: str | Path, payload: Mapping[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _save_sim_map(sim_map: np.ndarray, points: np.ndarray, path: Path) -> None:
    values = np.clip(np.asarray(sim_map, dtype=np.float32), 0.0, 1.0)
    gray = np.round(values * 255.0).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    _draw_square_points(draw, points)
    image.save(path)


def _save_overlay(
    query_rgb: np.ndarray,
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    points: np.ndarray,
    path: Path,
) -> None:
    base = np.asarray(query_rgb, dtype=np.uint8).copy()
    overlay = base.copy()
    overlay[pred_mask] = np.round(0.55 * overlay[pred_mask] + 0.45 * ATLANTIC[None, :]).astype(
        np.uint8
    )
    pred_edge = boundary_band(pred_mask, radius=1)
    target_edge = boundary_band(target_mask, radius=1)
    overlay[pred_edge] = LIME
    overlay[target_edge] = GARNET
    image = Image.fromarray(overlay, mode="RGB")
    draw = ImageDraw.Draw(image)
    _draw_square_points(draw, points)
    image.save(path)


def _draw_square_points(draw: ImageDraw.ImageDraw, points: np.ndarray, size: int = 5) -> None:
    for x, y in np.asarray(points, dtype=int).tolist():
        draw.rectangle(
            (x - size - 1, y - size - 1, x + size + 1, y + size + 1),
            outline=tuple(BLACK.tolist()),
            width=2,
        )
        draw.rectangle(
            (x - size, y - size, x + size, y + size),
            fill=tuple(WHITE.tolist()),
            outline=tuple(GARNET.tolist()),
            width=2,
        )


def _load_rgb(path: Path, image_size: int | None) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if image_size is not None:
        image = image.resize((image_size, image_size), resample=Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def _load_mask(path: Path, image_size: int | None) -> np.ndarray:
    mask = Image.open(path).convert("L")
    if image_size is not None:
        mask = mask.resize((image_size, image_size), resample=Image.Resampling.NEAREST)
    return np.asarray(mask, dtype=np.uint8) >= 128


def _preprocess_rgb(rgb: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(rgb, dtype=np.float32).copy()).permute(2, 0, 1) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
    return (tensor - mean) / std


def _mask_patch_weights(mask: np.ndarray, patch_grid: int) -> torch.Tensor:
    patch_size = mask.shape[0] // patch_grid
    tensor = torch.from_numpy(np.asarray(mask, dtype=np.float32).copy()).view(
        patch_grid, patch_size, patch_grid, patch_size
    )
    return tensor.mean(dim=(1, 3)).reshape(-1)


def _binary_dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(mask, dtype=bool)
    tensor = torch.from_numpy(np.asarray(mask, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    kernel = 2 * radius + 1
    pooled = F.max_pool2d(tensor, kernel_size=kernel, stride=1, padding=radius)
    return pooled.squeeze(0).squeeze(0).numpy() > 0.5


def _binary_erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return np.asarray(mask, dtype=bool)
    inv = 1.0 - np.asarray(mask, dtype=np.float32)
    tensor = torch.from_numpy(inv).unsqueeze(0).unsqueeze(0)
    kernel = 2 * radius + 1
    pooled = F.max_pool2d(tensor, kernel_size=kernel, stride=1, padding=radius)
    return pooled.squeeze(0).squeeze(0).numpy() <= 0.5


def _pixel_to_grid_index(value: float, image_extent: int, grid: int) -> int:
    if image_extent <= 1:
        return 0
    scaled = (float(value) * float(grid) / float(image_extent)) - 0.5
    return int(np.clip(np.round(scaled), 0, grid - 1))


def _grid_index_to_pixel_center(index: int, image_extent: int, grid: int) -> float:
    value = ((float(index) + 0.5) / float(grid)) * float(image_extent)
    return float(np.clip(value, 0.0, image_extent - 1.0))
