"""Thin, lazy adapters for published one-shot segmentation systems.

External repositories are imported from user-supplied checkouts at runtime; no
third-party implementation is copied into this project.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import os
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from PIL import Image

from sam_rl.episode_manifest import EpisodeSpec


@dataclass(frozen=True)
class EpisodeInput:
    """Inputs visible to a predictor; deliberately excludes the query label."""

    dataset: str
    split: str
    class_name: str
    support_id: str
    query_id: str
    support_image_path: Path
    support_mask_path: Path
    query_image_path: Path

    @classmethod
    def from_episode(cls, episode: EpisodeSpec) -> EpisodeInput:
        return cls(
            dataset=episode.dataset,
            split=episode.split,
            class_name=episode.class_name,
            support_id=episode.support_id,
            query_id=episode.query_id,
            support_image_path=episode.support_image_path,
            support_mask_path=episode.support_mask_path,
            query_image_path=episode.query_image_path,
        )


class Predictor(Protocol):
    def predict(self, episode: EpisodeInput) -> np.ndarray: ...


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def load_binary_mask(path: Path) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L")) >= 128).astype(np.uint8)


def resize_binary_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array.squeeze()
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D mask, got shape {array.shape}.")
    if array.shape != (height, width):
        image = Image.fromarray((array > 0).astype(np.uint8) * 255)
        image = image.resize((width, height), Image.Resampling.NEAREST)
        array = np.asarray(image)
    return (array > 0).astype(np.uint8)


def _require_repo(root: str | Path, name: str) -> Path:
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{name} source checkout does not exist: {path}")
    return path


def _prepend_import_root(root: Path) -> None:
    value = str(root)
    if value not in sys.path:
        sys.path.insert(0, value)


class SegGPTAdapter:
    """Hugging Face SegGPT one-shot mask-prompt inference."""

    def __init__(
        self,
        *,
        model_id: str = "BAAI/seggpt-vit-large",
        revision: str | None = None,
        device: str = "cuda",
    ):
        try:
            import torch
            from transformers import SegGptForImageSegmentation, SegGptImageProcessor
        except ImportError as exc:
            raise RuntimeError(
                "SegGPT evaluation requires torch and transformers with SegGPT support."
            ) from exc
        self.torch = torch
        self.model_id = model_id
        self.revision = revision
        self.device = device
        kwargs = (
            {"revision": revision, "local_files_only": True}
            if revision
            else {"local_files_only": True}
        )
        processor_loader = cast(Any, SegGptImageProcessor).from_pretrained
        model_loader = cast(Any, SegGptForImageSegmentation).from_pretrained
        self.processor = processor_loader(model_id, **kwargs)
        self.model = (
            model_loader(
                model_id,
                torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
                **kwargs,
            )
            .to(device)
            .eval()
        )

    def predict(self, episode: EpisodeInput) -> np.ndarray:
        torch = self.torch
        query = Image.open(episode.query_image_path).convert("RGB")
        support = Image.open(episode.support_image_path).convert("RGB")
        support_mask = Image.fromarray(load_binary_mask(episode.support_mask_path), mode="L")
        query_size = (query.height, query.width)
        inputs = self.processor(
            images=[query],
            prompt_images=[support],
            prompt_masks=[support_mask],
            num_labels=2,
            return_tensors="pt",
        ).to(self.device)
        inputs = {
            key: value.to(self.model.dtype) if value.dtype.is_floating_point else value
            for key, value in inputs.items()
        }
        with torch.inference_mode():
            outputs = self.model(**inputs)
        labels = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[query_size], num_labels=2
        )[0]
        return (labels.detach().cpu().numpy() > 0).astype(np.uint8)


class PerSAMAdapter:
    """Official PerSAM predictor wrapper using the paired SAM-H checkpoint."""

    def __init__(
        self,
        *,
        repo: str | Path,
        checkpoint: str | Path,
        device: str = "cuda",
        sam_model_type: str = "vit_h",
    ):
        self.repo = _require_repo(repo, "PerSAM")
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"PerSAM SAM checkpoint not found: {self.checkpoint}")
        _prepend_import_root(self.repo)
        try:
            import torch
            import torch.nn.functional as functional

            persam_module = importlib.import_module("per_segment_anything")
        except ImportError as exc:
            raise RuntimeError(
                "PerSAM evaluation requires its source checkout and PyTorch dependencies."
            ) from exc
        self.torch, self.functional = torch, functional
        self.device = device
        self.sam_model_type = sam_model_type
        self.predictor = persam_module.SamPredictor(
            persam_module.sam_model_registry[sam_model_type](checkpoint=str(self.checkpoint))
            .to(device)
            .eval()
        )

    @staticmethod
    def _point_selection(similarity: Any) -> tuple[np.ndarray, np.ndarray]:
        flat = similarity.flatten()
        _, width = similarity.shape
        positive = int(flat.argmax().item())
        negative = int(flat.argmin().item())
        points = np.asarray(
            [[positive % width, positive // width], [negative % width, negative // width]]
        )
        return points, np.asarray([1, 0], dtype=np.int32)

    def predict(self, episode: EpisodeInput) -> np.ndarray:
        functional, predictor = self.functional, self.predictor
        support = load_rgb(episode.support_image_path)
        support_mask = np.repeat(
            (load_binary_mask(episode.support_mask_path)[:, :, None] * 255), 3, axis=2
        )
        query = load_rgb(episode.query_image_path)
        # This follows the public PerSAM inference sequence; no query annotation is read.
        ref_mask = predictor.set_image(support, support_mask)
        ref_features = predictor.features.squeeze().permute(1, 2, 0)
        resized_ref = functional.interpolate(ref_mask, size=ref_features.shape[:2], mode="bilinear")
        target_pixels = ref_features[resized_ref.squeeze()[0] > 0]
        if target_pixels.numel() == 0:
            raise RuntimeError("PerSAM support mask contains no foreground feature pixels.")
        embedding = target_pixels.mean(0, keepdim=True)
        normalized = embedding / embedding.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        target_embedding = embedding.unsqueeze(0)
        predictor.set_image(query)
        features = predictor.features.squeeze()
        channels, height, width = features.shape
        features = features / features.norm(dim=0, keepdim=True).clamp_min(1e-12)
        similarity = normalized @ features.reshape(channels, height * width)
        similarity = functional.interpolate(
            similarity.reshape(1, 1, height, width), scale_factor=4, mode="bilinear"
        )
        similarity = predictor.model.postprocess_masks(
            similarity, input_size=predictor.input_size, original_size=predictor.original_size
        ).squeeze()
        points, labels = self._point_selection(similarity)
        attention = (similarity - similarity.mean()) / similarity.std().clamp_min(1e-12)
        attention = (
            functional.interpolate(attention[None, None], size=(64, 64), mode="bilinear")
            .sigmoid()[None]
            .flatten(3)
        )
        _, _, logits, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=False,
            attn_sim=attention,
            target_embedding=target_embedding,
        )
        masks, scores, logits, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            mask_input=logits[:1],
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        # Match the official second cascade refinement. The upstream script also
        # assumes this selected mask is non-empty when constructing its box.
        ys, xs = np.nonzero(masks[best])
        if len(xs) == 0:
            raise RuntimeError(
                "PerSAM's first refinement returned an empty mask; its official box "
                "refinement cannot proceed for this episode."
            )
        box = np.asarray([[xs.min(), ys.min(), xs.max(), ys.max()]])
        masks, scores, _, _ = predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=box,
            mask_input=logits[best : best + 1],
            multimask_output=True,
        )
        best = int(np.argmax(scores))
        return np.asarray(masks[best], dtype=np.uint8)


class MatcherAdapter:
    """Matcher adapter around the official ``build_matcher_oss`` and prediction API."""

    def __init__(
        self,
        *,
        repo: str | Path,
        sam_checkpoint: str | Path,
        dinov2_checkpoint: str | Path,
        device: str = "cuda",
        seed: int = 0,
        matcher_args: dict[str, Any] | None = None,
    ):
        self.repo = _require_repo(repo, "Matcher")
        self.sam_checkpoint = Path(sam_checkpoint).expanduser().resolve()
        self.dinov2_checkpoint = Path(dinov2_checkpoint).expanduser().resolve()
        for checkpoint in (self.sam_checkpoint, self.dinov2_checkpoint):
            if not checkpoint.is_file():
                raise FileNotFoundError(f"Matcher checkpoint not found: {checkpoint}")
        _prepend_import_root(self.repo)
        try:
            import torch

            matcher_module = importlib.import_module("matcher.Matcher")
            from torchvision import transforms
        except ImportError as exc:
            raise RuntimeError(
                "Matcher evaluation requires its source checkout and its dedicated environment."
            ) from exc
        self.torch, self.transforms = torch, transforms
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        random.seed(seed)
        np.random.seed(seed)
        args = _matcher_default_args(
            torch.device(device),
            self.sam_checkpoint,
            self.dinov2_checkpoint,
            **(matcher_args or {}),
        )
        self.matcher = matcher_module.build_matcher_oss(args)
        self.image_transform = transforms.Compose(
            [transforms.Resize((518, 518)), transforms.ToTensor()]
        )
        self.device = device

    def predict(self, episode: EpisodeInput) -> np.ndarray:
        torch = self.torch
        support = (
            self.image_transform(Image.open(episode.support_image_path).convert("RGB"))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )
        query = (
            self.image_transform(Image.open(episode.query_image_path).convert("RGB"))
            .unsqueeze(0)
            .to(self.device)
        )
        raw_mask = load_binary_mask(episode.support_mask_path)
        mask_image = Image.fromarray(raw_mask * 255).resize((518, 518), Image.Resampling.NEAREST)
        support_mask = torch.from_numpy((np.asarray(mask_image) > 0).astype(np.float32))[
            None, None
        ].to(self.device)
        self.matcher.set_reference(support, support_mask)
        self.matcher.set_target(query)
        with torch.inference_mode():
            prediction = self.matcher.predict()
        self.matcher.clear()
        return prediction.detach().float().cpu().squeeze().numpy() > 0.5


def _matcher_default_args(
    device: Any,
    sam_checkpoint: Path,
    dinov2_checkpoint: Path,
    **overrides: Any,
) -> Any:
    from argparse import Namespace

    values = dict(
        device=device,
        dinov2_size="vit_large",
        sam_size="vit_h",
        dinov2_weights=str(dinov2_checkpoint),
        sam_weights=str(sam_checkpoint),
        points_per_side=64,
        pred_iou_thresh=0.88,
        stability_score_thresh=0.95,
        sel_stability_score_thresh=0.0,
        iou_filter=0.0,
        box_nms_thresh=1.0,
        output_layer=3,
        dense_multimask_output=0,
        use_dense_mask=0,
        multimask_output=0,
        num_centers=8,
        use_box=False,
        # Match the official FSS evaluation script; this flag's parser default is False.
        use_points_or_centers=False,
        sample_range=(4, 6),
        max_sample_iterations=30,
        alpha=0.8,
        beta=0.2,
        exp=1.0,
        emd_filter=0.0,
        purity_filter=0.0,
        coverage_filter=0.0,
        use_score_filter=False,
        deep_score_norm_filter=0.1,
        deep_score_filter=0.33,
        topk_scores_threshold=0.7,
        num_merging_mask=10,
    )
    values.update(overrides)
    sample_range = values["sample_range"]
    if isinstance(sample_range, str):
        sample_range = ast.literal_eval(sample_range)
    if not isinstance(sample_range, (tuple, list)) or len(sample_range) != 2:
        raise ValueError("Matcher sample_range must contain exactly two integers.")
    values["sample_range"] = (int(sample_range[0]), int(sample_range[1]))
    return Namespace(**values)


class SANSAAdapter:
    """SANSA generalist SAM2-Hiera-L adapter with a single mask support prompt."""

    def __init__(
        self,
        *,
        repo: str | Path,
        checkpoint: str | Path,
        sam2_checkpoint: str | Path,
        device: str = "cuda",
        threshold: float = 0.5,
        sam2_version: str = "large",
        adaptformer_stages: list[int] | None = None,
        channel_factor: float = 0.3,
        seed: int = 0,
    ):
        self.repo = _require_repo(repo, "SANSA")
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.sam2_checkpoint = Path(sam2_checkpoint).expanduser().resolve()
        for path in (self.checkpoint, self.sam2_checkpoint):
            if not path.is_file():
                raise FileNotFoundError(f"SANSA checkpoint not found: {path}")
        _prepend_import_root(self.repo)
        try:
            import torch
            import torch.nn.functional as functional

            sansa_module = importlib.import_module("models.sansa.sansa")
            commons_module = importlib.import_module("util.commons")
            promptable_module = importlib.import_module("util.promptable_utils")
        except ImportError as exc:
            raise RuntimeError(
                "SANSA evaluation requires its source checkout and dedicated dependencies."
            ) from exc
        self.torch, self.functional = torch, functional
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.build_prompt_dict = promptable_module.build_prompt_dict
        self.device, self.threshold = device, threshold
        self.sam2_version = sam2_version
        self.adaptformer_stages = adaptformer_stages or [2, 3]
        self.channel_factor = channel_factor
        sansa_paths = importlib.import_module("util.path_utils")

        old_weight, config_path = sansa_paths.SAM2_PATHS_CONFIG[sam2_version]
        sansa_paths.SAM2_PATHS_CONFIG[sam2_version] = (str(self.sam2_checkpoint), config_path)
        try:
            with _working_directory(self.repo):
                self.model = sansa_module.build_sansa(
                    sam2_version, self.adaptformer_stages, channel_factor, device
                )
        finally:
            sansa_paths.SAM2_PATHS_CONFIG[sam2_version] = (old_weight, config_path)
        commons_module.resume_from_checkpoint(str(self.checkpoint), self.model)
        self.model.to(device).eval()

    def predict(self, episode: EpisodeInput) -> np.ndarray:
        torch, functional = self.torch, self.functional
        image_transform = _sansa_image_transform()
        support = image_transform(Image.open(episode.support_image_path).convert("RGB"))
        query = image_transform(Image.open(episode.query_image_path).convert("RGB"))
        mask = Image.fromarray(load_binary_mask(episode.support_mask_path) * 255).resize(
            (518, 518), Image.Resampling.NEAREST
        )
        support_mask = torch.from_numpy((np.asarray(mask) > 0).astype(np.float32))[None, None].to(
            self.device
        )
        imgs = _stack_sansa_frames(torch, support, query).to(self.device)
        prompt = self.build_prompt_dict(
            support_mask, "mask", n_shots=1, train_mode=False, device=torch.device(self.device)
        )
        with torch.inference_mode():
            outputs = self.model(imgs, prompt)
        logits = outputs["pred_masks"]
        logits = functional.interpolate(
            logits.unsqueeze(0), size=(518, 518), mode="bilinear", align_corners=False
        )
        return (logits[0, -1].sigmoid().detach().cpu().numpy() > self.threshold).astype(np.uint8)


def _sansa_image_transform() -> Any:
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError(
            "SANSA evaluation requires torchvision in its isolated environment."
        ) from exc
    return transforms.Compose(
        [
            transforms.Resize((518, 518)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )


def _stack_sansa_frames(torch: Any, support: Any, query: Any) -> Any:
    """Make the [B,T,C,H,W] frame tensor SANSA's temporal model expects."""
    if support.ndim != 3 or query.ndim != 3 or support.shape != query.shape:
        raise ValueError(
            "SANSA expects matching unbatched CHW support/query tensors, "
            f"got {tuple(support.shape)} and {tuple(query.shape)}."
        )
    return torch.stack((support, query), dim=0).unsqueeze(0)


@contextmanager
def _working_directory(path: Path):
    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)


def build_adapter(method: str, config: dict[str, Any]) -> Predictor:
    adapters = {
        "seggpt": SegGPTAdapter,
        "persam": PerSAMAdapter,
        "matcher": MatcherAdapter,
        "sansa": SANSAAdapter,
    }
    try:
        adapter_type = adapters[method.lower()]
    except KeyError as exc:
        raise NotImplementedError(
            f"No faithful external adapter is implemented for {method!r}."
        ) from exc
    return adapter_type(**config)


def external_revision(repo: str | Path) -> dict[str, object]:
    """Return the pinned git revision and dirty-state marker for an external checkout."""
    import subprocess

    root = _require_repo(repo, "External")
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "-C", str(root), "diff", "HEAD", "--binary"],
        check=True,
        capture_output=True,
    ).stdout
    dirty_digest = hashlib.sha256(diff)
    for line in status.splitlines():
        dirty_file = root / line[3:]
        if dirty_file.is_file():
            dirty_digest.update(line.encode())
            dirty_digest.update(dirty_file.read_bytes())
    if not commit:
        raise RuntimeError(f"Could not identify the external source commit in {root}.")
    return {
        "root_name": root.name,
        "commit": commit,
        "dirty": bool(status),
        "working_tree_sha256": dirty_digest.hexdigest(),
        "status": status.splitlines(),
    }
