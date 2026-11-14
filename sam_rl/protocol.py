"""Shared SAM decoding and precision rules for training and evaluation."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import numpy as np
import torch

DECODER_POLICIES = {"single", "legacy_multimask_first"}
PRECISIONS = {"fp32", "fp16", "bf16"}


def normalize_decoder_policy(policy: str | None) -> str:
    """Validate a decoder policy; omitted policy preserves historical prefix behavior."""
    value = "legacy_multimask_first" if policy is None else str(policy).lower()
    if value not in DECODER_POLICIES:
        raise ValueError(
            f"decoder_policy must be one of {sorted(DECODER_POLICIES)}, got {value!r}."
        )
    return value


def multimask_for_prefix(policy: str | None, num_points: int) -> bool:
    """Return whether SAM should produce multimask candidates for this decoded prefix."""
    if int(num_points) < 0:
        raise ValueError("num_points must be non-negative.")
    return normalize_decoder_policy(policy) == "legacy_multimask_first" and int(num_points) == 1


def select_sam_mask(masks: Any, scores: Any, multimask_output: bool) -> np.ndarray:
    """Select SAM's score-best proposal when multimask output is enabled."""
    mask_array = _to_numpy(masks)
    if mask_array.ndim == 2:
        return np.asarray(mask_array, dtype=bool)
    if mask_array.ndim < 3 or mask_array.shape[0] < 1:
        raise ValueError(f"Expected SAM masks shaped [K,H,W], got {mask_array.shape}.")
    if not multimask_output:
        return np.asarray(mask_array[0], dtype=bool)
    score_array = _to_numpy(scores).reshape(-1)
    if score_array.size != mask_array.shape[0]:
        raise ValueError("SAM score count must match the number of returned masks.")
    if not np.isfinite(score_array).all():
        raise ValueError("SAM mask scores must be finite.")
    return np.asarray(mask_array[int(np.argmax(score_array))], dtype=bool)


def resolve_precision(device: str | torch.device, precision: str | None = "fp32") -> str:
    """Validate requested inference precision without silently changing it."""
    value = "fp32" if precision is None else str(precision).lower()
    if value not in PRECISIONS:
        raise ValueError(f"precision must be one of {sorted(PRECISIONS)}, got {value!r}.")
    device_obj = torch.device(device)
    if value == "fp32":
        return value
    if device_obj.type != "cuda":
        raise RuntimeError(
            f"{value} autocast is only supported by this protocol on CUDA; got {device_obj}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(f"Requested {value} precision but CUDA is unavailable.")
    if value == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "Requested bf16 precision but this CUDA device does not support bfloat16."
        )
    return value


def autocast_context(device: str | torch.device, precision: str | None = "fp32"):
    """Create an explicit autocast context for the validated precision."""
    effective = resolve_precision(device, precision)
    if effective == "fp32":
        return nullcontext()
    dtype = torch.float16 if effective == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)
