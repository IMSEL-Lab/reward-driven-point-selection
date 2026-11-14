"""Typed result structures shared across training, evaluation, and rollouts.

These ``TypedDict``s document the shape of the heterogeneous result/summary
dicts that the pipeline passes around. They are erased at runtime (plain dicts),
so they add static-typing precision without changing behavior.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
import torch

from sam_rl.episode_manifest import EpisodeSpec


class OracleExample(TypedDict):
    """One supervised example from the oracle-heatmap dataset."""

    obs: dict[str, torch.Tensor]
    target_idx: torch.Tensor
    example_index: torch.Tensor


class RolloutResult(TypedDict):
    """Per-episode result from a behaviour-cloning rollout."""

    query_rgb: np.ndarray
    sim_map: np.ndarray
    pred_mask: np.ndarray
    target_mask: np.ndarray
    points_xy: np.ndarray
    iou: float
    dice: float
    boundary_iou: float
    wallclock_ms: float


class StandardRolloutResult(TypedDict):
    """Per-episode result from a single-environment RL rollout."""

    query_rgb: np.ndarray
    sim_map: np.ndarray
    pred_mask: np.ndarray
    target_mask: np.ndarray
    points_xy: np.ndarray
    per_step_ious: list[float]
    iou: float
    dice: float
    boundary_iou: float
    wallclock_ms: float


class RolloutBatch(TypedDict):
    """A collected PPO rollout batch (observations plus stacked tensors)."""

    obs: dict[str, torch.Tensor]
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class EvalRow(TypedDict):
    """One per-episode row written to an evaluation CSV."""

    method: str
    N: int
    image_id: str
    seed: int
    training_seed: int
    eval_seed: int
    episode_key: str
    iou: float
    dice: float
    boundary_iou: float
    wallclock_ms: float


class PerStepRow(TypedDict):
    """Mean IoU at a given click index, aggregated over episodes."""

    N: int
    step: int
    mean_iou: float


class ClusterSpread(TypedDict):
    """Pairwise-distance statistics of selected points across episodes."""

    budget: int
    split: str
    num_episodes: int
    mean_pairwise_distance_px: float
    median_pairwise_distance_px: float
    std_pairwise_distance_px: float


class RLRollout(TypedDict):
    """Per-episode rollout retained in an evaluation summary."""

    episode: EpisodeSpec
    points_xy: list
    pred_mask: np.ndarray
    final_iou: float


class RLEvalSummary(TypedDict):
    """Aggregate metrics for an RL evaluation over a split."""

    budget: int
    episodes: float
    mean_iou: float
    mean_dice: float
    mean_boundary_iou: float
    mean_wallclock_ms: float
    per_step_rows: list[PerStepRow]
    cluster_spread: ClusterSpread
    rollouts: dict[str, RLRollout]


class SmokeRewardCheck(TypedDict):
    """Diagnostics from the reward/coordinate consistency smoke check."""

    max_reward_delta_error: float
    max_coord_error_px: float
    positive_reward_events: int
