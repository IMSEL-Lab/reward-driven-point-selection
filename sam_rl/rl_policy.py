from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from sam_rl.greedy_bc import build_policy as build_bc_policy


class PPOPointPolicy(nn.Module):
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
        value_hidden_dim: int = 256,
        use_current_mask: bool = True,
        reserve_mask_channel: bool = False,
    ) -> None:
        super().__init__()
        self.grid = int(grid)
        self.use_query_rgb = bool(use_query_rgb)
        self.use_query_dino_features = bool(use_query_dino_features)
        self.dino_feature_channels = int(dino_feature_channels)
        self.dino_proj_channels = int(dino_proj_channels)
        self.use_current_mask = bool(use_current_mask)
        self.reserve_mask_channel = bool(reserve_mask_channel)
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
        # New ablations retain one architecture and represent mask-off with zero input.
        # Legacy checkpoints can still request the historical 39-channel layout.
        self.has_mask_channel = self.use_current_mask or self.reserve_mask_channel
        in_channels = self.bc_input_channels + (1 if self.has_mask_channel else 0)
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
        self.value_head = nn.Sequential(
            nn.Linear(hidden_channels + 1, value_hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(value_hidden_dim, 1),
        )
        coords = torch.linspace(-1.0, 1.0, self.grid)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        self.register_buffer("_coord_x", xx.unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("_coord_y", yy.unsqueeze(0).unsqueeze(0), persistent=False)

    def _zero_current_mask_input(self) -> None:
        if not self.has_mask_channel:
            return
        with torch.no_grad():
            self.encoder[0].weight[:, self.bc_input_channels :, :, :] = 0.0
            self.skip.weight[:, self.bc_input_channels :, :] = 0.0

    def encode_features(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        stacked = self._stack_inputs(obs)
        return self.encoder(stacked) + self.skip(stacked)

    def actor_logits(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        logits = self.head(self.encode_features(obs)).flatten(1)
        action_mask = obs.get("action_mask")
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), torch.finfo(logits.dtype).min)
        return logits

    def actor_only(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return action logits without evaluating the privileged critic."""
        return self.actor_logits(obs)

    def actor_action(
        self, obs: dict[str, torch.Tensor], deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.actor_only(obs)
        dist = Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action)

    def value(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        features = self.encode_features(obs)
        pooled = features.mean(dim=(-2, -1))
        current_iou = ensure_scalar(
            obs.get("current_iou", torch.zeros(features.shape[0], 1, device=features.device))
        ).float()
        return self.value_head(torch.cat([pooled, current_iou], dim=1)).squeeze(-1)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.actor_logits(obs)

    def act(
        self,
        obs: dict[str, torch.Tensor],
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor_logits(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        elif generator is None:
            action = dist.sample()
        else:
            action = torch.multinomial(
                torch.softmax(logits, dim=-1), num_samples=1, generator=generator
            ).squeeze(-1)
        return action, dist.log_prob(action), self.value(obs)

    def evaluate_actions(
        self,
        obs: dict[str, torch.Tensor],
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.actor_logits(obs)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        value = self.value(obs)
        return log_prob, entropy, value, logits

    def _stack_inputs(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
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
        if self.use_query_dino_features:
            if self.dino_proj is None:
                raise RuntimeError("DINO projection is not initialized for this policy.")
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
            parts.insert(0, ensure_image(obs["query_rgb"]).float())
        stacked = torch.cat(parts, dim=1)
        if self.has_mask_channel:
            if self.use_current_mask:
                current_mask = ensure_single_channel(obs["current_mask"]).float()
            else:
                current_mask = torch.zeros_like(sim_map)
            stacked = torch.cat([stacked, current_mask], dim=1)
        return stacked


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


def build_rl_policy(cfg: dict) -> PPOPointPolicy:
    return PPOPointPolicy(
        grid=int(cfg.get("grid", 37)),
        hidden_channels=int(cfg.get("hidden_channels", 64)),
        depth=int(cfg.get("depth", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_query_rgb=bool(cfg.get("use_query_rgb", False)),
        use_query_dino_features=bool(cfg.get("use_query_dino_features", False)),
        dino_feature_channels=int(cfg.get("dino_feature_channels", 1024)),
        dino_proj_channels=int(cfg.get("dino_proj_channels", 32)),
        value_hidden_dim=int(cfg.get("value_hidden_dim", 256)),
        use_current_mask=bool(cfg.get("use_current_mask", True)),
        reserve_mask_channel=bool(cfg.get("reserve_mask_channel", False)),
    )


def bc_observation_from_rl_observation(
    obs: dict[str, torch.Tensor], reference_model: nn.Module | None = None
) -> dict[str, torch.Tensor]:
    allowed = {
        "sim_map",
        "click_map",
        "step_index",
        "budget",
        "query_rgb",
        "query_dino",
        "current_mask",
    }
    result = {key: value for key, value in obs.items() if key in allowed}
    if "current_mask" in result and reference_model is not None:
        model_uses_mask = bool(getattr(reference_model, "use_current_mask", False))
        if not model_uses_mask:
            result["current_mask"] = torch.zeros_like(result["current_mask"])
    return result


@dataclass(frozen=True)
class WarmStartResult:
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]


def load_bc_warm_start(
    policy: PPOPointPolicy, checkpoint_path: str | Path, device: str
) -> WarmStartResult:
    payload = torch.load(checkpoint_path, map_location=device)
    reference_model = build_bc_policy(payload["model_config"]).to(device)
    reference_model.load_state_dict(payload["model_state_dict"])
    reference_state = reference_model.state_dict()
    policy_state = policy.state_dict()
    loaded_names: set[str] = set()
    unexpected_keys: list[str] = []

    for name, value in reference_state.items():
        if name not in policy_state:
            unexpected_keys.append(name)
            continue
        target = policy_state[name]
        if target.shape == value.shape:
            target.copy_(value)
            loaded_names.add(name)
            continue
        if (
            name in {"encoder.0.weight", "skip.weight"}
            and target.shape[0] == value.shape[0]
            and target.shape[2:] == value.shape[2:]
        ):
            target.zero_()
            target[:, : value.shape[1], ...] = value
            loaded_names.add(name)
            continue
        unexpected_keys.append(name)

    policy.load_state_dict(policy_state, strict=False)
    missing_keys = sorted(name for name in policy_state.keys() if name not in loaded_names)
    return WarmStartResult(
        missing_keys=tuple(missing_keys),
        unexpected_keys=tuple(sorted(unexpected_keys)),
    )
