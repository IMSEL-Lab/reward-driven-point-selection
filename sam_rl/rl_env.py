from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from sam_rl.baseline_eval import FewShotEpisode, load_episode_arrays
from sam_rl.greedy_bc import DinoQueryFeatureCache, rgb_to_grid_tensor, scalar_map_to_grid_tensor
from sam_rl.sam_cache import (
    SamImageEmbeddingBatch,
    SamImageEmbeddingCache,
    set_predictor_image_embedding_batch,
)


@dataclass
class VectorizedPointSelectionState:
    episodes: list[FewShotEpisode]
    sim_grid: torch.Tensor
    query_masks: torch.Tensor
    query_rgb_grid: torch.Tensor | None
    query_dino: torch.Tensor | None
    embedding_batch: SamImageEmbeddingBatch
    click_map: torch.Tensor
    current_mask_grid: torch.Tensor
    current_pred_masks: torch.Tensor
    current_ious: torch.Tensor
    point_indices: torch.Tensor
    point_coords_px: torch.Tensor
    step_index: int


class VectorizedPointSelectionEnv:
    def __init__(
        self,
        predictor,
        sim_cache,
        sam_embedding_cache: SamImageEmbeddingCache,
        image_size: int,
        grid: int = 37,
        budget: int = 5,
        query_feature_cache: DinoQueryFeatureCache | None = None,
        use_query_rgb: bool = False,
        use_bfloat16: bool = True,
        decoder_policy: str = "legacy_multimask_first",
    ) -> None:
        self.predictor = predictor
        self.sim_cache = sim_cache
        self.sam_embedding_cache = sam_embedding_cache
        self.image_size = int(image_size)
        self.grid = int(grid)
        self.budget = int(budget)
        self.query_feature_cache = query_feature_cache
        self.use_query_rgb = bool(use_query_rgb)
        self.use_bfloat16 = bool(use_bfloat16)
        self.decoder_policy = str(decoder_policy)
        self.device = str(predictor.device)
        self.state: VectorizedPointSelectionState | None = None

    @property
    def action_dim(self) -> int:
        return self.grid * self.grid

    @property
    def num_envs(self) -> int:
        state = self._require_state()
        return len(state.episodes)

    def reset_batch(self, episodes: Sequence[FewShotEpisode]) -> dict[str, torch.Tensor]:
        episodes = list(episodes)
        if not episodes:
            raise ValueError(
                "VectorizedPointSelectionEnv.reset_batch requires at least one episode."
            )
        batch_size = len(episodes)
        sim_parts = []
        mask_parts = []
        rgb_parts = []
        dino_parts = []
        for episode in episodes:
            _, _, query_rgb, query_mask = load_episode_arrays(episode, image_size=self.image_size)
            sim_parts.append(scalar_map_to_grid_tensor(self.sim_cache.load(episode), self.grid))
            mask_parts.append(torch.from_numpy(np.asarray(query_mask, dtype=bool).copy()))
            if self.use_query_rgb:
                rgb_parts.append(rgb_to_grid_tensor(query_rgb, self.grid))
            if self.query_feature_cache is not None:
                dino_parts.append(
                    torch.from_numpy(
                        np.asarray(self.query_feature_cache.load(episode), dtype=np.float32).copy()
                    )
                )
        embedding_batch = self.sam_embedding_cache.load_batch(
            episodes, device=self.predictor.device
        )
        self.state = VectorizedPointSelectionState(
            episodes=episodes,
            sim_grid=torch.stack(sim_parts, dim=0).to(self.device),
            query_masks=torch.stack(mask_parts, dim=0).to(self.device),
            query_rgb_grid=(torch.stack(rgb_parts, dim=0).to(self.device) if rgb_parts else None),
            query_dino=(torch.stack(dino_parts, dim=0).to(self.device) if dino_parts else None),
            embedding_batch=embedding_batch,
            click_map=torch.zeros(
                batch_size, 1, self.grid, self.grid, dtype=torch.float32, device=self.device
            ),
            current_mask_grid=torch.zeros(
                batch_size, 1, self.grid, self.grid, dtype=torch.float32, device=self.device
            ),
            current_pred_masks=torch.zeros(
                batch_size, self.image_size, self.image_size, dtype=torch.bool, device=self.device
            ),
            current_ious=torch.zeros(batch_size, dtype=torch.float32, device=self.device),
            point_indices=torch.full(
                (batch_size, self.budget), -1, dtype=torch.long, device=self.device
            ),
            point_coords_px=torch.zeros(
                batch_size, self.budget, 2, dtype=torch.float32, device=self.device
            ),
            step_index=0,
        )
        return self.observation()

    def observation(self) -> dict[str, torch.Tensor]:
        state = self._require_state()
        obs = {
            "sim_map": state.sim_grid,
            "click_map": state.click_map,
            "current_mask": state.current_mask_grid,
            "step_index": torch.full(
                (self.num_envs, 1),
                float(min(state.step_index + 1, self.budget)),
                dtype=torch.float32,
                device=self.device,
            ),
            "budget": torch.full(
                (self.num_envs, 1), float(self.budget), dtype=torch.float32, device=self.device
            ),
            "current_iou": state.current_ious.view(-1, 1),
            "action_mask": self.action_mask(),
        }
        if state.query_dino is not None:
            obs["query_dino"] = state.query_dino
        if state.query_rgb_grid is not None:
            obs["query_rgb"] = state.query_rgb_grid
        return obs

    def action_mask(self) -> torch.Tensor:
        state = self._require_state()
        mask = torch.ones(self.num_envs, self.action_dim, dtype=torch.bool, device=self.device)
        used = state.point_indices[:, : state.step_index]
        if used.numel():
            valid_used = used >= 0
            if valid_used.any():
                rows = torch.arange(self.num_envs, device=self.device).unsqueeze(1).expand_as(used)
                mask[rows[valid_used], used[valid_used]] = False
        return mask

    def step(
        self, actions: torch.Tensor | np.ndarray | Sequence[int]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        state = self._require_state()
        action_tensor = torch.as_tensor(actions, dtype=torch.long, device=self.device).view(
            self.num_envs
        )
        mask = self.action_mask()
        invalid = ~mask.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        if invalid.any():
            fallback = mask.float().argmax(dim=1)
            action_tensor = torch.where(invalid, fallback, action_tensor)

        grid_x = action_tensor.remainder(self.grid)
        grid_y = torch.div(action_tensor, self.grid, rounding_mode="floor")
        point_x = ((grid_x.float() + 0.5) / float(self.grid)) * float(self.image_size)
        point_y = ((grid_y.float() + 0.5) / float(self.grid)) * float(self.image_size)
        point_x = point_x.clamp(0.0, float(self.image_size - 1))
        point_y = point_y.clamp(0.0, float(self.image_size - 1))
        point_x = torch.round(point_x)
        point_y = torch.round(point_y)

        state.point_indices[:, state.step_index] = action_tensor
        state.point_coords_px[:, state.step_index, 0] = point_x
        state.point_coords_px[:, state.step_index, 1] = point_y

        coords = state.point_coords_px[:, : state.step_index + 1, :]
        labels = torch.ones(coords.shape[:2], dtype=torch.int, device=self.device)
        pred_masks = batched_positive_point_masks(
            predictor=self.predictor,
            embedding_batch=state.embedding_batch,
            point_coords_px=coords,
            point_labels=labels,
            multimask_output=multimask_for_prefix(self.decoder_policy, coords.shape[1]),
            use_bfloat16=self.use_bfloat16,
        )

        prev_ious = state.current_ious.clone()
        current_ious = batch_iou(pred_masks, state.query_masks)
        rewards = current_ious - prev_ious
        state.current_ious = current_ious
        state.current_pred_masks = pred_masks
        state.current_mask_grid = F.interpolate(
            pred_masks.float().unsqueeze(1),
            size=(self.grid, self.grid),
            mode="bilinear",
            align_corners=False,
        )
        flat_click = state.click_map.view(self.num_envs, -1)
        flat_click[torch.arange(self.num_envs, device=self.device), action_tensor] = 1.0
        state.step_index += 1
        done = torch.full(
            (self.num_envs,), state.step_index >= self.budget, dtype=torch.bool, device=self.device
        )
        info = {
            "iou": current_ious.clone(),
            "reward": rewards.clone(),
            "grid_x": grid_x.clone(),
            "grid_y": grid_y.clone(),
            "point_x": point_x.clone(),
            "point_y": point_y.clone(),
        }
        return self.observation(), rewards, done, info

    def current_points_px(self) -> torch.Tensor:
        state = self._require_state()
        return state.point_coords_px[:, : state.step_index, :].clone()

    def current_episode_ids(self) -> list[str]:
        state = self._require_state()
        return [episode.image_id for episode in state.episodes]

    def current_pred_masks(self) -> torch.Tensor:
        state = self._require_state()
        return state.current_pred_masks.clone()

    def current_ious(self) -> torch.Tensor:
        state = self._require_state()
        return state.current_ious.clone()

    def _require_state(self) -> VectorizedPointSelectionState:
        if self.state is None:
            raise RuntimeError("VectorizedPointSelectionEnv has not been reset.")
        return self.state


def batched_positive_point_masks(
    predictor,
    embedding_batch: SamImageEmbeddingBatch,
    point_coords_px: torch.Tensor,
    point_labels: torch.Tensor,
    multimask_output: bool,
    use_bfloat16: bool = True,
) -> torch.Tensor:
    set_predictor_image_embedding_batch(predictor, embedding_batch)
    transformed = predictor.transform.apply_coords_torch(
        point_coords_px, embedding_batch.original_size
    )
    autocast_enabled = bool(use_bfloat16) and str(predictor.device).startswith("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        sparse_embeddings, dense_embeddings = predictor.model.prompt_encoder(
            points=(transformed, point_labels),
            boxes=None,
            masks=None,
        )
        low_res_masks, scores = matched_batch_mask_decode(
            predictor=predictor,
            image_embeddings=embedding_batch.embeddings.to(
                device=predictor.device, dtype=torch.float32
            ),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )
        masks = predictor.model.postprocess_masks(
            low_res_masks, embedding_batch.input_size, embedding_batch.original_size
        )
    if multimask_output:
        best = scores.argmax(dim=1)
        rows = torch.arange(masks.shape[0], device=masks.device)
        selected = masks[rows, best]
    else:
        selected = masks[:, 0]
    return selected > predictor.model.mask_threshold


def multimask_for_prefix(decoder_policy: str, num_points: int) -> bool:
    """Whether the shared decoder contract requests multimask output for a prefix."""
    try:
        from sam_rl.protocol import multimask_for_prefix as shared_multimask_for_prefix
    except ImportError:
        # Preserve old behavior when this module is used with a legacy checkout.
        return str(decoder_policy) == "legacy_multimask_first" and int(num_points) == 1
    return shared_multimask_for_prefix(decoder_policy, int(num_points))


def matched_batch_mask_decode(
    predictor,
    image_embeddings: torch.Tensor,
    sparse_prompt_embeddings: torch.Tensor,
    dense_prompt_embeddings: torch.Tensor,
    multimask_output: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    model = predictor.model
    mask_decoder = model.mask_decoder
    output_tokens = torch.cat(
        [mask_decoder.iou_token.weight, mask_decoder.mask_tokens.weight], dim=0
    )
    output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
    tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

    if image_embeddings.shape[0] == tokens.shape[0]:
        src = image_embeddings
        pos_src = model.prompt_encoder.get_dense_pe().expand(tokens.shape[0], -1, -1, -1)
    else:
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        pos_src = torch.repeat_interleave(
            model.prompt_encoder.get_dense_pe(), tokens.shape[0], dim=0
        )
    src = src + dense_prompt_embeddings
    b, c, h, w = src.shape

    hs, src = mask_decoder.transformer(src, pos_src, tokens)
    iou_token_out = hs[:, 0, :]
    mask_tokens_out = hs[:, 1 : (1 + mask_decoder.num_mask_tokens), :]
    src = src.transpose(1, 2).view(b, c, h, w)
    upscaled_embedding = mask_decoder.output_upscaling(src)
    hyper_in_list = []
    for mask_index in range(mask_decoder.num_mask_tokens):
        hyper_in_list.append(
            mask_decoder.output_hypernetworks_mlps[mask_index](mask_tokens_out[:, mask_index, :])
        )
    hyper_in = torch.stack(hyper_in_list, dim=1)
    b, c, h, w = upscaled_embedding.shape
    masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
    iou_pred = mask_decoder.iou_prediction_head(iou_token_out)
    if multimask_output:
        mask_slice = slice(1, None)
    else:
        mask_slice = slice(0, 1)
    return masks[:, mask_slice, :, :], iou_pred[:, mask_slice]


def batch_iou(pred_masks: torch.Tensor, target_masks: torch.Tensor) -> torch.Tensor:
    pred = pred_masks.to(dtype=torch.bool)
    target = target_masks.to(dtype=torch.bool)
    inter = torch.logical_and(pred, target).flatten(1).sum(dim=1).float()
    union = torch.logical_or(pred, target).flatten(1).sum(dim=1).float()
    return torch.where(union > 0.0, inter / union, torch.ones_like(union))
