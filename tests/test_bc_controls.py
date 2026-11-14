from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from sam_rl.baseline_eval import DinoSimilarityCache
from sam_rl.cli.train_greedy_bc import build_seeded_policy
from sam_rl.episode_manifest import EpisodeSpec
from sam_rl.greedy_bc import (
    CsvResultWriter,
    DinoQueryFeatureCache,
    GreedyClickHeatmapPolicy,
    OracleHeatmapDataset,
    atomic_torch_save,
    build_bc_result_row,
    build_candidate_coord_batch,
    load_click_policy,
    prefix_iou_scores,
    rollout_iou_aux_loss,
    rounded_grid_xy_to_pixel_xy,
    scalar_map_to_grid_tensor,
    select_points_with_model,
    train_click_policy,
)


class _OneBatchDataset(Dataset[dict[str, object]]):
    def __init__(self, batch: dict[str, object]) -> None:
        self.batch = batch

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict[str, object]:
        if index != 0:
            raise IndexError(index)
        return self.batch


class _RecordingPolicy(nn.Module):
    grid = 3
    use_query_rgb = False
    use_query_dino_features = False
    use_current_mask = True
    reserve_mask_channel = True

    def __init__(self) -> None:
        super().__init__()
        self.masks: list[torch.Tensor] = []

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        self.masks.append(obs["current_mask"].detach().cpu().clone())
        logits = torch.zeros(1, 9)
        logits[0, int(obs["click_map"].flatten().sum().item())] = 10.0
        return logits


class _FakePredictor:
    device = "cpu"

    def __init__(self) -> None:
        self.image = None
        self.set_image_calls = 0
        self.points: list[np.ndarray] = []

    def set_image(self, image: np.ndarray, image_format: str = "RGB") -> None:
        self.image = image
        self.set_image_calls += 1

    def predict(self, *, point_coords, point_labels, multimask_output):
        self.points.append(np.asarray(point_coords).copy())
        mask = np.zeros((1, 4, 4), dtype=bool)
        mask[0, :2, :2] = True
        return mask, np.ones(1, dtype=np.float32), None


class _FakeSimCache:
    def load(self, episode: EpisodeSpec) -> np.ndarray:
        del episode
        return np.zeros((14, 14), dtype=np.float32)


class BCControlsTest(unittest.TestCase):
    def test_seeded_policy_initialization_is_independent_of_prior_cache_draws(self) -> None:
        config: dict[str, int | float | bool] = {
            "grid": 3,
            "hidden_channels": 16,
            "depth": 1,
            "dropout": 0.0,
            "reserve_mask_channel": True,
            "use_current_mask": False,
        }
        torch.rand(11)
        first = build_seeded_policy(config, seed=73, device="cpu")
        torch.rand(97)
        second = build_seeded_policy(config, seed=73, device="cpu")
        for name, tensor in first.state_dict().items():
            torch.testing.assert_close(tensor, second.state_dict()[name])

    def test_atomic_checkpoint_write_does_not_advance_python_rng(self) -> None:
        random.seed(31)
        rng_state = random.getstate()
        with tempfile.TemporaryDirectory() as tmp:
            atomic_torch_save({"value": 1}, Path(tmp) / "state.pt")
        self.assertEqual(random.getstate(), rng_state)

    def test_new_model_reserves_mask_plane_even_when_disabled(self) -> None:
        reserved = GreedyClickHeatmapPolicy(
            grid=3, hidden_channels=16, depth=1, reserve_mask_channel=True
        )
        legacy = GreedyClickHeatmapPolicy(grid=3, hidden_channels=16, depth=1)
        self.assertEqual(reserved.encoder[0].in_channels, 8)
        self.assertEqual(legacy.encoder[0].in_channels, 7)

    def test_legacy_checkpoint_without_flags_keeps_39_channel_layout(self) -> None:
        config = {"grid": 3, "hidden_channels": 16, "depth": 1}
        model = GreedyClickHeatmapPolicy(grid=3, hidden_channels=16, depth=1)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "legacy.pt"
            torch.save({"model_config": config, "model_state_dict": model.state_dict()}, checkpoint)
            loaded, _ = load_click_policy(checkpoint, "cpu")
        self.assertEqual(loaded.encoder[0].in_channels, 7)

    def test_disabled_auxiliary_loss_short_circuits(self) -> None:
        logits = torch.ones(2, 9)
        loss = rollout_iou_aux_loss(
            logits,
            batch={},
            dataset=cast(OracleHeatmapDataset, None),
            predictor=None,
            image_size=8,
            grid=3,
            aux_cfg={"enabled": False, "weight": 0.1},
        )
        self.assertEqual(float(loss), 0.0)

    def test_result_row_matches_shared_csv_header(self) -> None:
        episode = EpisodeSpec(
            dataset="fss1000",
            split="test",
            class_name="bird",
            support_id="support-1",
            query_id="query-1",
            support_image_path=Path("support.jpg"),
            support_mask_path=Path("support.png"),
            query_image_path=Path("query.jpg"),
            query_mask_path=Path("query.png"),
        )
        row = build_bc_result_row(
            method="greedy_bc",
            budget=5,
            episode=episode,
            training_seed=17,
            eval_seed=0,
            result={"iou": 0.5, "dice": 0.6, "boundary_iou": 0.4, "wallclock_ms": 10.0},
        )
        self.assertEqual(set(row), set(CsvResultWriter.fieldnames))
        self.assertEqual(row["seed"], 17)
        self.assertEqual(row["training_seed"], 17)
        self.assertEqual(row["eval_seed"], 0)
        self.assertTrue(str(row["episode_key"]).startswith("fss1000:"))

    def test_fp32_dino_query_cache_preserves_fractional_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "dino.pth"
            checkpoint.write_bytes(b"fake checkpoint")
            source_paths = [root / name for name in ("si.jpg", "sm.png", "qi.jpg", "qm.png")]
            for path in source_paths:
                path.write_bytes(b"source asset")
            episode = EpisodeSpec(
                dataset="fss1000",
                split="val",
                class_name="bird",
                support_id="s1",
                query_id="q1",
                support_image_path=source_paths[0],
                support_mask_path=source_paths[1],
                query_image_path=source_paths[2],
                query_mask_path=source_paths[3],
            )
            cache = DinoQueryFeatureCache(
                cache_dir=root / "cache",
                dino_model=nn.Identity(),
                dino_checkpoint=checkpoint,
                device="cpu",
                image_size=14,
                precision="fp32",
            )
            expected = np.asarray([[[0.1234567, -0.7654321]]], dtype=np.float32)
            mock_image = np.zeros((14, 14, 3), dtype=np.uint8)
            with (
                patch(
                    "sam_rl.greedy_bc.load_episode_arrays",
                    return_value=(None, None, mock_image, None),
                ),
                patch(
                    "sam_rl.greedy_bc.compute_dino_query_feature_map", return_value=expected
                ) as compute,
            ):
                cache_path = cache.ensure(episode)
                compute.assert_called_once()
            with np.load(cache_path, allow_pickle=False) as payload:
                self.assertEqual(payload["query_dino"].dtype, np.dtype(np.float32))
                metadata = json.loads(str(payload["metadata_json"].item()))
            self.assertEqual(metadata["config"]["compute_precision"], "fp32")
            self.assertEqual(metadata["config"]["storage_dtype"], "float32")
            np.testing.assert_array_equal(cache.load(episode), expected)
            dataset = OracleHeatmapDataset.__new__(OracleHeatmapDataset)
            dataset._feature_cache = {}
            dataset.sim_cache = cast(DinoSimilarityCache, _FakeSimCache())
            dataset.grid = 3
            dataset.use_query_rgb = False
            dataset.query_feature_cache = cache
            dataset.image_size = 14
            features = dataset._episode_features(episode)
            self.assertEqual(features["query_dino"].dtype, torch.float32)
            torch.testing.assert_close(features["query_dino"], torch.from_numpy(expected))

    def test_rollout_mask_is_previous_prefix_only(self) -> None:
        policy = _RecordingPolicy()
        predictor = _FakePredictor()
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        prefix_masks: list[np.ndarray] = []
        points = select_points_with_model(
            model=policy,
            sim_map=np.zeros((4, 4), dtype=np.float32),
            budget=2,
            device="cpu",
            predictor=predictor,
            query_image=image,
            prefix_masks=prefix_masks,
        )
        self.assertEqual(points.shape, (2, 2))
        self.assertEqual(len(policy.masks), 2)
        self.assertEqual(float(policy.masks[0].sum()), 0.0)
        expected_mask_grid = scalar_map_to_grid_tensor(
            np.pad(np.ones((2, 2), dtype=np.float32), ((0, 2), (0, 2))), 3
        ).unsqueeze(0)
        torch.testing.assert_close(policy.masks[1], expected_mask_grid)
        self.assertEqual(predictor.points[0].shape[0], 1)
        self.assertEqual(predictor.set_image_calls, 1)
        np.testing.assert_array_equal(predictor.points[0], points[:1])
        np.testing.assert_array_equal(predictor.points[1], points[:2])
        self.assertEqual(len(prefix_masks), 2)
        target = np.zeros((4, 4), dtype=bool)
        target[:2, :] = True
        self.assertEqual(prefix_iou_scores(prefix_masks, target), [0.5, 0.5])
        candidate = build_candidate_coord_batch([0], [], 3, 4, 4, torch.device("cpu"))
        torch.testing.assert_close(candidate, torch.tensor([[[1.0, 1.0]]]))
        self.assertEqual(rounded_grid_xy_to_pixel_xy(0, 0, 4, 4, 3), (1.0, 1.0))
        test_mask = np.asarray([[1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
        expected_resize = F.interpolate(
            torch.from_numpy(test_mask)[None, None],
            size=(3, 3),
            mode="bilinear",
            align_corners=False,
        )[0]
        torch.testing.assert_close(scalar_map_to_grid_tensor(test_mask, 3), expected_resize)

    def test_last_checkpoint_resumes_optimizer_and_epoch(self) -> None:
        obs = {
            "sim_map": torch.tensor([[[[0.0, 1.0], [0.5, 0.2]]]]),
            "click_map": torch.zeros(1, 1, 2, 2),
            "step_index": torch.ones(1, 1),
            "budget": torch.ones(1, 1),
        }
        batches = [
            {"obs": obs, "target_idx": torch.tensor([1]), "example_index": torch.tensor([0])}
        ]
        loader = DataLoader(_OneBatchDataset(cast(dict[str, object], batches[0])), batch_size=None)
        model_cfg: dict[str, int | float | bool] = {
            "grid": 2,
            "hidden_channels": 16,
            "depth": 1,
            "dropout": 0.0,
            "reserve_mask_channel": True,
            "use_current_mask": False,
        }
        cfg = {"train": {"epochs": 1, "lr": 0.001}, "validation": {}, "model": model_cfg}
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            model = _policy_from_config(model_cfg)
            train_click_policy(
                model,
                loader,
                None,
                [],
                None,
                cast(DinoSimilarityCache, None),
                None,
                cfg,
                1,
                "cpu",
                run_dir,
            )
            checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
            self.assertIn("optimizer_state_dict", checkpoint)
            self.assertIn("torch_rng_state", checkpoint)
            cfg["train"]["epochs"] = 2
            resumed = _policy_from_config(model_cfg)
            train_click_policy(
                resumed,
                loader,
                None,
                [],
                None,
                cast(DinoSimilarityCache, None),
                None,
                cfg,
                1,
                "cpu",
                run_dir,
                resume_path=run_dir / "last.pt",
            )
            resumed_state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
            self.assertEqual(resumed_state["epoch"], 2)
            self.assertEqual(len(resumed_state["history"]), 2)

    def test_warm_start_resume_matches_uninterrupted_training(self) -> None:
        obs = {
            "sim_map": torch.tensor([[[[0.0, 1.0], [0.5, 0.2]]]]),
            "click_map": torch.zeros(1, 1, 2, 2),
            "step_index": torch.ones(1, 1),
            "budget": torch.ones(1, 1),
        }
        batch: dict[str, object] = {
            "obs": obs,
            "target_idx": torch.tensor([1]),
            "example_index": torch.tensor([0]),
        }
        loader = DataLoader(_OneBatchDataset(batch), batch_size=None)
        model_cfg: dict[str, int | float | bool] = {
            "grid": 2,
            "hidden_channels": 16,
            "depth": 1,
            "dropout": 0.0,
            "reserve_mask_channel": True,
            "use_current_mask": False,
        }
        base_cfg = {
            "train": {"epochs": 2, "lr": 0.001, "warm_start_epochs": 1},
            "validation": {},
            "model": model_cfg,
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            continuous_dir = root / "continuous"
            interrupted_dir = root / "interrupted"
            torch.manual_seed(1234)
            continuous = _policy_from_config(model_cfg)
            train_click_policy(
                continuous,
                loader,
                None,
                [],
                None,
                cast(DinoSimilarityCache, None),
                None,
                base_cfg,
                1,
                "cpu",
                continuous_dir,
            )
            interrupted_cfg = {
                **base_cfg,
                "train": {**base_cfg["train"], "epochs": 1},
            }
            torch.manual_seed(1234)
            interrupted = _policy_from_config(model_cfg)
            train_click_policy(
                interrupted,
                loader,
                None,
                [],
                None,
                cast(DinoSimilarityCache, None),
                None,
                interrupted_cfg,
                1,
                "cpu",
                interrupted_dir,
            )
            resumed = _policy_from_config(model_cfg)
            train_click_policy(
                resumed,
                loader,
                None,
                [],
                None,
                cast(DinoSimilarityCache, None),
                None,
                base_cfg,
                1,
                "cpu",
                interrupted_dir,
                resume_path=interrupted_dir / "last.pt",
            )
            continuous_state = torch.load(
                continuous_dir / "last.pt", map_location="cpu", weights_only=False
            )
            resumed_state = torch.load(
                interrupted_dir / "last.pt", map_location="cpu", weights_only=False
            )
            for key, tensor in continuous_state["model_state_dict"].items():
                torch.testing.assert_close(
                    tensor, resumed_state["model_state_dict"][key], rtol=0, atol=0
                )
            resumed_history = resumed_state["history"]
            self.assertEqual(sum("warm_start_epoch" in entry for entry in resumed_history), 1)


def _policy_from_config(config: dict[str, int | float | bool]) -> GreedyClickHeatmapPolicy:
    return GreedyClickHeatmapPolicy(
        grid=int(config["grid"]),
        hidden_channels=int(config["hidden_channels"]),
        depth=int(config["depth"]),
        dropout=float(config["dropout"]),
        reserve_mask_channel=bool(config["reserve_mask_channel"]),
        use_current_mask=bool(config["use_current_mask"]),
    )


if __name__ == "__main__":
    unittest.main()
