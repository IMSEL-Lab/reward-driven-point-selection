from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

import torch

from sam_rl._types import RolloutBatch
from sam_rl.cli.train_rl_finetune import (
    build_seeded_rl_policy,
    config_hash,
    disable_policy_dropout,
    initialize_rl_policy,
    install_predictor_episode_image,
    load_kl_reference,
    restore_generator_state,
    restore_rng_state,
    sample_episode_batch,
    update_policy,
    write_ppo_result_sidecars,
)
from sam_rl.episode_manifest import EpisodeSpec
from sam_rl.greedy_bc import build_policy as build_bc_policy
from sam_rl.rl_policy import (
    PPOPointPolicy,
    bc_observation_from_rl_observation,
    load_bc_warm_start,
)


def make_obs(grid: int = 3) -> dict[str, torch.Tensor]:
    return {
        "sim_map": torch.arange(grid * grid, dtype=torch.float32).view(1, 1, grid, grid),
        "click_map": torch.zeros(1, 1, grid, grid),
        "current_mask": torch.rand(1, 1, grid, grid),
        "step_index": torch.tensor([[2.0]]),
        "budget": torch.tensor([[5.0]]),
        "current_iou": torch.tensor([[0.25]]),
        "action_mask": torch.ones(1, grid * grid, dtype=torch.bool),
        "query_masks": torch.zeros(1, grid, grid),
    }


def bc_config(*, use_mask: bool, reserve_mask: bool = True) -> dict:
    return {
        "grid": 3,
        "hidden_channels": 16,
        "depth": 2,
        "dropout": 0.0,
        "use_current_mask": use_mask,
        "reserve_mask_channel": reserve_mask,
    }


class MappedTensor:
    """Model a tensor restored on a device that an RNG API does not accept."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor

    def cpu(self) -> torch.Tensor:
        return self.tensor.cpu()


class TestPPOControls(unittest.TestCase):
    def test_eval_embedding_install_avoids_predictor_reencoding(self) -> None:
        episode = EpisodeSpec("fss1000", "test", "class", "1", "2", *([Path("x")] * 4))
        predictor = Mock()
        predictor.device = "cuda:0"
        cache = Mock()
        record = object()
        cache.load.return_value = record
        image = torch.zeros(4, 4, 3).numpy()
        with patch("sam_rl.cli.train_rl_finetune.set_predictor_image_embedding") as install:
            install_predictor_episode_image(predictor, episode, image, cache)
        cache.load.assert_called_once_with(episode, device="cuda:0")
        predictor.set_image.assert_not_called()
        install.assert_called_once_with(predictor, record)

    def test_ppo_result_sidecar_records_decoder_and_protocol_contract(self) -> None:
        episode = EpisodeSpec("fss1000", "test", "class", "1", "2", *([Path("x")] * 4))
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "sam.pth"
            checkpoint.write_bytes(b"test decoder checkpoint")
            cfg = {
                "dataset": {
                    "name": "fss1000",
                    "episode_manifests": {"test": "configs/fss_test450.json"},
                },
                "models": {
                    "eval_sam_model_type": "vit_b",
                    "eval_sam_checkpoint": str(checkpoint),
                    "decoder_policy": "single",
                    "precision": "fp32",
                },
                "train": {"condition_id": "R0"},
                "rl": {"condition_id": "R0", "budget": 5},
                "seed": 17,
                "experiment": {"evaluation_protocol": {"fss_test450": "configs/fss_test450.json"}},
            }
            results_csv = Path(temp_dir) / "test_results.csv"
            write_ppo_result_sidecars(
                cfg=cfg,
                split_name="test",
                episodes=[episode],
                checkpoint_path=checkpoint,
                model_config={"grid": 37, "reserve_mask_channel": True},
                output_csv=results_csv,
            )
            metadata_path = results_csv.with_suffix(".metadata.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["decoder"], "vit_b")
            self.assertEqual(metadata["protocol"], "single|fss_test450")
            self.assertEqual(metadata["budget"], 5)
            self.assertEqual(metadata["precision"], "fp32")
            self.assertEqual(metadata["coordinate_policy"], "rounded_pixel_centers_v1")
            self.assertTrue(metadata["decoder_checkpoint_sha256"])

    def test_policy_initialization_is_independent_of_preprocessing_rng_use(self) -> None:
        cfg = {"grid": 3, "hidden_channels": 16, "depth": 2, "dropout": 0.0}
        torch.manual_seed(1)
        _ = torch.rand(128)
        first = build_seeded_rl_policy(cfg, seed=29, device="cpu")
        torch.manual_seed(800)
        _ = torch.rand(4096)
        second = build_seeded_rl_policy(cfg, seed=29, device="cpu")
        for first_value, second_value in zip(
            first.state_dict().values(), second.state_dict().values(), strict=True
        ):
            torch.testing.assert_close(first_value, second_value)

    def test_actor_logits_do_not_consume_target_or_critic_iou(self) -> None:
        policy = PPOPointPolicy(grid=3, hidden_channels=16, depth=2, dropout=0.0)
        obs = make_obs()
        policy.eval()
        logits = policy.actor_only(obs)
        changed = {key: value.clone() for key, value in obs.items()}
        changed["current_iou"].fill_(0.99)
        changed["query_masks"].fill_(1.0)
        torch.testing.assert_close(policy.actor_only(changed), logits)
        self.assertFalse(torch.equal(policy.value(changed), policy.value(obs)))

    def test_mask_on_and_off_have_common_channel_count(self) -> None:
        obs = make_obs()
        torch.manual_seed(12)
        on = PPOPointPolicy(
            grid=3, hidden_channels=16, depth=2, use_current_mask=True, reserve_mask_channel=True
        )
        torch.manual_seed(12)
        off = PPOPointPolicy(
            grid=3, hidden_channels=16, depth=2, use_current_mask=False, reserve_mask_channel=True
        )
        self.assertEqual(on.encoder[0].in_channels, 8)
        self.assertEqual(off.encoder[0].in_channels, 8)
        on.eval()
        off.eval()
        for on_value, off_value in zip(
            on.state_dict().values(), off.state_dict().values(), strict=True
        ):
            torch.testing.assert_close(on_value, off_value)
        off_obs = {key: value for key, value in obs.items() if key != "current_mask"}
        torch.testing.assert_close(
            off._stack_inputs(off_obs)[:, -1], torch.zeros_like(obs["sim_map"][:, 0])
        )
        zero_mask_obs = {**obs, "current_mask": torch.zeros_like(obs["current_mask"])}
        torch.testing.assert_close(on.actor_only(zero_mask_obs), off.actor_only(off_obs))

    def test_warm_start_preserves_learned_mask_weights(self) -> None:
        source_cfg = bc_config(use_mask=True)
        source = build_bc_policy(source_cfg)
        with torch.no_grad():
            source.encoder[0].weight[:, -1].fill_(0.125)
            source.skip.weight[:, -1].fill_(-0.25)
        policy = PPOPointPolicy(**source_cfg)
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "bc.pt"
            torch.save(
                {"model_config": source_cfg, "model_state_dict": source.state_dict()}, checkpoint
            )
            load_bc_warm_start(policy, checkpoint, "cpu")
        torch.testing.assert_close(policy.encoder[0].weight[:, -1], source.encoder[0].weight[:, -1])
        torch.testing.assert_close(policy.skip.weight[:, -1], source.skip.weight[:, -1])

    def test_legacy_39_channel_warm_start_preserves_action_distribution(self) -> None:
        legacy_cfg = bc_config(use_mask=False, reserve_mask=False)
        source = build_bc_policy(legacy_cfg).eval()
        target_cfg = {**legacy_cfg, "reserve_mask_channel": True}
        policy = PPOPointPolicy(**target_cfg).eval()
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = Path(temp_dir) / "legacy.pt"
            torch.save(
                {"model_config": legacy_cfg, "model_state_dict": source.state_dict()}, checkpoint
            )
            load_bc_warm_start(policy, checkpoint, "cpu")
        obs = make_obs()
        bc_obs = {
            key: value for key, value in obs.items() if key not in {"action_mask", "query_masks"}
        }
        torch.testing.assert_close(policy.actor_only(obs), source(bc_obs))

    def test_reference_adapter_preserves_mask_only_for_mask_reference(self) -> None:
        obs = make_obs()
        mask_aware = build_bc_policy(bc_config(use_mask=True)).eval()
        adapter_obs = bc_observation_from_rl_observation(obs, mask_aware)
        self.assertTrue(torch.equal(adapter_obs["current_mask"], obs["current_mask"]))
        mask_off = build_bc_policy(bc_config(use_mask=False)).eval()
        zero_obs = bc_observation_from_rl_observation(obs, mask_off)
        self.assertTrue(
            torch.equal(zero_obs["current_mask"], torch.zeros_like(obs["current_mask"]))
        )
        torch.testing.assert_close(
            mask_aware(adapter_obs), mask_aware(bc_observation_from_rl_observation(obs, mask_aware))
        )

    def test_dropout_is_disabled_and_old_new_logprobs_match(self) -> None:
        policy = PPOPointPolicy(grid=3, hidden_channels=16, depth=3, dropout=0.5)
        disable_policy_dropout(policy)
        self.assertTrue(
            all(
                module.p == 0
                for module in policy.modules()
                if isinstance(module, torch.nn.Dropout2d)
            )
        )
        policy.train()
        obs = make_obs()
        action, old_log_prob, _ = policy.act(obs)
        new_log_prob, _, _, _ = policy.evaluate_actions(obs, action)
        torch.testing.assert_close(old_log_prob, new_log_prob)

    def test_seeded_episode_sampling_state_can_resume_exactly(self) -> None:
        episodes = [
            EpisodeSpec("fss1000", "train", "class", str(i), str(i + 1), *([Path("x")] * 4))
            for i in range(11)
        ]
        import random

        episode_rng = random.Random(5)
        first, cursor, order = sample_episode_batch(episodes, 4, 0, episode_rng=episode_rng)
        episode_rng_state = episode_rng.getstate()
        next_batch, next_cursor, next_order = sample_episode_batch(
            episodes, 4, cursor, order, episode_rng
        )
        episode_rng.setstate(episode_rng_state)
        resumed, resumed_cursor, resumed_order = sample_episode_batch(
            episodes, 4, cursor, order, episode_rng
        )
        self.assertEqual(next_batch, resumed)
        self.assertEqual(next_cursor, resumed_cursor)
        self.assertEqual(next_order, resumed_order)
        self.assertEqual(len(first), 4)

    def test_mapped_rng_tensors_are_moved_to_cpu_before_restore(self) -> None:
        from sam_rl.cli.train_rl_finetune import capture_rng_state

        saved = capture_rng_state()
        mapped = {
            "python": saved["python"],
            "numpy": saved["numpy"],
            "torch_cpu": MappedTensor(saved["torch_cpu"]),
            "torch_cuda": [MappedTensor(torch.tensor([1, 2], dtype=torch.uint8))],
        }
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.set_rng_state_all") as set_cuda_states,
        ):
            restore_rng_state(mapped)
        restored_cuda_states = set_cuda_states.call_args.args[0]
        self.assertEqual(restored_cuda_states[0].device.type, "cpu")
        restore_rng_state(saved)

        source = torch.Generator().manual_seed(818)
        target = torch.Generator().manual_seed(1)
        expected_state = source.get_state()
        restore_generator_state(target, MappedTensor(expected_state))
        torch.testing.assert_close(target.get_state(), expected_state)

    def test_config_hash_ignores_resume_path(self) -> None:
        base = {
            "seed": 4,
            "rl": {"resume_checkpoint": None, "lr": 0.001},
            "paths": {"run_root": "one"},
            "train": {"run_name": "run-one"},
        }
        resumed = {
            "seed": 4,
            "rl": {"resume_checkpoint": "/tmp/resume.pt", "lr": 0.001},
            "paths": {"run_root": "two"},
            "train": {"run_name": "run-two"},
        }
        self.assertEqual(config_hash(base), config_hash(resumed))

    def test_random_start_and_no_anchor_need_no_bc_checkpoint(self) -> None:
        cfg = {
            "rl": {
                "init_mode": "random",
                "init_checkpoint": None,
                "warm_start_checkpoint": "/missing/bc.pt",
                "kl_reference_mode": "none",
                "kl_reference_checkpoint": None,
                "kl_beta_start": 0.0,
                "kl_beta_end": 0.0,
            }
        }
        policy = PPOPointPolicy(grid=3, hidden_channels=16, depth=2)
        mode, checkpoint, warm_start = initialize_rl_policy(cfg, policy, "cpu")
        reference_mode, reference_path, reference_policy = load_kl_reference(cfg, "cpu")
        self.assertEqual(mode, "random")
        self.assertIsNone(checkpoint)
        self.assertFalse(warm_start["enabled"])
        self.assertEqual(reference_mode, "none")
        self.assertIsNone(reference_path)
        self.assertIsNone(reference_policy)

    def test_ppo_update_reports_approx_kl_and_clip_fraction(self) -> None:
        policy = PPOPointPolicy(grid=3, hidden_channels=16, depth=2, dropout=0.0)
        obs = make_obs()
        action, log_prob, value = policy.act(obs)
        rollout = cast(
            RolloutBatch,
            {
                "obs": {key: tensor.unsqueeze(0) for key, tensor in obs.items()},
                "actions": action.view(1, 1),
                "log_probs": log_prob.view(1, 1),
                "advantages": torch.ones(1, 1),
                "returns": value.view(1, 1).detach(),
            },
        )
        optimizer = torch.optim.SGD(policy.parameters(), lr=0.0)
        stats = update_policy(
            policy,
            None,
            optimizer,
            rollout,
            update_epochs=1,
            minibatch_size=1,
            clip_coef=0.2,
            entropy_coef=0.0,
            value_coef=0.0,
            max_grad_norm=1.0,
            kl_beta=0.0,
            normalize_advantages=False,
            device="cpu",
        )
        self.assertIn("approx_kl", stats)
        self.assertEqual(stats["clip_fraction"], 0.0)
        self.assertAlmostEqual(stats["approx_kl"], 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
