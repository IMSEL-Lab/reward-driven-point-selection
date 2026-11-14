from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import random
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.version
from PIL import Image, ImageDraw, ImageFont

from sam_rl._types import (
    EvalRow,
    PerStepRow,
    RLEvalSummary,
    RLRollout,
    RolloutBatch,
    SmokeRewardCheck,
    StandardRolloutResult,
)
from sam_rl.baseline_eval import (
    ATLANTIC,
    BLACK,
    GARNET,
    LIME,
    WHITE,
    FewShotEpisode,
    boundary_band,
    boundary_iou_score,
    dice_score,
    iou_score,
    load_episode_arrays,
    load_sam_predictor,
    predict_mask_from_points,
    write_json,
)
from sam_rl.cli.common import choose_device, load_config, set_seed
from sam_rl.episode_manifest import episode_identity
from sam_rl.evaluation_provenance import (
    evaluation_config_sha256,
    evaluation_source_fingerprint,
)
from sam_rl.greedy_bc import (
    build_bc_feature_caches,
    build_fss_split_episodes,
    ensure_query_feature_caches,
    evaluate_policy,
    grid_xy_to_pixel_xy,
    load_click_policy,
    location_idx_to_grid_xy,
    rgb_to_grid_tensor,
    scalar_map_to_grid_tensor,
)
from sam_rl.mask_samples import mask_sample_dir_for_csv, write_final_mask_samples
from sam_rl.result_manifest import write_result_metadata
from sam_rl.rl_env import VectorizedPointSelectionEnv
from sam_rl.rl_policy import bc_observation_from_rl_observation, build_rl_policy, load_bc_warm_start
from sam_rl.sam_cache import (
    SamImageEmbeddingCache,
    ensure_sam_embedding_caches,
    set_predictor_image_embedding,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PPO fine-tuning for point selection with cached SAM embeddings."
    )
    parser.add_argument("--config", default="configs/rl/fss1000_ppo_warmstart_n5.yaml")
    parser.add_argument(
        "--mode", choices=["build_embeddings", "smoke", "train", "eval"], default="smoke"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--split", default="all", help="train, val, test, or all for build_embeddings/eval"
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--checkpoint", default=None, help="Checkpoint override for eval mode.")
    parser.add_argument("--seed", type=int, default=None, help="Learner seed override.")
    parser.add_argument("--resume-checkpoint", default=None, help="Complete PPO state checkpoint.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.resume_checkpoint:
        cfg.setdefault("rl", {})["resume_checkpoint"] = args.resume_checkpoint
    device = choose_device(
        args.device or cfg.get("rl", {}).get("device") or cfg["models"].get("device", "cpu")
    )
    set_seed(int(cfg.get("seed", 0)))
    if str(device).startswith("cuda"):
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass

    if args.mode == "build_embeddings":
        run_build_embeddings(
            cfg, device=device, split_arg=args.split, max_episodes=args.max_episodes
        )
        return
    if args.mode == "smoke":
        payload = run_smoke(cfg, device=device)
        print(json.dumps(payload, indent=2))
        return
    if args.mode == "train":
        payload = run_train(cfg, device=device)
        print(json.dumps(payload, indent=2))
        return
    split_names = resolve_splits(cfg, args.split)
    if len(split_names) == 1:
        payload = run_eval(
            cfg,
            device=device,
            split_arg=split_names[0],
            checkpoint_override=args.checkpoint,
            max_episodes=args.max_episodes,
        )
    else:
        payload = {
            "splits": [
                run_eval(
                    cfg,
                    device=device,
                    split_arg=split_name,
                    checkpoint_override=args.checkpoint,
                    max_episodes=args.max_episodes,
                )
                for split_name in split_names
            ]
        }
    print(json.dumps(payload, indent=2))


def run_build_embeddings(cfg: dict, device: str, split_arg: str, max_episodes: int | None) -> None:
    split_names = resolve_splits(cfg, split_arg)
    for purpose in ("train", "eval"):
        predictor, embedding_cache = build_embedding_runtime(cfg, device=device, purpose=purpose)
        for split_name in split_names:
            episodes = build_episodes(cfg, split_name, max_episodes=max_episodes)
            print(
                f"[sam-embed:{purpose}] building split={split_name} episodes={len(episodes)} device={device}",
                flush=True,
            )
            ensure_sam_embedding_caches(
                episodes=episodes,
                embedding_cache=embedding_cache,
                log_every=int(cfg["models"].get("sam_embedding_log_every", 100)),
            )
        del predictor


def run_smoke(cfg: dict, device: str) -> dict:
    smoke_episodes = int(cfg["smoke"].get("episodes", 10))
    train_episodes = build_episodes(cfg, cfg["dataset"]["train_split"], max_episodes=smoke_episodes)
    sim_cache, query_feature_cache = build_bc_feature_caches(cfg, device=device)
    ensure_query_feature_caches(
        train_episodes,
        query_feature_cache,
        log_every=int(cfg["models"].get("dino_query_log_every", 100)),
    )

    train_predictor, train_embedding_cache = build_embedding_runtime(
        cfg, device=device, purpose="train"
    )
    ensure_sam_embedding_caches(
        train_episodes,
        embedding_cache=train_embedding_cache,
        log_every=int(cfg["models"].get("sam_embedding_log_every", 100)),
    )
    train_env = build_vector_env(
        cfg=cfg,
        predictor=train_predictor,
        sim_cache=sim_cache,
        query_feature_cache=query_feature_cache,
        embedding_cache=train_embedding_cache,
        purpose="train",
    )
    policy = build_rl_policy(cfg["model"]).to(device)
    warm_start = maybe_load_warm_start(cfg, policy, device=device)
    reward_check = smoke_reward_and_coordinate_check(train_env, policy, train_episodes)

    eval_predictor, eval_embedding_cache = build_embedding_runtime(
        cfg, device=device, purpose="eval"
    )
    ensure_sam_embedding_caches(
        train_episodes,
        embedding_cache=eval_embedding_cache,
        log_every=int(cfg["models"].get("sam_embedding_log_every", 100)),
    )
    rl_eval = evaluate_rl_policy_standard(
        cfg=cfg,
        policy=policy,
        episodes=train_episodes,
        predictor=eval_predictor,
        sim_cache=sim_cache,
        query_feature_cache=query_feature_cache,
        device=device,
        output_csv=None,
    )
    bc_model, _ = load_click_policy(cfg["rl"]["warm_start_checkpoint"], device=device)
    bc_eval = evaluate_policy(
        model=bc_model,
        episodes=train_episodes,
        predictor=eval_predictor,
        sim_cache=sim_cache,
        query_feature_cache=query_feature_cache,
        budget=int(cfg["rl"]["budget"]),
        grid=int(cfg["model"].get("grid", 37)),
        image_size=int(cfg["models"]["image_size"]),
        device=device,
        output_csv=None,
        overlay_dir=None,
        overlay_limit=0,
    )
    rollout_diff = abs(float(rl_eval["mean_iou"]) - float(bc_eval["mean_iou"]))
    payload = {
        "mode": "smoke",
        "episodes": smoke_episodes,
        "warm_start": warm_start,
        "reward_check": reward_check,
        "rl_eval_mean_iou": float(rl_eval["mean_iou"]),
        "bc_eval_mean_iou": float(bc_eval["mean_iou"]),
        "rollout_diff": rollout_diff,
        "within_tolerance": bool(rollout_diff <= float(cfg["smoke"].get("max_iou_diff", 0.01))),
    }
    write_json(ensure_rl_run_dir(cfg) / "smoke_summary.json", payload)
    return payload


def run_train(cfg: dict, device: str) -> dict:
    run_started = time.perf_counter()
    cfg = copy.deepcopy(cfg)
    requested_dropout = float(cfg["model"].get("dropout", 0.1))
    cfg["model"]["dropout"] = 0.0
    budget = int(cfg["rl"]["budget"])
    num_envs = int(cfg["rl"].get("num_envs", 32))
    rollout_steps = int(cfg["rl"].get("rollout_steps", budget))
    condition_id = str(cfg["train"].get("condition_id", cfg["rl"].get("condition_id", "legacy")))
    if rollout_steps != budget:
        raise ValueError(
            "PPO rollouts must contain complete episodes: rollout_steps must equal budget."
        )
    if condition_id != "legacy" and not bool(cfg["model"].get("reserve_mask_channel", False)):
        raise ValueError("New PPO conditions must reserve the common mask input channel.")
    if str(device).startswith("cuda"):
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    total_env_steps = int(cfg["rl"].get("total_env_steps", 1_000_000))
    steps_per_update = num_envs * rollout_steps
    if steps_per_update <= 0 or total_env_steps % steps_per_update:
        raise ValueError("total_env_steps must be divisible by num_envs * rollout_steps.")
    total_updates = total_env_steps // steps_per_update
    run_dir = ensure_rl_run_dir(cfg)
    resume_path = cfg["rl"].get("resume_checkpoint")
    if any(run_dir.iterdir()) and not resume_path:
        raise FileExistsError(f"Run directory already contains outputs: {run_dir}")

    train_episodes = build_episodes(cfg, cfg["dataset"]["train_split"])
    val_episodes = build_episodes(cfg, cfg["dataset"]["val_split"])
    code_hash = ppo_code_hash()
    run_metadata = {
        "run_id": run_dir.name,
        "condition_id": condition_id,
        "training_seed": int(cfg.get("seed", 0)),
        "device": device,
        "device_name": torch.cuda.get_device_name(device)
        if str(device).startswith("cuda")
        else "CPU",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "sam_embedding_cpu_cache_budget_bytes": int(
            cfg["models"].get("sam_embedding_cpu_cache_bytes", 0)
        ),
        "effective_config": cfg,
        "effective_config_hash": config_hash(cfg),
        "code_hash": code_hash,
        "train_episode_ids_sha256": hash_json([episode.stable_id for episode in train_episodes]),
        "validation_episode_ids_sha256": hash_json([episode.stable_id for episode in val_episodes]),
        "decoder_policy": str(cfg["models"].get("decoder_policy", "legacy_multimask_first")),
        "precision": {
            "train": str(
                cfg["models"].get("train_precision", cfg["models"].get("precision", "legacy"))
            ),
            "eval": str(
                cfg["models"].get("eval_precision", cfg["models"].get("precision", "legacy"))
            ),
        },
        "ppo_dropout_probability": 0.0,
        "requested_model_dropout_probability": requested_dropout,
        "model_assets": {
            key: {
                "path": str(cfg["models"][key]),
                "sha256": hash_file_if_present(cfg["models"][key]),
            }
            for key in ("train_sam_checkpoint", "eval_sam_checkpoint")
            if cfg["models"].get(key)
        },
        "code_assets": {
            "rl_policy.py": hash_file_if_present(Path(__file__).parents[1] / "rl_policy.py"),
            "rl_env.py": hash_file_if_present(Path(__file__).parents[1] / "rl_env.py"),
            "train_rl_finetune.py": hash_file_if_present(Path(__file__)),
        },
    }
    if not (run_dir / "run_metadata.json").exists():
        write_json(run_dir / "run_metadata.json", run_metadata)
    sim_cache, query_feature_cache = build_bc_feature_caches(cfg, device=device)
    ensure_query_feature_caches(
        train_episodes,
        query_feature_cache,
        log_every=int(cfg["models"].get("dino_query_log_every", 100)),
    )
    ensure_query_feature_caches(
        val_episodes,
        query_feature_cache,
        log_every=int(cfg["models"].get("dino_query_log_every", 100)),
    )

    train_predictor, train_embedding_cache = build_embedding_runtime(
        cfg, device=device, purpose="train"
    )
    ensure_sam_embedding_caches(
        train_episodes,
        embedding_cache=train_embedding_cache,
        log_every=int(cfg["models"].get("sam_embedding_log_every", 100)),
    )
    ensure_sam_embedding_caches(
        val_episodes,
        embedding_cache=train_embedding_cache,
        log_every=int(cfg["models"].get("sam_embedding_log_every", 100)),
    )
    train_env = build_vector_env(
        cfg=cfg,
        predictor=train_predictor,
        sim_cache=sim_cache,
        query_feature_cache=query_feature_cache,
        embedding_cache=train_embedding_cache,
        purpose="train",
    )

    eval_predictor, eval_embedding_cache = build_embedding_runtime(
        cfg, device=device, purpose="eval"
    )
    ensure_sam_embedding_caches(
        val_episodes,
        embedding_cache=eval_embedding_cache,
        log_every=int(cfg["models"].get("sam_embedding_log_every", 100)),
    )

    seed = int(cfg.get("seed", 0))
    policy = build_seeded_rl_policy(cfg["model"], seed=seed, device=device)
    disable_policy_dropout(policy)
    episode_rng = random.Random(seed + 10_007)
    action_generator = torch.Generator(device=device).manual_seed(seed + 20_011)
    minibatch_generator = torch.Generator(device="cpu").manual_seed(seed + 30_013)
    init_mode, init_checkpoint, warm_start = initialize_rl_policy(cfg, policy, device)
    reference_mode, reference_path, reference_policy = load_kl_reference(cfg, device)
    init_checkpoint_hash = hash_file_if_present(init_checkpoint) if init_checkpoint else None
    reference_checkpoint_hash = hash_file_if_present(reference_path) if reference_path else None
    run_metadata["initialization"] = {
        "mode": init_mode,
        "checkpoint": str(init_checkpoint) if init_checkpoint else None,
        "checkpoint_sha256": init_checkpoint_hash,
    }
    run_metadata["kl_reference"] = {
        "mode": reference_mode,
        "checkpoint": str(reference_path) if reference_path else None,
        "checkpoint_sha256": reference_checkpoint_hash,
    }
    write_json(run_dir / "run_metadata.json", run_metadata)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=float(cfg["rl"].get("lr", 3e-4)),
        weight_decay=float(cfg["rl"].get("weight_decay", 1e-4)),
    )

    best_metric = float("-inf")
    best_update = 0
    best_model_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    sample_cursor = 0
    sample_order: list[int] | None = None
    start_update = 1
    if resume_path:
        resume_payload = torch.load(resume_path, map_location=device, weights_only=False)
        expected_hash = config_hash(cfg)
        if resume_payload.get("config_hash") != expected_hash:
            raise ValueError("Resume checkpoint configuration hash does not match this run.")
        if resume_payload.get("code_hash") != code_hash:
            raise ValueError("Resume checkpoint code hash does not match this checkout.")
        if resume_payload.get("init_checkpoint_sha256") != init_checkpoint_hash:
            raise ValueError("Resume BC initialization checkpoint changed or does not match.")
        if resume_payload.get("kl_reference_checkpoint_sha256") != reference_checkpoint_hash:
            raise ValueError("Resume KL reference checkpoint changed or does not match.")
        policy.load_state_dict(resume_payload["model_state_dict"])
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        restore_rng_state(resume_payload["rng_state"])
        sample_cursor = int(resume_payload["sample_cursor"])
        sample_order = list(resume_payload["sample_order"])
        episode_rng.setstate(resume_payload["episode_rng_state"])
        restore_generator_state(action_generator, resume_payload["action_generator_state"])
        restore_generator_state(minibatch_generator, resume_payload["minibatch_generator_state"])
        best_metric = float(resume_payload["best_metric"])
        best_update = int(resume_payload["best_update"])
        best_model_state = resume_payload.get("best_model_state_dict")
        history = list(resume_payload["history"])
        start_update = int(resume_payload["update"]) + 1
        if start_update > total_updates + 1:
            raise ValueError("Resume checkpoint is past the configured training budget.")
    eval_every_updates = int(cfg["validation"].get("evaluate_every_updates", 250))
    eval_episodes = int(cfg["validation"].get("selection_episodes", 50))
    if eval_episodes <= 0 or eval_episodes > len(val_episodes):
        raise ValueError("selection_episodes must select a nonempty subset of validation episodes.")
    if condition_id != "legacy" and eval_episodes != len(val_episodes):
        raise ValueError(
            "New PPO conditions must select checkpoints on the full validation manifest."
        )
    checkpoint_interval = max(1, int(cfg["rl"].get("checkpoint_interval_updates", 100)))
    history_log_path = run_dir / "history.jsonl"
    if resume_path:
        truncate_history_jsonl(history_log_path, start_update - 1)

    for update_idx in range(start_update, total_updates + 1):
        update_started = time.perf_counter()
        lr = linear_schedule(float(cfg["rl"].get("lr", 3e-4)), update_idx - 1, total_updates)
        entropy_coef = linear_schedule(
            float(cfg["rl"].get("ent_coef_start", cfg["rl"].get("ent_coef", 0.01))),
            update_idx - 1,
            total_updates,
            final_value=float(cfg["rl"].get("ent_coef_end", 0.001)),
        )
        kl_beta = kl_beta_schedule(
            start=float(cfg["rl"].get("kl_beta_start", 0.1))
            if reference_policy is not None
            else 0.0,
            end=float(cfg["rl"].get("kl_beta_end", 0.01)) if reference_policy is not None else 0.0,
            step=update_idx - 1,
            total_steps=total_updates,
            decay_fraction=float(cfg["rl"].get("kl_beta_decay_fraction", 0.25)),
        )
        set_optimizer_lr(optimizer, lr)

        batch_episodes, sample_cursor, sample_order = sample_episode_batch(
            train_episodes, num_envs, sample_cursor, sample_order, episode_rng
        )
        rollout = collect_rollout_batch(
            env=train_env,
            policy=policy,
            episodes=batch_episodes,
            rollout_steps=rollout_steps,
            gamma=float(cfg["rl"].get("gamma", 1.0)),
            gae_lambda=float(cfg["rl"].get("gae_lambda", 0.95)),
            action_generator=action_generator,
        )
        losses = update_policy(
            policy=policy,
            reference_policy=reference_policy,
            optimizer=optimizer,
            rollout=rollout,
            update_epochs=int(cfg["rl"].get("update_epochs", 4)),
            minibatch_size=int(cfg["rl"].get("minibatch_size", 32)),
            clip_coef=float(cfg["rl"].get("clip_coef", 0.2)),
            entropy_coef=entropy_coef,
            value_coef=float(cfg["rl"].get("vf_coef", 0.5)),
            max_grad_norm=float(cfg["rl"].get("max_grad_norm", 0.5)),
            kl_beta=kl_beta,
            normalize_advantages=bool(cfg["rl"].get("normalize_advantages", True)),
            device=device,
            minibatch_generator=minibatch_generator,
        )
        update_seconds = time.perf_counter() - update_started
        entry: dict[str, float | int] = {
            "update": update_idx,
            "env_steps": update_idx * num_envs * rollout_steps,
            "lr": lr,
            "entropy_coef": entropy_coef,
            "kl_beta": kl_beta,
            "update_seconds": update_seconds,
            "env_steps_per_second": steps_per_update / max(update_seconds, 1e-9),
            "mean_return": float(rollout["returns"].mean().item()),
            "mean_reward": float(rollout["rewards"].mean().item()),
            **losses,
        }

        should_validate = (
            update_idx == 1 or update_idx % eval_every_updates == 0 or update_idx == total_updates
        )
        if should_validate:
            validation_started = time.perf_counter()
            eval_subset = val_episodes[:eval_episodes]
            val_summary = evaluate_rl_policy_standard(
                cfg=cfg,
                policy=policy,
                episodes=eval_subset,
                predictor=eval_predictor,
                sim_cache=sim_cache,
                query_feature_cache=query_feature_cache,
                device=device,
                output_csv=None,
                embedding_cache=eval_embedding_cache,
            )
            entry["validation_seconds"] = time.perf_counter() - validation_started
            entry["val_mean_iou"] = float(val_summary["mean_iou"])
            if float(val_summary["mean_iou"]) > best_metric:
                best_metric = float(val_summary["mean_iou"])
                best_update = update_idx
                best_model_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in policy.state_dict().items()
                }
                atomic_torch_save(
                    {
                        "model_state_dict": best_model_state,
                        "model_config": cfg["model"],
                        "budget": budget,
                        "update": update_idx,
                        "best_metric": best_metric,
                        "seed": int(cfg.get("seed", 0)),
                        "training_seed": int(cfg.get("seed", 0)),
                        "run_id": run_dir.name,
                        "config_hash": config_hash(cfg),
                        "code_hash": code_hash,
                    },
                    run_dir / "best.pt",
                )

        history.append(entry)
        with history_log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(entry, sort_keys=True) + "\n")
            log_file.flush()
            os.fsync(log_file.fileno())
        if update_idx % checkpoint_interval == 0 or should_validate or update_idx == total_updates:
            checkpoint = {
                "model_state_dict": policy.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "model_config": cfg["model"],
                "update": update_idx,
                "sample_cursor": sample_cursor,
                "sample_order": sample_order,
                "episode_rng_state": episode_rng.getstate(),
                "action_generator_state": action_generator.get_state(),
                "minibatch_generator_state": minibatch_generator.get_state(),
                "best_metric": best_metric,
                "best_update": best_update,
                "best_model_state_dict": best_model_state,
                "history": history,
                "rng_state": capture_rng_state(),
                "config_hash": config_hash(cfg),
                "code_hash": code_hash,
                "effective_config": cfg,
                "init_mode": init_mode,
                "init_checkpoint": str(init_checkpoint) if init_checkpoint else None,
                "init_checkpoint_sha256": init_checkpoint_hash,
                "kl_reference_mode": reference_mode,
                "kl_reference_checkpoint": str(reference_path) if reference_path else None,
                "kl_reference_checkpoint_sha256": reference_checkpoint_hash,
                "schedule_state": {"lr": lr, "entropy_coef": entropy_coef, "kl_beta": kl_beta},
                "training_seed": seed,
            }
            atomic_torch_save(checkpoint, run_dir / "resume.pt")
        print(
            f"[ppo] update {update_idx}/{total_updates} steps={entry['env_steps']} "
            f"mean_return={entry['mean_return']:.4f}"
            + (f" val_iou={entry['val_mean_iou']:.4f}" if "val_mean_iou" in entry else ""),
            flush=True,
        )

    last_export = {
        "model_state_dict": policy.state_dict(),
        "model_config": cfg["model"],
        "budget": budget,
        "update": total_updates,
        "best_metric": best_metric,
        "seed": int(cfg.get("seed", 0)),
        "run_id": run_dir.name,
    }
    atomic_torch_save(last_export, run_dir / "last.pt")
    write_json(run_dir / "history.json", {"history": history})
    if not (run_dir / "best.pt").exists():
        best_export = dict(last_export)
        best_export["model_state_dict"] = best_model_state or policy.state_dict()
        best_export["update"] = best_update or total_updates
        atomic_torch_save(best_export, run_dir / "best.pt")
    payload = {
        "framework": "cleanrl_ppo",
        "budget": budget,
        "total_updates": total_updates,
        "total_env_steps": total_updates * num_envs * rollout_steps,
        "best_update": best_update,
        "best_metric": best_metric,
        "warm_start": warm_start,
        "initialization": {
            "mode": init_mode,
            "checkpoint": str(init_checkpoint) if init_checkpoint else None,
        },
        "kl_reference": {
            "mode": reference_mode,
            "checkpoint": str(reference_path) if reference_path else None,
        },
        "training_seed": int(cfg.get("seed", 0)),
        "device": device,
        "device_name": torch.cuda.get_device_name(device)
        if str(device).startswith("cuda")
        else "CPU",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "wallclock_seconds": time.perf_counter() - run_started,
        "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(device)
        if str(device).startswith("cuda")
        else None,
        "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(device)
        if str(device).startswith("cuda")
        else None,
        "sam_embedding_cpu_cache_budget_bytes": int(
            cfg["models"].get("sam_embedding_cpu_cache_bytes", 0)
        ),
        "effective_config_hash": config_hash(cfg),
        "resume_checkpoint": str(run_dir / "resume.pt"),
        "run_dir": str(run_dir),
    }
    run_metadata["runtime"] = {
        key: payload[key]
        for key in (
            "device",
            "device_name",
            "torch_version",
            "cuda_version",
            "wallclock_seconds",
            "peak_gpu_memory_allocated_bytes",
            "peak_gpu_memory_reserved_bytes",
            "sam_embedding_cpu_cache_budget_bytes",
        )
    }
    write_json(run_dir / "run_metadata.json", run_metadata)
    write_json(run_dir / "train_summary.json", payload)
    return payload


def run_eval(
    cfg: dict,
    device: str,
    split_arg: str,
    checkpoint_override: str | None,
    max_episodes: int | None,
) -> dict:
    split_name = normalize_split(cfg, split_arg)
    episodes = build_episodes(cfg, split_name, max_episodes=max_episodes)
    sim_cache, query_feature_cache = build_bc_feature_caches(cfg, device=device)
    ensure_query_feature_caches(
        episodes, query_feature_cache, log_every=int(cfg["models"].get("dino_query_log_every", 100))
    )
    predictor, embedding_cache = build_embedding_runtime(cfg, device=device, purpose="eval")
    ensure_sam_embedding_caches(
        episodes,
        embedding_cache=embedding_cache,
        log_every=int(cfg["models"].get("sam_embedding_log_every", 100)),
    )
    run_dir = ensure_rl_run_dir(cfg)
    checkpoint_path = Path(checkpoint_override) if checkpoint_override else run_dir / "best.pt"
    checkpoint_sha256 = hash_file_if_present(checkpoint_path)
    if checkpoint_sha256 is None:
        raise FileNotFoundError(f"Evaluation checkpoint does not exist: {checkpoint_path}.")
    payload = torch.load(checkpoint_path, map_location=device)
    policy = build_rl_policy(payload["model_config"]).to(device)
    policy.load_state_dict(payload["model_state_dict"])
    policy.eval()

    output_csv = run_dir / f"{split_name}_results.csv"
    summary = evaluate_rl_policy_standard(
        cfg=cfg,
        policy=policy,
        episodes=episodes,
        predictor=predictor,
        sim_cache=sim_cache,
        query_feature_cache=query_feature_cache,
        device=device,
        output_csv=output_csv,
        embedding_cache=embedding_cache,
    )
    write_json(run_dir / f"{split_name}_summary.json", serializable_rl_summary(summary))
    if output_csv.exists():
        write_ppo_result_sidecars(
            cfg=cfg,
            split_name=split_name,
            episodes=episodes,
            checkpoint_path=checkpoint_path,
            model_config=payload["model_config"],
            output_csv=output_csv,
            checkpoint_sha256=checkpoint_sha256,
        )
        sample_count = int(cfg.get("evaluation", {}).get("final_mask_sample_count", 0))
        if sample_count:
            rollouts_by_key = {
                episode_identity(rollout["episode"]): rollout["pred_mask"]
                for rollout in summary["rollouts"].values()
            }
            selected_episodes = episodes[: min(sample_count, len(episodes))]
            write_final_mask_samples(
                output_dir=mask_sample_dir_for_csv(output_csv),
                episodes=episodes,
                mask_entries=[
                    (episode, rollouts_by_key[episode_identity(episode)])
                    for episode in selected_episodes
                ],
                sample_count=sample_count,
                result_csv=output_csv,
            )
    if split_name == str(cfg["dataset"]["test_split"]):
        write_step_csv(run_dir / "per_step_rollout_iou.csv", summary["per_step_rows"])
        write_json(run_dir / "cluster_spread.json", summary["cluster_spread"])
        if bool(cfg.get("evaluation", {}).get("write_contact_sheet", True)):
            build_contact_sheet_rl(
                cfg=cfg,
                episodes=episodes,
                policy_rollouts=summary["rollouts"],
                sim_cache=sim_cache,
                predictor=predictor,
                output_dir=run_dir,
            )
    return {
        "split": split_name,
        "checkpoint": str(checkpoint_path),
        "mean_iou": float(summary["mean_iou"]),
        "output_csv": str(output_csv),
        "run_dir": str(run_dir),
    }


def build_embedding_runtime(cfg: dict, device: str, purpose: str):
    checkpoint = str(cfg["models"][f"{purpose}_sam_checkpoint"])
    model_type = str(cfg["models"][f"{purpose}_sam_model_type"])
    cache_dir = str(cfg["models"][f"{purpose}_sam_embedding_cache_dir"])
    precision = str(
        cfg["models"].get(f"{purpose}_precision", cfg["models"].get("precision", "fp32"))
    )
    predictor = load_sam_predictor(checkpoint=checkpoint, model_type=model_type, device=device)
    embedding_cache = SamImageEmbeddingCache(
        cache_dir=cache_dir,
        predictor=predictor,
        sam_checkpoint=checkpoint,
        sam_model_type=model_type,
        image_size=int(cfg["models"]["image_size"]),
        precision=precision,
        cpu_cache_max_bytes=int(cfg["models"].get("sam_embedding_cpu_cache_bytes", 0)),
    )
    return predictor, embedding_cache


def build_vector_env(
    cfg: dict,
    predictor,
    sim_cache,
    query_feature_cache,
    embedding_cache,
    purpose: str,
) -> VectorizedPointSelectionEnv:
    precision = str(
        cfg["models"].get(f"{purpose}_precision", cfg["models"].get("precision", "fp32"))
    ).lower()
    return VectorizedPointSelectionEnv(
        predictor=predictor,
        sim_cache=sim_cache,
        sam_embedding_cache=embedding_cache,
        image_size=int(cfg["models"]["image_size"]),
        grid=int(cfg["model"].get("grid", 37)),
        budget=int(cfg["rl"]["budget"]),
        query_feature_cache=query_feature_cache,
        use_query_rgb=bool(cfg["model"].get("use_query_rgb", False)),
        use_bfloat16=precision in {"bf16", "bfloat16"},
        decoder_policy=str(cfg["models"].get("decoder_policy", "legacy_multimask_first")),
    )


def maybe_load_warm_start(cfg: dict, policy, device: str) -> dict[str, object]:
    checkpoint = cfg["rl"].get("warm_start_checkpoint")
    if not checkpoint:
        return {"enabled": False}
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        return {"enabled": False, "missing_checkpoint": str(checkpoint_path)}
    status = load_bc_warm_start(policy, checkpoint_path=checkpoint_path, device=device)
    return {
        "enabled": True,
        "checkpoint": str(checkpoint_path),
        "missing_keys": list(status.missing_keys),
        "unexpected_keys": list(status.unexpected_keys),
    }


def initialize_rl_policy(
    cfg: dict, policy, device: str
) -> tuple[str, str | None, dict[str, object]]:
    rl_cfg = cfg["rl"]
    init_mode = str(
        rl_cfg.get("init_mode", "bc" if rl_cfg.get("warm_start_checkpoint") else "random")
    )
    init_checkpoint = rl_cfg.get("init_checkpoint")
    if init_checkpoint is None and init_mode == "bc":
        init_checkpoint = rl_cfg.get("warm_start_checkpoint")
    if init_mode not in {"bc", "random"}:
        raise ValueError("rl.init_mode must be 'bc' or 'random'.")
    if init_mode == "bc":
        if not init_checkpoint:
            raise ValueError("BC initialization requires rl.init_checkpoint.")
        warm_start = maybe_load_warm_start(
            {"rl": {"warm_start_checkpoint": init_checkpoint}}, policy, device
        )
        if not warm_start.get("enabled"):
            raise FileNotFoundError(f"BC initialization checkpoint unavailable: {init_checkpoint}")
    else:
        # Random initialization is intentionally independent of any optional KL reference.
        warm_start: dict[str, object] = {"enabled": False, "mode": "random"}
        init_checkpoint = None
    return init_mode, str(init_checkpoint) if init_checkpoint else None, warm_start


def load_kl_reference(cfg: dict, device: str):
    rl_cfg = cfg["rl"]
    reference_path = rl_cfg.get("kl_reference_checkpoint")
    reference_mode = str(
        rl_cfg.get(
            "kl_reference_mode",
            "bc" if reference_path or rl_cfg.get("warm_start_checkpoint") else "none",
        )
    )
    if reference_path is None and reference_mode == "bc":
        reference_path = rl_cfg.get("warm_start_checkpoint")
    if reference_mode == "none":
        if (
            float(rl_cfg.get("kl_beta_start", 0.0)) != 0.0
            or float(rl_cfg.get("kl_beta_end", 0.0)) != 0.0
        ):
            raise ValueError("KL beta must be zero when kl_reference_mode is 'none'.")
        return reference_mode, None, None
    if reference_mode != "bc":
        raise ValueError("rl.kl_reference_mode currently supports 'bc' or 'none'.")
    if not reference_path or not Path(reference_path).exists():
        raise FileNotFoundError(f"KL reference checkpoint unavailable: {reference_path}")
    reference_policy, _ = load_click_policy(reference_path, device=device)
    reference_policy.eval()
    for param in reference_policy.parameters():
        param.requires_grad_(False)
    return reference_mode, str(reference_path), reference_policy


def ensure_rl_run_dir(cfg: dict) -> Path:
    run_root = Path(cfg["paths"].get("run_root", "artifacts/rl/runs"))
    run_name = str(cfg["train"].get("run_name", "fss1000_rl"))
    condition = str(cfg["train"].get("condition_id", cfg["rl"].get("condition_id", "legacy")))
    seed = int(cfg.get("seed", 0))
    run_dir = run_root / f"{run_name}__{condition}__seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def disable_policy_dropout(policy) -> None:
    """PPO ratios require old and current logits from the same deterministic network."""
    for module in policy.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            module.p = 0.0


def build_seeded_rl_policy(model_cfg: dict, seed: int, device: str):
    """Reset learner RNG after lazy model/cache work, then initialize the actor."""
    set_seed(int(seed), deterministic=None)
    return build_rl_policy(model_cfg).to(device)


def config_hash(cfg: dict) -> str:
    canonical = copy.deepcopy(cfg)
    canonical.get("rl", {}).pop("resume_checkpoint", None)
    canonical.get("paths", {}).pop("run_root", None)
    canonical.get("train", {}).pop("run_name", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def ppo_code_hash() -> str:
    source_root = Path(__file__).parents[1]
    code_paths = {
        name: hash_file_if_present(source_root / name)
        for name in (
            "rl_policy.py",
            "rl_env.py",
            "greedy_bc.py",
            "baseline_eval.py",
            "sam_cache.py",
            "episode_manifest.py",
            "result_manifest.py",
        )
    }
    code_paths["train_rl_finetune.py"] = hash_file_if_present(Path(__file__))
    code_paths["common.py"] = hash_file_if_present(Path(__file__).with_name("common.py"))
    code_paths["provenance.py"] = hash_file_if_present(source_root / "provenance.py")
    return hash_json(code_paths)


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([cuda_state.cpu() for cuda_state in state["torch_cuda"]])


def restore_generator_state(generator: torch.Generator, state: Any) -> None:
    generator.set_state(state.cpu())


def atomic_torch_save(payload: object, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    temp_path = Path(temp_name)
    try:
        torch.save(payload, temp_path)
        os.replace(temp_path, destination)
    finally:
        temp_path.unlink(missing_ok=True)


def hash_json(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def hash_file_if_present(path: str | Path) -> str | None:
    file_path = Path(path)
    if not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def truncate_history_jsonl(path: Path, completed_update: int) -> None:
    if not path.exists():
        return
    retained = []
    with path.open("r", encoding="utf-8") as history_file:
        for line in history_file:
            try:
                if int(json.loads(line)["update"]) <= completed_update:
                    retained.append(line)
            except (KeyError, ValueError, json.JSONDecodeError):
                continue
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text("".join(retained), encoding="utf-8")
    os.replace(temp_path, path)


def resolve_splits(cfg: dict, split_arg: str) -> list[str]:
    if split_arg == "all":
        return [
            str(cfg["dataset"]["train_split"]),
            str(cfg["dataset"]["val_split"]),
            str(cfg["dataset"]["test_split"]),
        ]
    return [normalize_split(cfg, split_arg)]


def normalize_split(cfg: dict, split_name: str) -> str:
    split_name = str(split_name)
    valid = {
        str(cfg["dataset"]["train_split"]),
        str(cfg["dataset"]["val_split"]),
        str(cfg["dataset"]["test_split"]),
    }
    if split_name not in valid:
        raise ValueError(f"Unsupported split '{split_name}'. Expected one of {sorted(valid)}.")
    return split_name


def build_episodes(
    cfg: dict, split_name: str, max_episodes: int | None = None
) -> list[FewShotEpisode]:
    split_limit = resolve_split_limit(cfg, split_name, max_episodes=max_episodes)
    return build_fss_split_episodes(cfg, split_name, max_episodes=split_limit)


def resolve_split_limit(cfg: dict, split_name: str, max_episodes: int | None) -> int | None:
    if max_episodes is not None:
        return int(max_episodes)
    split_roles = {
        str(cfg["dataset"]["train_split"]): "train",
        str(cfg["dataset"]["val_split"]): "val",
        str(cfg["dataset"]["test_split"]): "test",
    }
    split_role = split_roles.get(str(split_name))
    if split_role is None:
        return None
    return cfg["dataset"].get(f"max_{split_role}_episodes")


# Shuffled episode order, cached across calls so a pass visits every episode once.
def sample_episode_batch(
    episodes: Sequence[FewShotEpisode],
    batch_size: int,
    cursor: int,
    order: list[int] | None = None,
    episode_rng: random.Random | None = None,
) -> tuple[list[FewShotEpisode], int, list[int]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if order is None or len(order) != len(episodes):
        order = list(range(len(episodes)))
        (episode_rng or random).shuffle(order)
    picked: list[FewShotEpisode] = []
    while len(picked) < batch_size:
        if cursor >= len(order):
            (episode_rng or random).shuffle(order)
            cursor = 0
        picked.append(episodes[order[cursor]])
        cursor += 1
    return picked, cursor, order


def collect_rollout_batch(
    env: VectorizedPointSelectionEnv,
    policy,
    episodes: Sequence[FewShotEpisode],
    rollout_steps: int,
    gamma: float,
    gae_lambda: float,
    action_generator: torch.Generator | None = None,
) -> RolloutBatch:
    obs = env.reset_batch(episodes)
    storage_obs: dict[str, list[torch.Tensor]] = {key: [] for key in obs.keys()}
    actions = []
    log_probs = []
    values = []
    rewards = []
    dones = []
    for _ in range(rollout_steps):
        for key, value in obs.items():
            storage_obs[key].append(value.clone())
        with torch.no_grad():
            action, log_prob, value = policy.act(
                obs, deterministic=False, generator=action_generator
            )
        next_obs, reward, done, _ = env.step(action)
        actions.append(action)
        log_probs.append(log_prob)
        values.append(value)
        rewards.append(reward)
        dones.append(done.float())
        obs = next_obs

    obs_batch = {key: torch.stack(value, dim=0) for key, value in storage_obs.items()}
    actions_t = torch.stack(actions, dim=0)
    log_probs_t = torch.stack(log_probs, dim=0)
    values_t = torch.stack(values, dim=0)
    rewards_t = torch.stack(rewards, dim=0)
    dones_t = torch.stack(dones, dim=0)
    advantages, returns = compute_gae(
        rewards=rewards_t,
        values=values_t,
        dones=dones_t,
        gamma=gamma,
        gae_lambda=gae_lambda,
    )
    return {
        "obs": obs_batch,
        "actions": actions_t,
        "log_probs": log_probs_t,
        "values": values_t,
        "rewards": rewards_t,
        "advantages": advantages,
        "returns": returns,
    }


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    rollout_steps, num_envs = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(num_envs, dtype=torch.float32, device=rewards.device)
    next_value = torch.zeros(num_envs, dtype=torch.float32, device=rewards.device)
    for step in reversed(range(rollout_steps)):
        nonterminal = 1.0 - dones[step]
        delta = rewards[step] + (gamma * next_value * nonterminal) - values[step]
        last_advantage = delta + (gamma * gae_lambda * nonterminal * last_advantage)
        advantages[step] = last_advantage
        next_value = values[step]
    returns = advantages + values
    return advantages, returns


def update_policy(
    policy,
    reference_policy,
    optimizer,
    rollout: RolloutBatch,
    update_epochs: int,
    minibatch_size: int,
    clip_coef: float,
    entropy_coef: float,
    value_coef: float,
    max_grad_norm: float,
    kl_beta: float,
    normalize_advantages: bool,
    device: str,
    minibatch_generator: torch.Generator | None = None,
) -> dict[str, float]:
    obs = rollout["obs"]
    actions = rollout["actions"].reshape(-1)
    old_log_probs = rollout["log_probs"].reshape(-1)
    advantages = rollout["advantages"].reshape(-1)
    returns = rollout["returns"].reshape(-1)
    if normalize_advantages and advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(
            1e-6
        )
    flat_obs = {key: value.reshape(-1, *value.shape[2:]) for key, value in obs.items()}
    num_samples = actions.shape[0]
    indices = torch.arange(num_samples)
    policy_losses = []
    value_losses = []
    entropy_values = []
    kl_values = []
    approx_kls = []
    clip_fractions = []

    for _ in range(update_epochs):
        perm = indices[torch.randperm(num_samples, generator=minibatch_generator)]
        for start in range(0, num_samples, minibatch_size):
            mb_idx = perm[start : start + minibatch_size]
            mb_obs = {key: value[mb_idx].to(device) for key, value in flat_obs.items()}
            mb_actions = actions[mb_idx].to(device)
            mb_old_log_probs = old_log_probs[mb_idx].to(device)
            mb_advantages = advantages[mb_idx].to(device)
            mb_returns = returns[mb_idx].to(device)
            new_log_prob, entropy, value, logits = policy.evaluate_actions(mb_obs, mb_actions)
            ratio = (new_log_prob - mb_old_log_probs).exp()
            pg_loss_1 = -mb_advantages * ratio
            pg_loss_2 = -mb_advantages * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
            policy_loss = torch.maximum(pg_loss_1, pg_loss_2).mean()
            value_loss = 0.5 * F.mse_loss(value, mb_returns)
            entropy_loss = entropy.mean()
            approx_kl = ((ratio - 1.0) - (new_log_prob - mb_old_log_probs)).mean()
            clip_fraction = ((ratio - 1.0).abs() > clip_coef).float().mean()
            if reference_policy is not None and kl_beta != 0.0:
                with torch.no_grad():
                    ref_logits = reference_policy(
                        bc_observation_from_rl_observation(mb_obs, reference_policy)
                    )
                    ref_logits = ref_logits.masked_fill(
                        ~mb_obs["action_mask"].bool(), torch.finfo(ref_logits.dtype).min
                    )
                    ref_probs = torch.softmax(ref_logits, dim=-1)
                kl_anchor = F.kl_div(
                    F.log_softmax(logits, dim=-1), ref_probs, reduction="batchmean"
                )
            else:
                kl_anchor = logits.new_zeros(())
            loss = (
                policy_loss
                + (value_coef * value_loss)
                - (entropy_coef * entropy_loss)
                + (kl_beta * kl_anchor)
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropy_values.append(float(entropy_loss.item()))
            kl_values.append(float(kl_anchor.item()))
            approx_kls.append(float(approx_kl.item()))
            clip_fractions.append(float(clip_fraction.item()))

    return {
        "policy_loss": float(sum(policy_losses) / max(len(policy_losses), 1)),
        "value_loss": float(sum(value_losses) / max(len(value_losses), 1)),
        "entropy": float(sum(entropy_values) / max(len(entropy_values), 1)),
        "kl_anchor": float(sum(kl_values) / max(len(kl_values), 1)),
        "approx_kl": float(sum(approx_kls) / max(len(approx_kls), 1)),
        "clip_fraction": float(sum(clip_fractions) / max(len(clip_fractions), 1)),
    }


@torch.inference_mode()
def evaluate_rl_policy(
    cfg: dict,
    policy,
    env: VectorizedPointSelectionEnv,
    episodes: Sequence[FewShotEpisode],
    batch_envs: int,
    device: str,
    output_csv: str | Path | None,
) -> RLEvalSummary:
    was_training = bool(policy.training)
    policy.eval()
    rows: list[EvalRow] = []
    rollouts: dict[str, RLRollout] = {}
    per_step_sums = [0.0 for _ in range(int(cfg["rl"]["budget"]))]
    total_iou = 0.0
    total_dice = 0.0
    total_boundary_iou = 0.0
    total_wallclock = 0.0
    cluster_distances: list[float] = []
    method_name = str(cfg["evaluation"].get("method_name", "ppo_rl"))
    budget = int(cfg["rl"]["budget"])

    for start in range(0, len(episodes), max(batch_envs, 1)):
        batch = list(episodes[start : start + batch_envs])
        batch_start = time.perf_counter()
        obs = env.reset_batch(batch)
        per_step_ious = []
        torch.zeros(len(batch), dtype=torch.bool, device=device)
        for _ in range(budget):
            action, _ = policy.actor_action(obs, deterministic=True)
            obs, _, _done, info = env.step(action)
            per_step_ious.append(info["iou"].detach().cpu())
        elapsed_ms = ((time.perf_counter() - batch_start) * 1000.0) / max(len(batch), 1)
        final_masks = env.current_pred_masks().detach().cpu().numpy().astype(bool)
        final_points = env.current_points_px().detach().cpu().numpy()
        final_ious = env.current_ious().detach().cpu().numpy()
        for step_idx, values in enumerate(per_step_ious, start=1):
            per_step_sums[step_idx - 1] += float(values.sum().item())
        for index, episode in enumerate(batch):
            _, _, _, target_mask = load_episode_arrays(
                episode, image_size=int(cfg["models"]["image_size"])
            )
            pred_mask = final_masks[index]
            points = np.round(final_points[index]).astype(np.int32)
            iou = float(iou_score(pred_mask, target_mask))
            dice = float(dice_score(pred_mask, target_mask))
            boundary_iou = float(boundary_iou_score(pred_mask, target_mask))
            row: EvalRow = {
                "method": method_name,
                "N": budget,
                "image_id": episode.image_id,
                "seed": int(cfg.get("seed", 0)),
                "training_seed": int(cfg.get("seed", 0)),
                "eval_seed": 0,
                "episode_key": episode.stable_id,
                "iou": iou,
                "dice": dice,
                "boundary_iou": boundary_iou,
                "wallclock_ms": float(elapsed_ms),
            }
            rows.append(row)
            rollouts[episode.image_id] = {
                "episode": episode,
                "points_xy": points.tolist(),
                "pred_mask": pred_mask,
                "final_iou": float(final_ious[index]),
            }
            total_iou += iou
            total_dice += dice
            total_boundary_iou += boundary_iou
            total_wallclock += float(elapsed_ms)
            cluster_distances.append(mean_pairwise_distance(points))

    if output_csv is not None:
        write_eval_csv(output_csv, rows)

    denom = max(len(rows), 1)
    per_step_rows: list[PerStepRow] = [
        {
            "N": budget,
            "step": step_idx,
            "mean_iou": float(per_step_sums[step_idx - 1] / max(len(episodes), 1)),
        }
        for step_idx in range(1, budget + 1)
    ]
    summary: RLEvalSummary = {
        "budget": budget,
        "episodes": float(len(rows)),
        "mean_iou": float(total_iou / denom),
        "mean_dice": float(total_dice / denom),
        "mean_boundary_iou": float(total_boundary_iou / denom),
        "mean_wallclock_ms": float(total_wallclock / denom),
        "per_step_rows": per_step_rows,
        "cluster_spread": {
            "budget": budget,
            "split": "eval",
            "num_episodes": len(cluster_distances),
            "mean_pairwise_distance_px": float(np.mean(cluster_distances))
            if cluster_distances
            else 0.0,
            "median_pairwise_distance_px": float(np.median(cluster_distances))
            if cluster_distances
            else 0.0,
            "std_pairwise_distance_px": float(np.std(cluster_distances))
            if cluster_distances
            else 0.0,
        },
        "rollouts": rollouts,
    }
    if was_training:
        policy.train()
    return summary


@torch.inference_mode()
def evaluate_rl_policy_standard(
    cfg: dict,
    policy,
    episodes: Sequence[FewShotEpisode],
    predictor,
    sim_cache,
    query_feature_cache,
    device: str,
    output_csv: str | Path | None,
    embedding_cache: SamImageEmbeddingCache | None = None,
) -> RLEvalSummary:
    was_training = bool(policy.training)
    policy.eval()
    rows: list[EvalRow] = []
    per_episode_step_rows: list[dict[str, object]] = []
    rollouts: dict[str, RLRollout] = {}
    budget = int(cfg["rl"]["budget"])
    per_step_sums = [0.0 for _ in range(budget)]
    total_iou = 0.0
    total_dice = 0.0
    total_boundary_iou = 0.0
    total_wallclock = 0.0
    cluster_distances: list[float] = []
    method_name = str(cfg["evaluation"].get("method_name", "ppo_rl"))

    for episode in episodes:
        result = rollout_rl_episode_standard(
            cfg=cfg,
            policy=policy,
            episode=episode,
            predictor=predictor,
            sim_cache=sim_cache,
            query_feature_cache=query_feature_cache,
            device=device,
            embedding_cache=embedding_cache,
        )
        points = np.asarray(result["points_xy"], dtype=np.int32)
        for step_idx, value in enumerate(result["per_step_ious"], start=1):
            per_step_sums[step_idx - 1] += float(value)
            per_episode_step_rows.append(
                {
                    "episode_key": episode.stable_id,
                    "training_seed": int(cfg.get("seed", 0)),
                    "eval_seed": 0,
                    "step": step_idx,
                    "iou": float(value),
                    "point_x": int(points[step_idx - 1, 0]),
                    "point_y": int(points[step_idx - 1, 1]),
                    "point_label": 1,
                }
            )
        row: EvalRow = {
            "method": method_name,
            "N": budget,
            "image_id": episode.image_id,
            "seed": int(cfg.get("seed", 0)),
            "training_seed": int(cfg.get("seed", 0)),
            "eval_seed": 0,
            "episode_key": episode.stable_id,
            "iou": float(result["iou"]),
            "dice": float(result["dice"]),
            "boundary_iou": float(result["boundary_iou"]),
            "wallclock_ms": float(result["wallclock_ms"]),
        }
        rows.append(row)
        rollouts[episode.image_id] = {
            "episode": episode,
            "points_xy": points.tolist(),
            "pred_mask": np.asarray(result["pred_mask"], dtype=bool),
            "final_iou": float(result["iou"]),
        }
        total_iou += float(result["iou"])
        total_dice += float(result["dice"])
        total_boundary_iou += float(result["boundary_iou"])
        total_wallclock += float(result["wallclock_ms"])
        cluster_distances.append(mean_pairwise_distance(points))

    if output_csv is not None:
        write_eval_csv(output_csv, rows)
        write_per_episode_step_csv(
            Path(output_csv).with_suffix(".per_step.csv"), per_episode_step_rows
        )

    denom = max(len(rows), 1)
    per_step_rows: list[PerStepRow] = [
        {
            "N": budget,
            "step": step_idx,
            "mean_iou": float(per_step_sums[step_idx - 1] / max(len(episodes), 1)),
        }
        for step_idx in range(1, budget + 1)
    ]
    summary: RLEvalSummary = {
        "budget": budget,
        "episodes": float(len(rows)),
        "mean_iou": float(total_iou / denom),
        "mean_dice": float(total_dice / denom),
        "mean_boundary_iou": float(total_boundary_iou / denom),
        "mean_wallclock_ms": float(total_wallclock / denom),
        "per_step_rows": per_step_rows,
        "cluster_spread": {
            "budget": budget,
            "split": "eval",
            "num_episodes": len(cluster_distances),
            "mean_pairwise_distance_px": float(np.mean(cluster_distances))
            if cluster_distances
            else 0.0,
            "median_pairwise_distance_px": float(np.median(cluster_distances))
            if cluster_distances
            else 0.0,
            "std_pairwise_distance_px": float(np.std(cluster_distances))
            if cluster_distances
            else 0.0,
        },
        "rollouts": rollouts,
    }
    if was_training:
        policy.train()
    return summary


@torch.inference_mode()
def rollout_rl_episode_standard(
    cfg: dict,
    policy,
    episode: FewShotEpisode,
    predictor,
    sim_cache,
    query_feature_cache,
    device: str,
    embedding_cache: SamImageEmbeddingCache | None = None,
) -> StandardRolloutResult:
    budget = int(cfg["rl"]["budget"])
    grid = int(cfg["model"].get("grid", 37))
    image_size = int(cfg["models"]["image_size"])
    _, _, query_rgb, query_mask = load_episode_arrays(episode, image_size=image_size)
    sim_map = sim_cache.load(episode)
    query_dino = query_feature_cache.load(episode) if query_feature_cache is not None else None
    install_predictor_episode_image(predictor, episode, query_rgb, embedding_cache)

    sim_grid = scalar_map_to_grid_tensor(sim_map, grid).unsqueeze(0).to(device)
    click_map = torch.zeros(1, 1, grid, grid, dtype=torch.float32, device=device)
    current_mask_grid = torch.zeros(1, 1, grid, grid, dtype=torch.float32, device=device)
    current_iou = torch.zeros(1, 1, dtype=torch.float32, device=device)
    query_rgb_tensor = None
    if getattr(policy, "use_query_rgb", False):
        query_rgb_tensor = rgb_to_grid_tensor(query_rgb, grid).unsqueeze(0).to(device)
    query_dino_tensor = None
    if getattr(policy, "use_query_dino_features", False):
        if query_dino is None:
            raise ValueError(
                f"Policy requires query_dino features for {episode.image_id}, but none were loaded."
            )
        query_dino_tensor = (
            torch.from_numpy(np.asarray(query_dino, dtype=np.float32).copy())
            .unsqueeze(0)
            .to(device)
        )

    points_xy: list[tuple[int, int]] = []
    per_step_ious: list[float] = []
    start = time.perf_counter()
    image_h, image_w = np.asarray(query_mask).shape[:2]
    for step_index in range(1, budget + 1):
        obs = {
            "sim_map": sim_grid,
            "click_map": click_map,
            "current_mask": current_mask_grid,
            "step_index": torch.tensor([[float(step_index)]], dtype=torch.float32, device=device),
            "budget": torch.tensor([[float(budget)]], dtype=torch.float32, device=device),
            "current_iou": current_iou,
            "action_mask": ~(click_map.flatten(1) > 0.5),
        }
        if query_rgb_tensor is not None:
            obs["query_rgb"] = query_rgb_tensor
        if query_dino_tensor is not None:
            obs["query_dino"] = query_dino_tensor
        action, _ = policy.actor_action(obs, deterministic=True)
        location_idx = int(action.item())
        grid_x, grid_y = location_idx_to_grid_xy(location_idx, grid)
        point_x, point_y = grid_xy_to_pixel_xy(
            grid_x, grid_y, image_width=image_w, image_height=image_h, grid=grid
        )
        point_xy = (round(point_x), round(point_y))
        points_xy.append(point_xy)
        click_map.view(-1)[location_idx] = 1.0
        pred_mask = predict_mask_from_points(
            predictor,
            np.asarray(points_xy, dtype=np.int32),
            decoder_policy=str(cfg["models"].get("decoder_policy", "legacy_multimask_first")),
            precision=str(
                cfg["models"].get("eval_precision", cfg["models"].get("precision", "fp32"))
            ),
        )
        step_iou = float(iou_score(pred_mask, query_mask))
        per_step_ious.append(step_iou)
        current_iou.fill_(step_iou)
        current_mask_grid = (
            scalar_map_to_grid_tensor(np.asarray(pred_mask, dtype=np.float32), grid)
            .unsqueeze(0)
            .to(device)
        )

    wallclock_ms = (time.perf_counter() - start) * 1000.0
    final_mask = np.asarray(pred_mask, dtype=bool)
    return {
        "query_rgb": query_rgb,
        "sim_map": sim_map,
        "pred_mask": final_mask,
        "target_mask": query_mask,
        "points_xy": np.asarray(points_xy, dtype=np.int32),
        "per_step_ious": per_step_ious,
        "iou": float(iou_score(final_mask, query_mask)),
        "dice": float(dice_score(final_mask, query_mask)),
        "boundary_iou": float(boundary_iou_score(final_mask, query_mask)),
        "wallclock_ms": float(wallclock_ms),
    }


def smoke_reward_and_coordinate_check(
    env: VectorizedPointSelectionEnv,
    policy,
    episodes: Sequence[FewShotEpisode],
) -> SmokeRewardCheck:
    was_training = bool(policy.training)
    policy.eval()
    obs = env.reset_batch(episodes)
    max_reward_delta_error = 0.0
    max_coord_error = 0.0
    positive_reward_events = 0
    done = torch.zeros(len(episodes), dtype=torch.bool, device=env.device)
    while not bool(done.all()):
        prev_iou = obs["current_iou"].clone()
        action, _ = policy.actor_action(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        expected = info["iou"].view(-1, 1) - prev_iou
        max_reward_delta_error = max(
            max_reward_delta_error, float((reward.view(-1, 1) - expected).abs().max().item())
        )
        max_coord_error = max(
            max_coord_error,
            float(
                torch.maximum(
                    (info["point_x"] - info["point_x"].clamp(0.0, env.image_size - 1)).abs(),
                    (info["point_y"] - info["point_y"].clamp(0.0, env.image_size - 1)).abs(),
                )
                .max()
                .item()
            ),
        )
        positive_reward_events += int((reward > 0.0).sum().item())
    payload: SmokeRewardCheck = {
        "max_reward_delta_error": max_reward_delta_error,
        "max_coord_error_px": max_coord_error,
        "positive_reward_events": positive_reward_events,
    }
    if was_training:
        policy.train()
    return payload


def linear_schedule(
    initial_value: float, step: int, total_steps: int, final_value: float = 0.0
) -> float:
    if total_steps <= 1:
        return float(final_value)
    frac = min(max(float(step) / float(total_steps - 1), 0.0), 1.0)
    return float(initial_value + frac * (final_value - initial_value))


def kl_beta_schedule(
    start: float, end: float, step: int, total_steps: int, decay_fraction: float
) -> float:
    decay_steps = max(1, round(float(total_steps) * float(decay_fraction)))
    if step >= decay_steps:
        return float(end)
    frac = min(max(float(step) / float(decay_steps), 0.0), 1.0)
    return float(start + frac * (end - start))


def set_optimizer_lr(optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def write_eval_csv(path: str | Path, rows: Sequence[EvalRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "N",
                "image_id",
                "seed",
                "training_seed",
                "eval_seed",
                "episode_key",
                "iou",
                "dice",
                "boundary_iou",
                "wallclock_ms",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_per_episode_step_csv(path: str | Path, rows: Sequence[dict[str, object]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "episode_key",
                "training_seed",
                "eval_seed",
                "step",
                "iou",
                "point_x",
                "point_y",
                "point_label",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def install_predictor_episode_image(
    predictor,
    episode: FewShotEpisode,
    query_rgb: np.ndarray,
    embedding_cache: SamImageEmbeddingCache | None,
) -> None:
    if embedding_cache is None:
        predictor.set_image(query_rgb, image_format="RGB")
        return
    record = embedding_cache.load(episode, device=predictor.device)
    set_predictor_image_embedding(predictor, record)


def write_ppo_result_sidecars(
    *,
    cfg: dict,
    split_name: str,
    episodes: Sequence[FewShotEpisode],
    checkpoint_path: str | Path,
    model_config: Mapping[str, object],
    output_csv: Path,
    checkpoint_sha256: str | None = None,
) -> None:
    models = cfg["models"]
    dataset_cfg = cfg["dataset"]
    dataset = str(dataset_cfg.get("name", dataset_cfg.get("dataset", "fss1000")))
    condition = str(cfg["train"].get("condition_id", cfg["rl"].get("condition_id", "legacy")))
    decoder_model = str(models.get("eval_sam_model_type", "unknown"))
    decoder_protocol = str(models.get("decoder_policy", "legacy_multimask_first"))
    manifest_path = dataset_cfg.get("episode_manifest")
    if manifest_path is None:
        manifest_path = dataset_cfg.get("episode_manifests", {}).get(split_name)
    benchmark = split_name
    protocol_map = cfg.get("experiment", {}).get("evaluation_protocol", {})
    if manifest_path is not None:
        manifest_norm = Path(str(manifest_path)).as_posix()
        for name, candidate in protocol_map.items():
            if Path(str(candidate)).as_posix() == manifest_norm:
                benchmark = str(name)
                break
    decoder_checkpoint = models.get("eval_sam_checkpoint")
    current_checkpoint_sha256 = hash_file_if_present(checkpoint_path)
    if current_checkpoint_sha256 is None:
        raise FileNotFoundError(f"Evaluation checkpoint does not exist: {checkpoint_path}.")
    if checkpoint_sha256 is not None and current_checkpoint_sha256 != checkpoint_sha256:
        raise ValueError("Evaluation checkpoint changed after it was loaded.")
    extra = {
        "protocol": f"{decoder_protocol}|{benchmark}",
        "budget": int(cfg["rl"]["budget"]),
        "checkpoint_sha256": checkpoint_sha256 or current_checkpoint_sha256,
        "evaluation_config_sha256": evaluation_config_sha256(cfg),
        "evaluation_source_sha256": evaluation_source_fingerprint(
            Path(__file__).resolve().parents[2]
        )["evaluation_source_sha256"],
        "precision": str(models.get("eval_precision", models.get("precision", "fp32"))),
        "decoder_checkpoint_sha256": (
            hash_file_if_present(decoder_checkpoint) if decoder_checkpoint else None
        ),
        "coordinate_policy": "rounded_pixel_centers_v1",
    }
    write_result_metadata(
        csv_path=output_csv,
        episodes=episodes,
        dataset=dataset,
        split=split_name,
        condition=condition,
        model_config=model_config,
        expected_result_rows=len(episodes),
        training_seed=int(cfg.get("seed", 0)),
        eval_seed=0,
        decoder=decoder_model,
        checkpoint_path=checkpoint_path,
        extra=extra,
    )
    write_result_metadata(
        csv_path=output_csv.with_suffix(".per_step.csv"),
        episodes=episodes,
        dataset=dataset,
        split=split_name,
        condition=condition,
        model_config=model_config,
        expected_result_rows=len(episodes) * int(cfg["rl"]["budget"]),
        training_seed=int(cfg.get("seed", 0)),
        eval_seed=0,
        decoder=decoder_model,
        checkpoint_path=checkpoint_path,
        extra=extra,
    )


def write_step_csv(path: str | Path, rows: Sequence[PerStepRow]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["N", "step", "mean_iou"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def serializable_rl_summary(summary: Mapping[str, object]) -> dict[str, object]:
    payload = dict(summary)
    rollouts = payload.get("rollouts")
    if isinstance(rollouts, dict):
        cleaned_rollouts: dict[str, dict[str, object]] = {}
        for image_id, rollout in rollouts.items():
            if not isinstance(rollout, dict):
                continue
            cleaned_rollout = dict(rollout)
            episode = cleaned_rollout.pop("episode", None)
            if isinstance(episode, FewShotEpisode):
                cleaned_rollout["episode_image_id"] = episode.image_id
                cleaned_rollout["episode_class_name"] = episode.class_name
                cleaned_rollout["episode_split"] = episode.split
            pred_mask = cleaned_rollout.pop("pred_mask", None)
            if pred_mask is not None:
                cleaned_rollout["pred_mask_shape"] = list(np.asarray(pred_mask).shape)
            cleaned_rollouts[str(image_id)] = cleaned_rollout
        payload["rollouts"] = cleaned_rollouts
    return payload


def mean_pairwise_distance(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] <= 1:
        return 0.0
    diff = points[:, None, :] - points[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=-1))
    upper = distances[np.triu_indices(points.shape[0], k=1)]
    return float(np.mean(upper)) if upper.size else 0.0


def load_failed_contact_episode_ids() -> list[str]:
    path = Path("artifacts/greedy_bc/analysis/failed_v1/failure_contact_sheet_N5.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [str(item["image_id"]) for item in payload["episodes"]]


def build_contact_sheet_rl(
    cfg: dict,
    episodes: Sequence[FewShotEpisode],
    policy_rollouts: Mapping[str, RLRollout],
    sim_cache,
    predictor,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    selected_ids = load_failed_contact_episode_ids()
    by_id = {episode.image_id: episode for episode in episodes}
    panel_size = int(cfg["models"]["image_size"])
    pad = 18
    header_h = 48
    caption_h = 44
    cols = 4
    canvas_w = pad + cols * (panel_size + pad)
    canvas_h = pad + len(selected_ids) * (header_h + panel_size + caption_h + pad)
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=tuple(WHITE.tolist()))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    metadata = {"budget": int(cfg["rl"]["budget"]), "episodes": []}

    oracle_dir = Path(cfg["oracle"]["cache_root"]) / str(cfg["dataset"]["test_split"])
    for row, image_id in enumerate(selected_ids):
        episode = by_id[image_id]
        rollout = policy_rollouts[image_id]
        _, _, query_rgb, target_mask = load_episode_arrays(
            episode, image_size=int(cfg["models"]["image_size"])
        )
        sim_map = sim_cache.load(episode)
        policy_mask = np.asarray(rollout["pred_mask"], dtype=bool)
        policy_points = np.asarray(rollout["points_xy"], dtype=np.int32)
        oracle_payload = json.loads(
            (oracle_dir / f"{episode.split}_{episode.image_id}.json").read_text(encoding="utf-8")
        )
        oracle_points = np.asarray(
            [
                [step["point_x"], step["point_y"]]
                for step in oracle_payload["steps"][: int(cfg["rl"]["budget"])]
            ],
            dtype=np.float32,
        )
        oracle_points_int = np.round(oracle_points).astype(np.int32)
        predictor.set_image(query_rgb, image_format="RGB")
        oracle_mask = predict_mask_sequence(predictor, oracle_points_int)
        oracle_iou = float(iou_score(oracle_mask, target_mask))
        policy_iou = float(iou_score(policy_mask, target_mask))
        panels = [
            (
                "Query+GT",
                render_overlay(
                    query_rgb,
                    np.zeros_like(target_mask, dtype=bool),
                    target_mask,
                    np.empty((0, 2), dtype=int),
                ),
            ),
            ("Sim+Clicks", render_sim(sim_map, oracle_points_int, policy_points)),
            (
                f"Oracle IoU {oracle_iou:.3f}",
                render_overlay(query_rgb, oracle_mask, target_mask, oracle_points_int),
            ),
            (
                f"RL IoU {policy_iou:.3f}",
                render_overlay(query_rgb, policy_mask, target_mask, policy_points),
            ),
        ]
        top_y = pad + row * (header_h + panel_size + caption_h + pad)
        draw.text((pad, top_y), episode.image_id, fill=tuple(BLACK.tolist()), font=font)
        for col, (label, image) in enumerate(panels):
            x = pad + col * (panel_size + pad)
            y = top_y + header_h
            canvas.paste(image, (x, y))
            draw.rectangle(
                (x, y, x + panel_size, y + panel_size), outline=tuple(BLACK.tolist()), width=2
            )
            draw.text((x, y + panel_size + 8), label, fill=tuple(BLACK.tolist()), font=font)
        metadata["episodes"].append(
            {
                "image_id": episode.image_id,
                "oracle_iou_vith": oracle_iou,
                "policy_iou_vith": policy_iou,
                "oracle_points_xy": oracle_points_int.tolist(),
                "policy_points_xy": policy_points.tolist(),
            }
        )

    canvas.save(output_dir / "contact_sheet_N5.png")
    write_json(output_dir / "contact_sheet_N5.json", metadata)


def predict_mask_sequence(predictor, points: np.ndarray) -> np.ndarray:
    coords = np.asarray(points, dtype=np.float32)
    labels = np.ones(coords.shape[0], dtype=np.int32)
    masks, _, _ = predictor.predict(
        point_coords=coords, point_labels=labels, multimask_output=False
    )
    return np.asarray(masks[0], dtype=bool)


def render_overlay(
    query_rgb: np.ndarray, pred_mask: np.ndarray, target_mask: np.ndarray, points: np.ndarray
) -> Image.Image:
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
    for x, y in np.asarray(points, dtype=int).tolist():
        draw.rectangle((x - 6, y - 6, x + 6, y + 6), outline=tuple(BLACK.tolist()), width=2)
        draw.rectangle(
            (x - 5, y - 5, x + 5, y + 5),
            outline=tuple(GARNET.tolist()),
            fill=tuple(WHITE.tolist()),
            width=2,
        )
    return image


def render_sim(
    sim_map: np.ndarray, oracle_points: np.ndarray, policy_points: np.ndarray
) -> Image.Image:
    values = np.clip(np.asarray(sim_map, dtype=np.float32), 0.0, 1.0)
    gray = np.round(values * 255.0).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    for x, y in np.asarray(oracle_points, dtype=int).tolist():
        draw.rectangle((x - 6, y - 6, x + 6, y + 6), outline=tuple(BLACK.tolist()), width=2)
        draw.rectangle((x - 5, y - 5, x + 5, y + 5), outline=tuple(LIME.tolist()), width=2)
    for x, y in np.asarray(policy_points, dtype=int).tolist():
        draw.line((x - 6, y - 6, x + 6, y + 6), fill=tuple(GARNET.tolist()), width=2)
        draw.line((x - 6, y + 6, x + 6, y - 6), fill=tuple(GARNET.tolist()), width=2)
    return image


if __name__ == "__main__":
    main()
