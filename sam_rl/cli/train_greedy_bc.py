from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.version import cuda as torch_cuda_version

from sam_rl.cli.common import choose_device, dump_json, load_config, set_seed
from sam_rl.evaluation_provenance import (
    evaluation_config_sha256,
    evaluation_source_fingerprint,
)
from sam_rl.greedy_bc import (
    OracleHeatmapDataset,
    build_bc_feature_caches,
    build_fss_split_episodes,
    build_policy,
    build_sam_embedding_cache,
    build_sam_predictor,
    collate_oracle_heatmap_batch,
    ensure_oracle_caches,
    ensure_query_feature_caches,
    evaluate_policy,
    load_click_policy,
    resolve_sam_spec,
    train_click_policy,
    write_json,
)
from sam_rl.provenance import file_fingerprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or evaluate greedy-oracle behavioral cloning click policies."
    )
    parser.add_argument("--config", default="configs/greedy_bc/fss1000_parallel_v2.yaml")
    parser.add_argument("--mode", choices=["build_oracle", "train", "eval", "full"], default="full")
    parser.add_argument(
        "--budget", type=int, default=None, help="Click budget N for train/eval/full modes."
    )
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cuda:0.")
    parser.add_argument(
        "--split", default="all", help="Split for build_oracle/eval. Use train, val, test, or all."
    )
    parser.add_argument("--checkpoint", default=None, help="Checkpoint override for eval mode.")
    parser.add_argument("--resume", default=None, help="Full-state checkpoint to resume.")
    parser.add_argument("--run-dir", default=None, help="Explicit output directory override.")
    parser.add_argument("--seed", type=int, default=None, help="Learner seed override.")
    parser.add_argument("--output-root", default=None, help="Run output root override.")
    parser.add_argument(
        "--max-episodes", type=int, default=None, help="Optional split episode cap override."
    )
    parser.add_argument("--shard-index", type=int, default=0, help="Oracle-build shard index.")
    parser.add_argument("--num-shards", type=int, default=1, help="Oracle-build shard count.")
    parser.add_argument(
        "--overwrite-oracle", action="store_true", help="Rebuild oracle caches even if files exist."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    if args.output_root is not None:
        cfg.setdefault("paths", {})["run_root"] = args.output_root
    budget = int(args.budget if args.budget is not None else cfg.get("train", {}).get("budget", 0))
    if args.mode in {"train", "eval", "full"} and budget <= 0:
        raise ValueError("A positive --budget is required for train/eval/full modes.")
    device = choose_device(
        args.device or cfg.get("train", {}).get("device") or cfg["models"].get("device", "cpu")
    )
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    if str(device).startswith("cuda"):
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass

    if args.mode == "build_oracle":
        run_build_oracle(
            cfg=cfg,
            device=device,
            split_arg=args.split,
            max_episodes=args.max_episodes,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
            overwrite=args.overwrite_oracle,
        )
        return

    if args.mode == "train":
        summary = run_train(
            cfg=cfg,
            device=device,
            budget=budget,
            ensure_oracle=True,
            resume_path=args.resume,
            run_dir_override=args.run_dir,
        )
        print(dump_json(summary))
        return

    if args.mode == "eval":
        summary = run_eval(
            cfg=cfg,
            device=device,
            budget=budget,
            split_name=args.split,
            checkpoint_override=args.checkpoint,
            max_episodes=args.max_episodes,
        )
        print(dump_json(summary))
        return

    summary = run_full(
        cfg=cfg,
        device=device,
        budget=budget,
        max_episodes=args.max_episodes,
    )
    print(dump_json(summary))


def run_build_oracle(
    cfg: dict,
    device: str,
    split_arg: str,
    max_episodes: int | None,
    shard_index: int,
    num_shards: int,
    overwrite: bool,
) -> None:
    split_names = resolve_build_splits(cfg, split_arg)
    oracle_predictor = build_sam_predictor(cfg, device=device, purpose="oracle")
    oracle_embedding_cache = build_sam_embedding_cache(
        cfg, predictor=oracle_predictor, device=device, purpose="oracle"
    )
    oracle_checkpoint, oracle_model_type = resolve_sam_spec(cfg, purpose="oracle")
    sim_cache, _ = build_bc_feature_caches(cfg, device=device)
    oracle_cfg = cfg["oracle"]
    for split_name in split_names:
        episodes = build_episodes(cfg, split_name, max_episodes=max_episodes)
        print(
            f"[oracle] building split={split_name} episodes={len(episodes)} device={device}",
            flush=True,
        )
        ensure_oracle_caches(
            episodes=episodes,
            oracle_dir=oracle_dir_for_split(cfg, split_name),
            predictor=oracle_predictor,
            sim_cache=sim_cache,
            grid=int(cfg["model"].get("grid", 37)),
            max_budget=int(oracle_cfg["max_budget"]),
            candidate_batch_size=int(oracle_cfg.get("candidate_batch_size", 256)),
            candidate_pool_size=oracle_cfg.get("candidate_pool_size"),
            use_bfloat16=bool(oracle_cfg.get("use_bfloat16", False)),
            sam_checkpoint=oracle_checkpoint,
            sam_model_type=oracle_model_type,
            overwrite=overwrite,
            shard_index=shard_index,
            num_shards=num_shards,
            log_every=int(oracle_cfg.get("log_every", 25)),
            decoder_policy=str(cfg.get("protocol", {}).get("decoder_policy", "single")),
            sam_embedding_cache=oracle_embedding_cache,
        )


def run_train(
    cfg: dict,
    device: str,
    budget: int,
    ensure_oracle: bool,
    resume_path: str | None = None,
    run_dir_override: str | None = None,
) -> dict:
    train_split = str(cfg["dataset"]["train_split"])
    val_split = str(cfg["dataset"]["val_split"])
    train_episodes = build_episodes(cfg, train_split)
    val_episodes = build_episodes(cfg, val_split)
    predictor = build_sam_predictor(cfg, device=device, purpose="train")
    sam_embedding_cache = build_sam_embedding_cache(
        cfg, predictor=predictor, device=device, purpose="train"
    )
    sim_cache, query_feature_cache = build_bc_feature_caches(cfg, device=device)
    query_feature_log_every = int(cfg["models"].get("dino_query_log_every", 100))
    if ensure_oracle:
        models_cfg = cfg["models"]
        oracle_precision = str(
            models_cfg.get("oracle_precision", models_cfg.get("precision", "fp32"))
        )
        train_precision = str(
            models_cfg.get("train_precision", models_cfg.get("precision", "fp32"))
        )
        if (
            resolve_sam_spec(cfg, purpose="oracle") == resolve_sam_spec(cfg, purpose="train")
            and oracle_precision == train_precision
        ):
            oracle_predictor = predictor
            oracle_embedding_cache = sam_embedding_cache
        else:
            oracle_predictor = build_sam_predictor(cfg, device=device, purpose="oracle")
            oracle_embedding_cache = build_sam_embedding_cache(
                cfg, predictor=oracle_predictor, device=device, purpose="oracle"
            )
        oracle_checkpoint, oracle_model_type = resolve_sam_spec(cfg, purpose="oracle")
        oracle_cfg = cfg["oracle"]
        ensure_oracle_caches(
            episodes=train_episodes,
            oracle_dir=oracle_dir_for_split(cfg, train_split),
            predictor=oracle_predictor,
            sim_cache=sim_cache,
            grid=int(cfg["model"].get("grid", 37)),
            max_budget=int(oracle_cfg["max_budget"]),
            candidate_batch_size=int(oracle_cfg.get("candidate_batch_size", 256)),
            candidate_pool_size=oracle_cfg.get("candidate_pool_size"),
            use_bfloat16=bool(oracle_cfg.get("use_bfloat16", False)),
            sam_checkpoint=oracle_checkpoint,
            sam_model_type=oracle_model_type,
            overwrite=False,
            log_every=int(oracle_cfg.get("log_every", 25)),
            decoder_policy=str(cfg.get("protocol", {}).get("decoder_policy", "single")),
            sam_embedding_cache=oracle_embedding_cache,
        )
        ensure_oracle_caches(
            episodes=val_episodes,
            oracle_dir=oracle_dir_for_split(cfg, val_split),
            predictor=oracle_predictor,
            sim_cache=sim_cache,
            grid=int(cfg["model"].get("grid", 37)),
            max_budget=int(oracle_cfg["max_budget"]),
            candidate_batch_size=int(oracle_cfg.get("candidate_batch_size", 256)),
            candidate_pool_size=oracle_cfg.get("candidate_pool_size"),
            use_bfloat16=bool(oracle_cfg.get("use_bfloat16", False)),
            sam_checkpoint=oracle_checkpoint,
            sam_model_type=oracle_model_type,
            overwrite=False,
            log_every=int(oracle_cfg.get("log_every", 25)),
            decoder_policy=str(cfg.get("protocol", {}).get("decoder_policy", "single")),
            sam_embedding_cache=oracle_embedding_cache,
        )
    ensure_query_feature_caches(
        train_episodes, query_feature_cache, log_every=query_feature_log_every
    )
    ensure_query_feature_caches(
        val_episodes, query_feature_cache, log_every=query_feature_log_every
    )

    run_dir = Path(run_dir_override) if run_dir_override else run_dir_for_budget(cfg, budget)
    if run_dir.exists() and any(run_dir.iterdir()) and resume_path is None:
        raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")
    if (
        bool(cfg["model"].get("use_current_mask", False))
        and int(cfg["train"].get("num_workers", 0)) != 0
    ):
        raise ValueError(
            "Mask-aware BC decodes causal prefixes in the main process; set num_workers=0."
        )
    train_dataset = OracleHeatmapDataset(
        episodes=train_episodes,
        oracle_dir=oracle_dir_for_split(cfg, train_split),
        sim_cache=sim_cache,
        image_size=int(cfg["models"]["image_size"]),
        grid=int(cfg["model"].get("grid", 37)),
        budget=budget,
        query_feature_cache=query_feature_cache,
        use_query_rgb=bool(cfg["model"].get("use_query_rgb", False)),
        predictor=predictor if bool(cfg["model"].get("use_current_mask", False)) else None,
        use_current_mask=bool(cfg["model"].get("use_current_mask", False)),
        mask_cache_dir=oracle_dir_for_split(cfg, train_split) / "prefix_masks",
        decoder_policy=str(cfg.get("protocol", {}).get("decoder_policy", "single")),
        decoder_checkpoint=resolve_sam_spec(cfg, purpose="train")[0],
        sam_embedding_cache=sam_embedding_cache,
    )
    val_dataset = OracleHeatmapDataset(
        episodes=val_episodes,
        oracle_dir=oracle_dir_for_split(cfg, val_split),
        sim_cache=sim_cache,
        image_size=int(cfg["models"]["image_size"]),
        grid=int(cfg["model"].get("grid", 37)),
        budget=budget,
        query_feature_cache=query_feature_cache,
        use_query_rgb=bool(cfg["model"].get("use_query_rgb", False)),
        predictor=predictor if bool(cfg["model"].get("use_current_mask", False)) else None,
        use_current_mask=bool(cfg["model"].get("use_current_mask", False)),
        mask_cache_dir=oracle_dir_for_split(cfg, val_split) / "prefix_masks",
        decoder_policy=str(cfg.get("protocol", {}).get("decoder_policy", "single")),
        decoder_checkpoint=resolve_sam_spec(cfg, purpose="train")[0],
        sam_embedding_cache=sam_embedding_cache,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg["train"].get("batch_size", 64)),
        shuffle=True,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        collate_fn=collate_oracle_heatmap_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(cfg["train"].get("eval_batch_size", cfg["train"].get("batch_size", 64))),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_oracle_heatmap_batch,
    )
    is_cuda = str(device).startswith("cuda")
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    train_start = time.perf_counter()
    model = build_seeded_policy(cfg["model"], seed=int(cfg.get("seed", 0)), device=device)
    train_summary = train_click_policy(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        val_episodes=val_episodes,
        predictor=predictor,
        sim_cache=sim_cache,
        query_feature_cache=query_feature_cache,
        cfg=cfg,
        budget=budget,
        device=device,
        run_dir=run_dir,
        resume_path=resume_path,
        sam_embedding_cache=sam_embedding_cache,
    )
    train_summary["training_wallclock_seconds"] = time.perf_counter() - train_start
    train_summary["gpu_name"] = torch.cuda.get_device_name(device) if is_cuda else None
    train_summary["torch_version"] = torch.__version__
    train_summary["cuda_version"] = torch_cuda_version
    train_summary["cuda_peak_memory_allocated_bytes"] = (
        int(torch.cuda.max_memory_allocated(device)) if is_cuda else 0
    )
    train_summary["cuda_peak_memory_reserved_bytes"] = (
        int(torch.cuda.max_memory_reserved(device)) if is_cuda else 0
    )
    train_summary["sam_embedding_cpu_cache_max_bytes"] = int(
        cfg["models"].get("sam_embedding_cpu_cache_bytes", 0)
    )
    train_summary.update(
        {
            "run_dir": str(run_dir),
            "train_examples": len(train_dataset),
            "val_examples": len(val_dataset),
            "train_episodes": len(train_episodes),
            "val_episodes": len(val_episodes),
            "device": device,
            "learner_seed": int(cfg.get("seed", 0)),
            "aux_rollout_iou": {
                "enabled": bool(cfg["train"].get("aux_rollout_iou", {}).get("enabled", False)),
                "weight": float(cfg["train"].get("aux_rollout_iou", {}).get("weight", 0.0)),
                "objective": (
                    "imitation_plus_expected_decoder_iou"
                    if bool(cfg["train"].get("aux_rollout_iou", {}).get("enabled", False))
                    and float(cfg["train"].get("aux_rollout_iou", {}).get("weight", 0.0)) > 0.0
                    else "pure_imitation"
                ),
            },
            "config_sha256": hashlib.sha256(
                json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")
            ).hexdigest(),
        }
    )
    write_json(run_dir / "train_summary.json", train_summary)
    return train_summary


def run_eval(
    cfg: dict,
    device: str,
    budget: int,
    split_name: str,
    checkpoint_override: str | None,
    max_episodes: int | None,
) -> dict:
    split_name = normalize_eval_split(cfg, split_name)
    episodes = build_episodes(cfg, split_name, max_episodes=max_episodes)
    predictor = build_sam_predictor(cfg, device=device, purpose="eval")
    sam_embedding_cache = build_sam_embedding_cache(
        cfg, predictor=predictor, device=device, purpose="eval"
    )
    sim_cache, query_feature_cache = build_bc_feature_caches(cfg, device=device)
    ensure_query_feature_caches(
        episodes, query_feature_cache, log_every=int(cfg["models"].get("dino_query_log_every", 100))
    )
    checkpoint_path = (
        Path(checkpoint_override)
        if checkpoint_override is not None
        else run_dir_for_budget(cfg, budget) / "best.pt"
    )
    checkpoint_sha256 = file_fingerprint(checkpoint_path)["sha256"]
    model, checkpoint_payload = load_click_policy(checkpoint_path, device=device)
    if file_fingerprint(checkpoint_path)["sha256"] != checkpoint_sha256:
        raise ValueError("BC checkpoint changed while it was being loaded.")
    evaluation_source_sha256 = evaluation_source_fingerprint(Path(__file__).resolve().parents[2])[
        "evaluation_source_sha256"
    ]
    resolved_eval_config_sha256 = evaluation_config_sha256(cfg)
    training_seed = int(checkpoint_payload.get("training_seed", cfg.get("seed", 0)))
    eval_seed = int(cfg.get("evaluation", {}).get("eval_seed", 0))
    run_dir = run_dir_for_budget(cfg, budget)
    split_csv = run_dir / f"{split_name}_results.csv"
    split_overlay_dir = run_dir / "overlays" / split_name
    summary = evaluate_policy(
        model=model,
        episodes=episodes,
        predictor=predictor,
        sim_cache=sim_cache,
        query_feature_cache=query_feature_cache,
        budget=budget,
        grid=int(cfg["model"].get("grid", 37)),
        image_size=int(cfg["models"]["image_size"]),
        device=device,
        output_csv=split_csv,
        overlay_dir=split_overlay_dir,
        overlay_limit=int(cfg["evaluation"].get("overlay_limit", 12)),
        method_name=str(cfg["evaluation"].get("method_name", "greedy_bc")),
        decoder_policy=str(cfg.get("protocol", {}).get("decoder_policy", "single")),
        training_seed=training_seed,
        eval_seed=eval_seed,
        result_metadata={
            "dataset": episodes[0].dataset
            if episodes
            else str(cfg["dataset"].get("name", "unknown")),
            "split": split_name,
            "condition": str(
                cfg.get(
                    "condition",
                    cfg.get("experiment", {}).get(
                        "condition",
                        cfg.get("train", {}).get(
                            "condition_id", cfg.get("train", {}).get("run_name", "greedy_bc")
                        ),
                    ),
                )
            ),
            "model_config": cfg["model"],
            "decoder": resolve_sam_spec(cfg, purpose="eval")[1],
            "extra": {
                "protocol": _evaluation_protocol(cfg),
                "budget": int(budget),
                "precision": str(
                    cfg["models"].get("eval_precision", cfg["models"].get("precision", "fp32"))
                ),
                "image_size": int(cfg["models"]["image_size"]),
                "decoder_checkpoint_sha256": file_fingerprint(
                    resolve_sam_spec(cfg, purpose="eval")[0]
                )["sha256"],
                "checkpoint_sha256": checkpoint_sha256,
                "evaluation_config_sha256": resolved_eval_config_sha256,
                "evaluation_source_sha256": evaluation_source_sha256,
            },
        },
        sam_embedding_cache=sam_embedding_cache,
        final_mask_sample_count=int(cfg["evaluation"].get("final_mask_sample_count", 0)),
    )
    payload = {
        "budget": int(budget),
        "split": split_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint_payload.get("epoch", -1)),
        "training_seed": training_seed,
        "eval_seed": eval_seed,
        "device": device,
        **summary,
    }
    write_json(run_dir / f"{split_name}_summary.json", payload)
    return payload


def run_full(cfg: dict, device: str, budget: int, max_episodes: int | None) -> dict:
    train_summary = run_train(cfg=cfg, device=device, budget=budget, ensure_oracle=True)
    val_summary = run_eval(
        cfg=cfg,
        device=device,
        budget=budget,
        split_name=str(cfg["dataset"]["val_split"]),
        checkpoint_override=None,
        max_episodes=max_episodes,
    )
    test_summary = run_eval(
        cfg=cfg,
        device=device,
        budget=budget,
        split_name=str(cfg["dataset"]["test_split"]),
        checkpoint_override=None,
        max_episodes=max_episodes,
    )
    summary = {
        "budget": int(budget),
        "device": device,
        "run_dir": str(run_dir_for_budget(cfg, budget)),
        "train": train_summary,
        "val": val_summary,
        "test": test_summary,
    }
    write_json(run_dir_for_budget(cfg, budget) / "summary.json", summary)
    return summary


def run_dir_for_budget(cfg: dict, budget: int) -> Path:
    run_root = Path(cfg["paths"].get("run_root", "artifacts/greedy_bc/runs"))
    run_name = str(cfg["train"].get("run_name", "fss1000_greedy_bc"))
    return run_root / f"{run_name}_seed{int(cfg.get('seed', 0))}_N{int(budget)}"


def build_seeded_policy(model_cfg: dict, seed: int, device: str) -> torch.nn.Module:
    """Reset learner RNG after frozen-model/data-cache setup before actor initialization."""
    set_seed(int(seed), deterministic=None)
    return build_policy(model_cfg).to(device)


def _evaluation_protocol(cfg: dict) -> str:
    policy = str(cfg.get("protocol", {}).get("decoder_policy", "single"))
    benchmark = cfg.get("experiment", {}).get("benchmark")
    return f"{policy}|{benchmark}" if benchmark else policy


def oracle_dir_for_split(cfg: dict, split_name: str) -> Path:
    return Path(cfg["oracle"]["cache_root"]) / str(split_name)


def build_episodes(cfg: dict, split_name: str, max_episodes: int | None = None):
    split_limit = resolve_split_limit(cfg, split_name, max_episodes=max_episodes)
    return build_fss_split_episodes(cfg, split_name, max_episodes=split_limit)


def resolve_build_splits(cfg: dict, split_arg: str) -> list[str]:
    if split_arg == "all":
        return [str(cfg["dataset"]["train_split"]), str(cfg["dataset"]["val_split"])]
    return [normalize_eval_split(cfg, split_arg)]


def normalize_eval_split(cfg: dict, split_name: str) -> str:
    split_name = str(split_name)
    valid = {
        str(cfg["dataset"]["train_split"]),
        str(cfg["dataset"]["val_split"]),
        str(cfg["dataset"]["test_split"]),
    }
    if split_name not in valid:
        raise ValueError(f"Unsupported split '{split_name}'. Expected one of {sorted(valid)}.")
    return split_name


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


if __name__ == "__main__":
    main()
