from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int, *, deterministic: bool | None = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic is not None:
        torch.use_deterministic_algorithms(deterministic)


def choose_device(name: str | None, *, allow_fallback: bool = False) -> str:
    requested = str(name or "cpu")
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            if allow_fallback:
                return "cpu"
            raise RuntimeError(
                f"Requested {requested}, but CUDA is unavailable. "
                "Check the driver and environment or explicitly select --device cpu."
            )
        index = torch.device(requested).index
        if index is not None and index >= torch.cuda.device_count():
            raise ValueError(f"Requested unavailable device {requested}.")
    return requested


def ensure_run_dir(cfg: dict) -> Path:
    run_root = Path(cfg["paths"].get("run_root", "artifacts/runs"))
    run_name = str(cfg["train"].get("run_name", "default_run"))
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def checkpoint_path(cfg: dict) -> Path:
    return ensure_run_dir(cfg) / str(cfg["paths"].get("checkpoint_name", "candidate.pt"))


def metrics_path(cfg: dict) -> Path:
    return ensure_run_dir(cfg) / str(cfg["paths"].get("metrics_name", "metrics.json"))


def to_device_batch(batch: dict, device: str) -> dict:
    converted = {}
    for key, value in batch.items():
        if isinstance(value, dict):
            converted[key] = {
                k: v.to(device) if torch.is_tensor(v) else v for k, v in value.items()
            }
        else:
            converted[key] = value.to(device) if torch.is_tensor(value) else value
    return converted


def dump_json(payload: dict) -> str:
    return json.dumps(payload, indent=2, sort_keys=True)
