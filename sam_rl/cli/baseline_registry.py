from __future__ import annotations

from pathlib import Path

BASELINE_PATHS = {
    "mock_smoke": Path("configs/baselines/mock_smoke.json"),
    "local_real": Path("configs/baselines/local_real.json"),
    "local_real_search_best": Path("configs/baselines/local_real_search_best.json"),
    "local_real_promotion_best": Path("configs/baselines/local_real_promotion_best.json"),
}


def resolve_baseline_path(name: str) -> Path:
    if name not in BASELINE_PATHS:
        raise KeyError(f"Unknown baseline: {name}")
    return BASELINE_PATHS[name]
