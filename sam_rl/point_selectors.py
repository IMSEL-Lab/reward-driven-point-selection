from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from sam_rl.greedy_bc import BudgetedGreedyBCSelector

PointSelector = Callable[[np.ndarray, int, np.random.Generator], np.ndarray]


@dataclass(frozen=True)
class BaselineMethod:
    family: str
    method: str
    params: dict[str, object]
    select_points: PointSelector


def build_method(family: str, params: dict[str, object] | None = None) -> BaselineMethod:
    resolved = dict(params or {})
    builders: dict[str, Callable[..., BaselineMethod]] = {
        "topk_argmax": _build_topk_argmax,
        "topk_nms": _build_topk_nms,
        "farthest_point_sampling": _build_farthest_point_sampling,
        "sim_weighted_fps": _build_sim_weighted_fps,
        "topk_kmeans": _build_topk_kmeans,
        "local_maxima_peaks": _build_local_maxima_peaks,
        "connected_component_centers": _build_connected_component_centers,
        "mean_shift_modes": _build_mean_shift_modes,
        "quantile_spaced": _build_quantile_spaced,
        "sim_weighted_random": _build_sim_weighted_random,
        "dpp": _build_dpp,
        "greedy_bc_policy": _build_greedy_bc_policy,
        "mask_aware_greedy": _build_mask_aware_greedy,
    }
    if family not in builders:
        raise KeyError(f"Unsupported baseline family: {family}")
    return builders[family](resolved)


def supported_families() -> tuple[str, ...]:
    return (
        "topk_argmax",
        "topk_nms",
        "farthest_point_sampling",
        "sim_weighted_fps",
        "topk_kmeans",
        "local_maxima_peaks",
        "connected_component_centers",
        "mean_shift_modes",
        "quantile_spaced",
        "sim_weighted_random",
        "dpp",
        "greedy_bc_policy",
        "mask_aware_greedy",
    )


def _build_topk_argmax(params: dict[str, int | float]) -> BaselineMethod:
    def select_points(sim_map: np.ndarray, n_points: int, rng: np.random.Generator) -> np.ndarray:
        """Return the N highest-similarity pixels in descending similarity order."""
        del rng
        order, width = _flatten_rank(sim_map)
        return _coords_from_indices(order[:n_points], width)

    return BaselineMethod(
        family="topk_argmax",
        method="topk_argmax",
        params={},
        select_points=select_points,
    )


def _build_topk_nms(params: dict[str, int | float]) -> BaselineMethod:
    radius = int(params.get("radius", 12))

    def select_points(sim_map: np.ndarray, n_points: int, rng: np.random.Generator) -> np.ndarray:
        """Greedy top-k with radius-NMS; if suppression yields too few points, backfill from global top-k."""
        del rng
        order, width = _flatten_rank(sim_map)
        coords = _coords_from_indices(order, width)
        selected: list[int] = []
        selected_coords: list[np.ndarray] = []
        radius_sq = float(radius * radius)
        for index, coord in zip(order.tolist(), coords, strict=True):
            if selected_coords:
                d_sq = np.sum((np.vstack(selected_coords) - coord[None, :]) ** 2, axis=1)
                if bool(np.any(d_sq <= radius_sq)):
                    continue
            selected.append(int(index))
            selected_coords.append(coord)
            if len(selected) == n_points:
                break
        return _finalize_selection(sim_map, selected, n_points)

    return BaselineMethod(
        family="topk_nms",
        method=_format_method_name("topk_nms", {"radius": radius}),
        params={"radius": radius},
        select_points=select_points,
    )


def _build_farthest_point_sampling(params: dict[str, int | float]) -> BaselineMethod:
    del params

    def select_points(sim_map: np.ndarray, n_points: int, rng: np.random.Generator) -> np.ndarray:
        """Seed with the global argmax, then repeatedly pick the pixel farthest from the selected set."""
        del rng
        flat = sim_map.reshape(-1)
        width = sim_map.shape[1]
        coords = _all_pixel_coords(sim_map.shape)
        order, _ = _flatten_rank(sim_map)
        selected = [int(order[0])]
        min_sqdist = np.sum((coords - coords[selected[0]][None, :]) ** 2, axis=1)
        min_sqdist[selected[0]] = -1.0
        for _ in range(1, n_points):
            next_index = _argmax_with_similarity_tiebreak(min_sqdist, flat, selected)
            selected.append(next_index)
            d_sq = np.sum((coords - coords[next_index][None, :]) ** 2, axis=1)
            min_sqdist = np.minimum(min_sqdist, d_sq)
            min_sqdist[selected] = -1.0
        return _coords_from_indices(np.asarray(selected, dtype=np.int64), width)

    return BaselineMethod(
        family="farthest_point_sampling",
        method="farthest_point_sampling",
        params={},
        select_points=select_points,
    )


def _build_sim_weighted_fps(params: dict[str, int | float]) -> BaselineMethod:
    alpha = float(params.get("alpha", 0.5))

    def select_points(sim_map: np.ndarray, n_points: int, rng: np.random.Generator) -> np.ndarray:
        """Score pixels by alpha*sim + (1-alpha)*normalized min-distance, seeded from the global argmax."""
        del rng
        flat = sim_map.reshape(-1)
        width = sim_map.shape[1]
        coords = _all_pixel_coords(sim_map.shape)
        order, _ = _flatten_rank(sim_map)
        selected = [int(order[0])]
        max_dist = max(float(np.hypot(sim_map.shape[0] - 1, sim_map.shape[1] - 1)), 1.0)
        min_sqdist = np.sum((coords - coords[selected[0]][None, :]) ** 2, axis=1)
        min_sqdist[selected[0]] = -1.0
        for _ in range(1, n_points):
            min_dist = np.sqrt(np.clip(min_sqdist, 0.0, None)) / max_dist
            score = alpha * flat + (1.0 - alpha) * min_dist
            score[selected] = -1.0
            next_index = _argmax_with_similarity_tiebreak(score, flat, selected)
            selected.append(next_index)
            d_sq = np.sum((coords - coords[next_index][None, :]) ** 2, axis=1)
            min_sqdist = np.minimum(min_sqdist, d_sq)
            min_sqdist[selected] = -1.0
        return _coords_from_indices(np.asarray(selected, dtype=np.int64), width)

    return BaselineMethod(
        family="sim_weighted_fps",
        method=_format_method_name("sim_weighted_fps", {"alpha": alpha}),
        params={"alpha": alpha},
        select_points=select_points,
    )


def _build_topk_kmeans(params: dict[str, int | float]) -> BaselineMethod:
    candidate_pool = int(params.get("candidate_pool", 256))

    def select_points(sim_map: np.ndarray, n_points: int, rng: np.random.Generator) -> np.ndarray:
        """Cluster the top-K' pixels into N groups, then keep the highest-similarity pixel in each cluster."""
        del rng
        order, width = _flatten_rank(sim_map)
        pool = order[: min(candidate_pool, order.size)]
        if pool.size <= n_points:
            return _finalize_selection(sim_map, pool.tolist(), n_points)
        pool_coords = _coords_from_indices(pool, width).astype(np.float32)
        pool_scores = sim_map.reshape(-1)[pool]
        centers = _lloyd_kmeans(pool_coords, n_points, pool_scores)
        if centers.size == 0:
            return _finalize_selection(sim_map, pool.tolist(), n_points)
        dists = np.sum((pool_coords[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        assignments = np.argmin(dists, axis=1)
        selected: list[int] = []
        for cluster_id in range(centers.shape[0]):
            members = np.flatnonzero(assignments == cluster_id)
            if members.size == 0:
                continue
            best_member = members[int(np.argmax(pool_scores[members]))]
            selected.append(int(pool[best_member]))
        return _finalize_selection(sim_map, selected, n_points)

    return BaselineMethod(
        family="topk_kmeans",
        method=_format_method_name("topk_kmeans", {"candidate_pool": candidate_pool}),
        params={"candidate_pool": candidate_pool},
        select_points=select_points,
    )


def _build_local_maxima_peaks(params: dict[str, int | float]) -> BaselineMethod:
    min_separation = int(params.get("min_separation", 10))

    def select_points(sim_map: np.ndarray, n_points: int, rng: np.random.Generator) -> np.ndarray:
        """Keep local maxima separated by at least min_separation; backfill from top-k if too few peaks exist."""
        del rng
        if min_separation < 1:
            return build_method("topk_argmax").select_points(
                sim_map, n_points, np.random.default_rng(0)
            )
        peaks = _local_maxima_mask(sim_map, min_separation)
        flat = sim_map.reshape(-1)
        width = sim_map.shape[1]
        peak_indices = np.flatnonzero(peaks.reshape(-1))
        if peak_indices.size == 0:
            return build_method("topk_argmax").select_points(
                sim_map, n_points, np.random.default_rng(0)
            )
        peak_scores = flat[peak_indices]
        peak_order = peak_indices[np.argsort(-peak_scores, kind="stable")]
        peak_coords = _coords_from_indices(peak_order, width)
        selected: list[int] = []
        selected_coords: list[np.ndarray] = []
        radius_sq = float(min_separation * min_separation)
        for index, coord in zip(peak_order.tolist(), peak_coords, strict=True):
            if selected_coords:
                d_sq = np.sum((np.vstack(selected_coords) - coord[None, :]) ** 2, axis=1)
                if bool(np.any(d_sq <= radius_sq)):
                    continue
            selected.append(int(index))
            selected_coords.append(coord)
            if len(selected) == n_points:
                break
        return _finalize_selection(sim_map, selected, n_points)

    return BaselineMethod(
        family="local_maxima_peaks",
        method=_format_method_name("local_maxima_peaks", {"min_separation": min_separation}),
        params={"min_separation": min_separation},
        select_points=select_points,
    )


def _build_connected_component_centers(params: dict[str, int | float]) -> BaselineMethod:
    threshold = float(params.get("threshold", 0.7))

    def select_points(sim_map: np.ndarray, n_points: int, rng: np.random.Generator) -> np.ndarray:
        """Threshold the map, take the argmax inside each connected component, then backfill from top-k."""
        del rng
        mask = sim_map >= threshold
        components = _connected_components(mask)
        flat = sim_map.reshape(-1)
        sim_map.shape[1]
        selected: list[int] = []
        for component in components:
            if component.size == 0:
                continue
            best = component[int(np.argmax(flat[component]))]
            selected.append(int(best))
        selected.sort(key=lambda index: float(flat[index]), reverse=True)
        return _finalize_selection(sim_map, selected, n_points)

    return BaselineMethod(
        family="connected_component_centers",
        method=_format_method_name("connected_component_centers", {"threshold": threshold}),
        params={"threshold": threshold},
        select_points=select_points,
    )


def _build_mean_shift_modes(params: dict[str, int | float]) -> BaselineMethod:
    bandwidth = float(params.get("bandwidth", 12.0))

    def select_points(sim_map: np.ndarray, n_points: int, rng: np.random.Generator) -> np.ndarray:
        """Run mean-shift on a high-similarity candidate pool and backfill from top-k if fewer than N modes remain."""
        del rng
        flat = sim_map.reshape(-1).astype(np.float64)
        width = sim_map.shape[1]
        order, _ = _flatten_rank(sim_map)
        candidate_size = min(max(64 * n_points, 256), order.size)
        candidates = order[:candidate_size]
        coords = _coords_from_indices(candidates, width).astype(np.float64)
        weights = flat[candidates]
        start_count = min(max(4 * n_points, 16), candidates.size)
        starts = coords[:start_count]
        mode_points: list[np.ndarray] = []
        mode_scores: list[float] = []
        bandwidth_sq = max(float(bandwidth * bandwidth), 1e-6)
        for start in starts:
            current = start.copy()
            score = 0.0
            for _ in range(12):
                diff = coords - current[None, :]
                sq = np.sum(diff * diff, axis=1)
                kernel = np.exp(-sq / (2.0 * bandwidth_sq))
                shifted_weights = weights * kernel
                denom = float(shifted_weights.sum())
                if denom <= 0.0:
                    break
                next_point = (shifted_weights[:, None] * coords).sum(axis=0) / denom
                score = denom
                if float(np.linalg.norm(next_point - current)) < 0.5:
                    current = next_point
                    break
                current = next_point
            mode_points.append(current)
            mode_scores.append(score)
        merged_points, merged_scores = _merge_modes(
            mode_points, mode_scores, merge_radius=max(bandwidth / 2.0, 1.0)
        )
        if not merged_points:
            return build_method("topk_argmax").select_points(
                sim_map, n_points, np.random.default_rng(0)
            )
        selected: list[int] = []
        for point in [
            pt for _, pt in sorted(zip(merged_scores, merged_points, strict=True), reverse=True)
        ]:
            nearest = int(np.argmin(np.sum((coords - point[None, :]) ** 2, axis=1)))
            selected.append(int(candidates[nearest]))
        return _finalize_selection(sim_map, selected, n_points)

    return BaselineMethod(
        family="mean_shift_modes",
        method=_format_method_name("mean_shift_modes", {"bandwidth": bandwidth}),
        params={"bandwidth": bandwidth},
        select_points=select_points,
    )


def _build_quantile_spaced(params: dict[str, int | float]) -> BaselineMethod:
    del params

    def select_points(sim_map: np.ndarray, n_points: int, rng: np.random.Generator) -> np.ndarray:
        """Sort pixels by similarity and pick them at evenly spaced ranks; backfill from top-k if ranks collide."""
        del rng
        order, _width = _flatten_rank(sim_map)
        if order.size == 0:
            raise ValueError("sim_map must contain at least one pixel.")
        ranks = np.linspace(0, order.size - 1, num=n_points)
        selected = [int(order[round(rank)]) for rank in ranks]
        selected = _dedupe_preserve_order(selected)
        return _finalize_selection(sim_map, selected, n_points)

    return BaselineMethod(
        family="quantile_spaced",
        method="quantile_spaced",
        params={},
        select_points=select_points,
    )


def _build_sim_weighted_random(params: dict[str, int | float]) -> BaselineMethod:
    temperature = float(params.get("temperature", 0.15))

    def select_points(sim_map: np.ndarray, n_points: int, rng: np.random.Generator) -> np.ndarray:
        """Sample N unique pixels without replacement with probability proportional to softmax(sim / temperature)."""
        flat = sim_map.reshape(-1).astype(np.float64)
        sim_map.shape[1]
        scale = max(temperature, 1e-6)
        shifted = flat / scale
        shifted -= shifted.max()
        probs = np.exp(shifted)
        total = float(probs.sum())
        if total <= 0.0 or not np.isfinite(total):
            return build_method("topk_argmax").select_points(
                sim_map, n_points, np.random.default_rng(0)
            )
        probs /= total
        choice = rng.choice(flat.size, size=min(n_points, flat.size), replace=False, p=probs)
        return _finalize_selection(sim_map, choice.tolist(), n_points)

    return BaselineMethod(
        family="sim_weighted_random",
        method=_format_method_name("sim_weighted_random", {"temperature": temperature}),
        params={"temperature": temperature},
        select_points=select_points,
    )


def _build_dpp(params: dict[str, int | float]) -> BaselineMethod:
    length_scale = float(params.get("length_scale", 16.0))

    def select_points(sim_map: np.ndarray, n_points: int, rng: np.random.Generator) -> np.ndarray:
        """Sample a diverse high-quality subset from a DPP-style kernel; if gains vanish, backfill from top-k."""
        flat = sim_map.reshape(-1).astype(np.float64)
        width = sim_map.shape[1]
        order, _ = _flatten_rank(sim_map)
        pool_size = min(max(32 * n_points, 128), order.size)
        candidates = order[:pool_size]
        coords = _coords_from_indices(candidates, width).astype(np.float64)
        qualities = flat[candidates] + 1e-6
        scale_sq = max(float(length_scale * length_scale), 1e-6)
        diff = coords[:, None, :] - coords[None, :, :]
        dist_sq = np.sum(diff * diff, axis=2)
        similarity = np.exp(-dist_sq / (2.0 * scale_sq))
        kernel = (qualities[:, None] * similarity) * qualities[None, :]
        selected_local: list[int] = []
        for _ in range(min(n_points, candidates.size)):
            gains: list[float] = []
            remaining: list[int]
            if not selected_local:
                remaining = list(range(candidates.size))
                gains = np.clip(np.diag(kernel), 0.0, None).tolist()
            else:
                kernel_ss = kernel[np.ix_(selected_local, selected_local)] + 1e-8 * np.eye(
                    len(selected_local)
                )
                remaining = [idx for idx in range(candidates.size) if idx not in selected_local]
                for idx in remaining:
                    cross = kernel[np.ix_(selected_local, [idx])].reshape(-1)
                    gain = float(kernel[idx, idx] - cross @ np.linalg.solve(kernel_ss, cross))
                    gains.append(max(gain, 0.0))
            if not remaining:
                break
            probs = np.asarray(gains, dtype=np.float64)
            total = float(probs.sum())
            if total <= 0.0 or not np.isfinite(total):
                break
            probs /= total
            best_idx = int(rng.choice(np.asarray(remaining, dtype=np.int64), p=probs))
            selected_local.append(best_idx)
        selected = [int(candidates[idx]) for idx in selected_local]
        return _finalize_selection(sim_map, selected, n_points)

    return BaselineMethod(
        family="dpp",
        method=_format_method_name("dpp", {"length_scale": length_scale}),
        params={"length_scale": length_scale},
        select_points=select_points,
    )


def _build_mask_aware_greedy(params: dict[str, object]) -> BaselineMethod:
    """Marker selector; the evaluator supplies the causal predicted-mask feedback loop."""
    del params

    def select_points(sim_map: np.ndarray, n_points: int, rng: np.random.Generator) -> np.ndarray:
        del rng
        order, width = _flatten_rank(sim_map)
        return _coords_from_indices(order[:n_points], width)

    return BaselineMethod("mask_aware_greedy", "mask_aware_greedy", {}, select_points)


def _build_greedy_bc_policy(params: dict[str, object]) -> BaselineMethod:
    manifest_path = params.get("checkpoint_manifest")
    if manifest_path is None:
        raise ValueError("greedy_bc_policy requires a checkpoint_manifest JSON path.")
    payload = json.loads(Path(str(manifest_path)).read_text(encoding="utf-8"))
    checkpoints_by_budget = payload.get("checkpoints_by_budget", payload)
    selector = BudgetedGreedyBCSelector(
        checkpoints_by_budget={
            int(key): value for key, value in dict(checkpoints_by_budget).items()
        },
        device=str(params.get("device", "cpu")),
    )

    def select_points(sim_map: np.ndarray, n_points: int, rng: np.random.Generator) -> np.ndarray:
        """Load budget-specific greedy-BC checkpoints and emit full-resolution pixel coordinates."""
        return selector.select_points(sim_map, n_points, rng)

    return BaselineMethod(
        family="greedy_bc_policy",
        method=str(params.get("method_name", "greedy_bc_policy")),
        params={
            "checkpoint_manifest": str(manifest_path),
            "device": str(params.get("device", "cpu")),
        },
        select_points=select_points,
    )


def _flatten_rank(sim_map: np.ndarray) -> tuple[np.ndarray, int]:
    if sim_map.ndim != 2:
        raise ValueError(f"sim_map must be 2D, got shape {sim_map.shape}")
    flat = np.asarray(sim_map, dtype=np.float32).reshape(-1)
    order = np.argsort(-flat, kind="stable")
    return order, int(sim_map.shape[1])


def _coords_from_indices(indices: np.ndarray | list[int], width: int) -> np.ndarray:
    idx = np.asarray(indices, dtype=np.int64)
    ys = idx // width
    xs = idx % width
    return np.column_stack([xs, ys]).astype(np.int32)


def _all_pixel_coords(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    yy, xx = np.meshgrid(
        np.arange(height, dtype=np.int32), np.arange(width, dtype=np.int32), indexing="ij"
    )
    return np.column_stack([xx.reshape(-1), yy.reshape(-1)]).astype(np.float32)


def _finalize_selection(sim_map: np.ndarray, selected: list[int], n_points: int) -> np.ndarray:
    width = int(sim_map.shape[1])
    deduped = _dedupe_preserve_order(selected)
    if len(deduped) < n_points:
        order, _ = _flatten_rank(sim_map)
        seen = set(deduped)
        for index in order.tolist():
            if index in seen:
                continue
            deduped.append(int(index))
            seen.add(int(index))
            if len(deduped) == n_points:
                break
    return _coords_from_indices(deduped[:n_points], width)


def _dedupe_preserve_order(values: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for value in values:
        resolved = int(value)
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def _format_method_name(family: str, params: dict[str, int | float]) -> str:
    payload = ",".join(
        f"{key}={_format_param_value(value)}" for key, value in sorted(params.items())
    )
    return f"{family}[{payload}]"


def _format_param_value(value: int | float) -> str:
    if isinstance(value, float):
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return text or "0"
    return str(value)


def _argmax_with_similarity_tiebreak(
    scores: np.ndarray, similarity: np.ndarray, banned: list[int]
) -> int:
    masked = np.asarray(scores, dtype=np.float64).copy()
    masked[np.asarray(banned, dtype=np.int64)] = -np.inf
    best = float(masked.max())
    candidates = np.flatnonzero(np.isclose(masked, best))
    if candidates.size == 1:
        return int(candidates[0])
    return int(candidates[np.argmax(similarity[candidates])])


def _lloyd_kmeans(
    points: np.ndarray, k: int, weights: np.ndarray, iterations: int = 12
) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    count = min(k, points.shape[0])
    centers = _fps_seed_centers(points, count).astype(np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    for _ in range(iterations):
        dists = np.sum((points[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        assignments = np.argmin(dists, axis=1)
        next_centers = centers.copy()
        for cluster_id in range(count):
            members = np.flatnonzero(assignments == cluster_id)
            if members.size == 0:
                continue
            member_weights = weights[members]
            total = float(member_weights.sum())
            if total <= 0.0:
                next_centers[cluster_id] = points[members].mean(axis=0)
            else:
                next_centers[cluster_id] = (points[members] * member_weights[:, None]).sum(
                    axis=0
                ) / total
        if np.allclose(next_centers, centers):
            break
        centers = next_centers
    return centers


def _fps_seed_centers(points: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return np.zeros((0, 2), dtype=np.float32)
    selected = [0]
    min_sqdist = np.sum((points - points[0][None, :]) ** 2, axis=1)
    min_sqdist[0] = -1.0
    for _ in range(1, k):
        next_index = int(np.argmax(min_sqdist))
        selected.append(next_index)
        d_sq = np.sum((points - points[next_index][None, :]) ** 2, axis=1)
        min_sqdist = np.minimum(min_sqdist, d_sq)
        min_sqdist[selected] = -1.0
    return points[np.asarray(selected, dtype=np.int64)]


def _local_maxima_mask(sim_map: np.ndarray, radius: int) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(sim_map, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    pooled = F.max_pool2d(tensor, kernel_size=2 * radius + 1, stride=1, padding=radius)
    return np.isclose(sim_map, pooled.squeeze(0).squeeze(0).numpy(), atol=1e-6)


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    binary = np.asarray(mask, dtype=bool)
    height, width = binary.shape
    visited = np.zeros_like(binary, dtype=bool)
    components: list[np.ndarray] = []
    for y in range(height):
        for x in range(width):
            if not binary[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            members: list[int] = []
            while stack:
                cy, cx = stack.pop()
                members.append(cy * width + cx)
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if not binary[ny, nx] or visited[ny, nx]:
                            continue
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            components.append(np.asarray(members, dtype=np.int64))
    return components


def _merge_modes(
    points: list[np.ndarray], scores: list[float], merge_radius: float
) -> tuple[list[np.ndarray], list[float]]:
    merged_points: list[np.ndarray] = []
    merged_scores: list[float] = []
    radius_sq = float(merge_radius * merge_radius)
    for point, score in sorted(
        zip(points, scores, strict=True), key=lambda item: item[1], reverse=True
    ):
        found = False
        for index, existing in enumerate(merged_points):
            if float(np.sum((existing - point) ** 2)) <= radius_sq:
                if score > merged_scores[index]:
                    merged_points[index] = point
                    merged_scores[index] = score
                found = True
                break
        if not found:
            merged_points.append(point)
            merged_scores.append(score)
    return merged_points, merged_scores
