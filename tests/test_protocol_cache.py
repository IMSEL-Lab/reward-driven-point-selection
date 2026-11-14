from __future__ import annotations

import random
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from PIL import Image

from sam_rl.baseline_eval import load_dino_model, load_sam_predictor, predict_mask_from_points
from sam_rl.episode_manifest import EpisodeSpec
from sam_rl.protocol import multimask_for_prefix, select_sam_mask
from sam_rl.sam_cache import SamImageEmbeddingCache


class FakePredictor:
    device = "cpu"

    def __init__(self) -> None:
        self.original_size = (12, 16)
        self.input_size = (12, 16)
        self.features = torch.ones((1, 2, 3, 4), dtype=torch.float32)
        self.set_image_calls = 0

    def set_image(self, image, image_format="RGB") -> None:
        self.set_image_calls += 1
        self.image = np.asarray(image)
        self.features = torch.full((1, 2, 3, 4), float(self.image.sum() % 23))

    def get_image_embedding(self):
        return self.features


class FakePointPredictor:
    device = "cpu"

    def __init__(self) -> None:
        self.multimask_requests: list[bool] = []

    def predict(self, *, point_coords, point_labels, multimask_output):
        self.multimask_requests.append(bool(multimask_output))
        masks = np.asarray([[[0, 0]], [[1, 1]], [[1, 0]]], dtype=np.uint8)
        scores = np.asarray([0.1, 0.9, 0.2], dtype=np.float32)
        return masks, scores, None


def _episode(root: Path) -> EpisodeSpec:
    paths = []
    for name, color in (
        ("support.png", (10, 20, 30)),
        ("support_mask.png", 255),
        ("query.png", (40, 50, 60)),
        ("query_mask.png", 255),
    ):
        path = root / name
        mode = "L" if isinstance(color, int) else "RGB"
        Image.new(mode, (16, 12), color).save(path)
        paths.append(path)
    return EpisodeSpec(
        dataset="tiny",
        split="test",
        class_name="item",
        support_id="s0",
        query_id="q0",
        support_image_path=paths[0],
        support_mask_path=paths[1],
        query_image_path=paths[2],
        query_mask_path=paths[3],
    )


def test_decoder_policies_and_selection_match_shared_contract() -> None:
    assert not multimask_for_prefix("single", 0)
    assert not multimask_for_prefix("single", 1)
    assert not multimask_for_prefix("legacy_multimask_first", 0)
    assert multimask_for_prefix("legacy_multimask_first", 1)
    masks = np.array([[[0, 0]], [[1, 1]], [[1, 0]]], dtype=np.uint8)
    np.testing.assert_array_equal(select_sam_mask(masks, [0.1, 0.9, 0.2], True), [[1, 1]])
    np.testing.assert_array_equal(select_sam_mask(masks, [0.1, 0.9, 0.2], False), [[0, 0]])

    predictor = FakePointPredictor()
    first_single = predict_mask_from_points(
        predictor, np.asarray([[3, 4]]), decoder_policy="single"
    )
    first_legacy = predict_mask_from_points(
        predictor, np.asarray([[3, 4]]), decoder_policy="legacy_multimask_first"
    )
    later_legacy = predict_mask_from_points(
        predictor, np.asarray([[3, 4], [8, 9]]), decoder_policy="legacy_multimask_first"
    )
    assert predictor.multimask_requests == [False, True, False]
    np.testing.assert_array_equal(first_single, [[0, 0]])
    np.testing.assert_array_equal(first_legacy, [[1, 1]])
    np.testing.assert_array_equal(later_legacy, [[0, 0]])


def test_embedding_cache_rebuilds_corrupt_and_invalidated_entries(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    checkpoint = tmp_path / "sam.pth"
    checkpoint.write_bytes(b"checkpoint-a")
    cache = SamImageEmbeddingCache(tmp_path / "cache", FakePredictor(), checkpoint, "vit_b", 16)
    first = cache.ensure(episode)
    with np.load(first, allow_pickle=False) as payload:
        first_embedding = payload["embedding"].copy()

    first.write_bytes(b"interrupted")
    assert cache.ensure(episode) == first
    with np.load(first, allow_pickle=False) as payload:
        assert payload["embedding"].shape == first_embedding.shape

    Image.new("RGB", (16, 12), (200, 50, 60)).save(episode.query_image_path)
    changed_source = cache.ensure(episode)
    assert changed_source != first

    checkpoint.write_bytes(b"checkpoint-b")
    changed_checkpoint = cache.ensure(episode)
    assert changed_checkpoint != changed_source


def test_concurrent_embedding_cache_writes_publish_complete_npz(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    checkpoint = tmp_path / "sam.pth"
    checkpoint.write_bytes(b"checkpoint")

    def ensure_one(_: int) -> Path:
        cache = SamImageEmbeddingCache(tmp_path / "cache", FakePredictor(), checkpoint, "vit_b", 16)
        return cache.ensure(episode)

    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(ensure_one, range(8)))
    assert len(set(paths)) == 1
    with np.load(paths[0], allow_pickle=False) as payload:
        assert payload["embedding"].shape == (1, 2, 3, 4)
        assert np.isfinite(payload["embedding"]).all()


def test_fp32_embedding_cache_does_not_quantize_decoder_inputs(tmp_path: Path) -> None:
    class FractionalPredictor(FakePredictor):
        def set_image(self, image, image_format="RGB") -> None:
            super().set_image(image, image_format)
            self.features.fill_(1.0001)

    episode = _episode(tmp_path)
    checkpoint = tmp_path / "sam.pth"
    checkpoint.write_bytes(b"checkpoint")
    predictor = FractionalPredictor()
    cache = SamImageEmbeddingCache(tmp_path / "cache", predictor, checkpoint, "vit_b", 16)
    restored = cache.load(episode, device="cpu").embedding
    torch.testing.assert_close(restored, predictor.get_image_embedding(), rtol=0, atol=0)
    assert restored.dtype == torch.float32
    assert not torch.equal(restored, restored.half().float())


def test_bounded_cpu_cache_returns_owned_fp32_records_and_respects_capacity(
    tmp_path: Path,
) -> None:
    episode_a = _episode(tmp_path)
    episode_b = replace(episode_a, query_id="q1")
    checkpoint = tmp_path / "sam.pth"
    checkpoint.write_bytes(b"checkpoint")
    predictor = FakePredictor()
    cache = SamImageEmbeddingCache(
        tmp_path / "cache", predictor, checkpoint, "vit_b", 16, cpu_cache_max_bytes=100
    )

    original = cache.load(episode_a, device="cpu").embedding
    original_copy = original.clone()
    original.zero_()
    restored = cache.load(episode_a, device="cpu").embedding
    torch.testing.assert_close(restored, original_copy, rtol=0, atol=0)
    assert cache._cpu_cache_bytes == 96

    cache.load(episode_b, device="cpu")
    assert cache._cpu_cache_bytes <= cache.cpu_cache_max_bytes
    assert len(cache._cpu_cache) == 1
    # The first item was evicted from RAM but reloaded from its complete disk cache.
    cache.load(episode_a, device="cpu")
    assert predictor.set_image_calls == 2


def test_ensure_warms_cpu_cache_for_following_load(tmp_path: Path) -> None:
    episode = _episode(tmp_path)
    checkpoint = tmp_path / "sam.pth"
    checkpoint.write_bytes(b"checkpoint")
    predictor = FakePredictor()
    cache = SamImageEmbeddingCache(
        tmp_path / "cache", predictor, checkpoint, "vit_b", 16, cpu_cache_max_bytes=100
    )
    cache.ensure(episode)
    calls_after_ensure = predictor.set_image_calls
    record = cache.load(episode, device="cpu")
    assert predictor.set_image_calls == calls_after_ensure == 1
    assert record.embedding.dtype == torch.float32


def test_load_reads_valid_npz_once_without_cpu_cache(tmp_path: Path, monkeypatch) -> None:
    episode = _episode(tmp_path)
    checkpoint = tmp_path / "sam.pth"
    checkpoint.write_bytes(b"checkpoint")
    cache = SamImageEmbeddingCache(tmp_path / "cache", FakePredictor(), checkpoint, "vit_b", 16)
    cache.ensure(episode)

    original_np_load = np.load
    reads = 0

    def counted_np_load(*args, **kwargs):
        nonlocal reads
        reads += 1
        return original_np_load(*args, **kwargs)

    monkeypatch.setattr(np, "load", counted_np_load)
    record = cache.load(episode, device="cpu")
    assert record.embedding.shape == (1, 2, 3, 4)
    assert reads == 1


class RandomInitModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(3))
        self.python_draw = random.random()
        self.numpy_draw = float(np.random.random())


def _capture_global_rng_states():
    numpy_state = cast(Any, np.random.get_state())
    return random.getstate(), numpy_state, torch.get_rng_state().clone()


def _assert_global_rng_states_equal(before) -> None:
    python_state, numpy_state, torch_state = before
    after_numpy_state = cast(Any, np.random.get_state())
    assert random.getstate() == python_state
    assert after_numpy_state[0] == numpy_state[0]
    np.testing.assert_array_equal(after_numpy_state[1], numpy_state[1])
    assert after_numpy_state[2:] == numpy_state[2:]
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)


def test_dino_loader_preserves_global_rng_streams(tmp_path: Path, monkeypatch) -> None:
    expected = RandomInitModule().state_dict()
    checkpoint = tmp_path / "dino.pth"
    torch.save(expected, checkpoint)

    def fake_hub_load(*args, **kwargs):
        del args, kwargs
        return RandomInitModule()

    monkeypatch.setattr(torch.hub, "load", fake_hub_load)
    random.seed(51)
    np.random.seed(51)
    torch.manual_seed(51)
    state_before = _capture_global_rng_states()
    model = load_dino_model(checkpoint, device="cpu")
    _assert_global_rng_states_equal(state_before)
    assert not model.training
    torch.testing.assert_close(model.weight, expected["weight"], rtol=0, atol=0)


def test_sam_loader_preserves_global_rng_streams(tmp_path: Path, monkeypatch) -> None:
    class FakeSamPredictor:
        def __init__(self, model) -> None:
            self.model = model

    def fake_registry_factory(*, checkpoint: str):
        assert checkpoint.endswith("sam.pth")
        return RandomInitModule()

    fake_segment_anything = cast(Any, types.ModuleType("segment_anything"))
    fake_segment_anything.SamPredictor = FakeSamPredictor
    fake_segment_anything.sam_model_registry = {"vit_b": fake_registry_factory}
    monkeypatch.setitem(sys.modules, "segment_anything", fake_segment_anything)

    random.seed(91)
    np.random.seed(91)
    torch.manual_seed(91)
    state_before = _capture_global_rng_states()
    predictor = load_sam_predictor(tmp_path / "sam.pth", "vit_b", device="cpu")
    _assert_global_rng_states_equal(state_before)
    assert isinstance(predictor, FakeSamPredictor)
    assert isinstance(predictor.model, RandomInitModule)
