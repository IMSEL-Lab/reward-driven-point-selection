from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sam_rl.baseline_eval import FewShotEpisode, load_episode_arrays
from sam_rl.protocol import autocast_context, resolve_precision
from sam_rl.provenance import atomic_write_npz, cache_metadata, cache_path, file_fingerprint


@dataclass(frozen=True)
class SamImageEmbeddingRecord:
    embedding: torch.Tensor
    input_size: tuple[int, int]
    original_size: tuple[int, int]
    image_id: str


@dataclass(frozen=True)
class SamImageEmbeddingBatch:
    embeddings: torch.Tensor
    input_size: tuple[int, int]
    original_size: tuple[int, int]
    image_ids: tuple[str, ...]


class SamImageEmbeddingCache:
    def __init__(
        self,
        cache_dir: str | Path,
        predictor,
        sam_checkpoint: str | Path,
        sam_model_type: str,
        image_size: int,
        precision: str = "fp32",
        cpu_cache_max_bytes: int = 0,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.predictor = predictor
        self.sam_checkpoint = Path(sam_checkpoint)
        self.sam_model_type = str(sam_model_type)
        self.image_size = int(image_size)
        self.device = str(getattr(predictor, "device", "cpu"))
        self.precision = resolve_precision(self.device, precision)
        self.cpu_cache_max_bytes = int(cpu_cache_max_bytes)
        if self.cpu_cache_max_bytes < 0:
            raise ValueError("cpu_cache_max_bytes must be non-negative.")
        # Entries own immutable-by-convention FP32 CPU tensors. `load` returns a
        # clone for CPU callers and a device copy for accelerators, so mutations
        # cannot poison this shared cache.
        self._cpu_cache: OrderedDict[str, SamImageEmbeddingRecord] = OrderedDict()
        self._cpu_cache_bytes = 0

    def path_for(self, episode: FewShotEpisode) -> Path:
        return cache_path(self.cache_dir, "sam_embedding", episode, self._metadata(episode))

    def _metadata(self, episode: FewShotEpisode) -> dict:
        return cache_metadata(
            kind="sam_embedding",
            episode=episode,
            config={
                "sam_checkpoint": file_fingerprint(self.sam_checkpoint),
                "sam_model_type": self.sam_model_type,
                "image_size": self.image_size,
                "preprocessing": "load_episode_arrays_rgb_resize",
                "precision": self.precision,
                "embedding_storage_dtype": "float32",
            },
            schema_version=3,
        )

    def ensure(self, episode: FewShotEpisode) -> Path:
        path, _ = self._get_or_create_record(episode, validate_disk=True)
        return path

    def load(self, episode: FewShotEpisode, device: str | torch.device) -> SamImageEmbeddingRecord:
        _, record = self._get_or_create_record(episode)
        requested_device = torch.device(device)
        embedding = record.embedding.to(requested_device)
        if requested_device.type == "cpu":
            embedding = embedding.clone()
        return SamImageEmbeddingRecord(
            embedding=embedding,
            input_size=record.input_size,
            original_size=record.original_size,
            image_id=episode.image_id,
        )

    def _get_or_create_record(
        self, episode: FewShotEpisode, *, validate_disk: bool = False
    ) -> tuple[Path, SamImageEmbeddingRecord]:
        metadata = self._metadata(episode)
        key = str(metadata["cache_key"])
        path = cache_path(self.cache_dir, "sam_embedding", episode, metadata)
        cached = self._cpu_cache_get(key)
        if cached is not None:
            if validate_disk and path.is_file():
                valid_disk = self._read_valid_record(path, metadata, episode.image_id)
                if valid_disk is None:
                    self._publish_record(path, metadata, cached)
            elif not path.exists():
                self._publish_record(path, metadata, cached)
            return path, cached

        if path.is_file():
            record = self._read_valid_record(path, metadata, episode.image_id)
            if record is not None:
                return path, self._remember(key, record)
            path.unlink(missing_ok=True)

        _, _, query_rgb, _ = load_episode_arrays(episode, image_size=self.image_size)
        with torch.inference_mode(), autocast_context(self.device, self.precision):
            self.predictor.set_image(query_rgb, image_format="RGB")
        embedding = (
            self.predictor.get_image_embedding()
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .clone()
            .contiguous()
        )
        record = SamImageEmbeddingRecord(
            embedding=embedding,
            input_size=(int(self.predictor.input_size[0]), int(self.predictor.input_size[1])),
            original_size=(
                int(self.predictor.original_size[0]),
                int(self.predictor.original_size[1]),
            ),
            image_id=episode.image_id,
        )
        self._publish_record(path, metadata, record)
        return path, self._remember(key, record)

    def _read_valid_record(
        self, path: Path, expected_metadata: dict[str, Any], image_id: str
    ) -> SamImageEmbeddingRecord | None:
        try:
            with np.load(path, allow_pickle=False) as payload:
                metadata = json.loads(str(payload["metadata_json"].item()))
                embedding_array = np.asarray(payload["embedding"])
                original_values = np.asarray(payload["original_size"], dtype=np.int32).tolist()
                input_values = np.asarray(payload["input_size"], dtype=np.int32).tolist()
            valid = (
                all(metadata.get(key) == value for key, value in expected_metadata.items())
                and embedding_array.dtype == np.float32
                and embedding_array.ndim == 4
                and embedding_array.size > 0
                and np.isfinite(embedding_array).all()
                and len(original_values) == 2
                and len(input_values) == 2
                and min(map(int, original_values)) > 0
                and min(map(int, input_values)) > 0
                and metadata.get("embedding_shape") == list(embedding_array.shape)
                and metadata.get("original_size") == list(map(int, original_values))
                and metadata.get("input_size") == list(map(int, input_values))
            )
            if not valid:
                return None
            # Copy out of NumPy's NPZ buffer. This owned tensor is the LRU's
            # canonical record and is never returned directly to CPU callers.
            embedding = torch.from_numpy(embedding_array.copy()).contiguous()
            return SamImageEmbeddingRecord(
                embedding=embedding,
                input_size=(int(input_values[0]), int(input_values[1])),
                original_size=(int(original_values[0]), int(original_values[1])),
                image_id=image_id,
            )
        except (OSError, ValueError, KeyError, json.JSONDecodeError, TypeError):
            return None

    def _publish_record(
        self, path: Path, metadata: dict[str, Any], record: SamImageEmbeddingRecord
    ) -> None:
        published_metadata = dict(metadata)
        published_metadata["original_size"] = list(record.original_size)
        published_metadata["input_size"] = list(record.input_size)
        published_metadata["embedding_shape"] = list(record.embedding.shape)
        atomic_write_npz(
            path,
            embedding=record.embedding.numpy(),
            original_size=np.asarray(record.original_size, dtype=np.int32),
            input_size=np.asarray(record.input_size, dtype=np.int32),
            metadata_json=np.asarray(json.dumps(published_metadata, sort_keys=True)),
        )

    def _cpu_cache_get(self, key: str) -> SamImageEmbeddingRecord | None:
        record = self._cpu_cache.pop(key, None)
        if record is not None:
            self._cpu_cache[key] = record
        return record

    def _remember(self, key: str, record: SamImageEmbeddingRecord) -> SamImageEmbeddingRecord:
        size_bytes = record.embedding.numel() * record.embedding.element_size()
        if self.cpu_cache_max_bytes <= 0 or size_bytes > self.cpu_cache_max_bytes:
            return record
        previous = self._cpu_cache.pop(key, None)
        if previous is not None:
            self._cpu_cache_bytes -= previous.embedding.numel() * previous.embedding.element_size()
        self._cpu_cache[key] = record
        self._cpu_cache_bytes += size_bytes
        while self._cpu_cache_bytes > self.cpu_cache_max_bytes and self._cpu_cache:
            _, evicted = self._cpu_cache.popitem(last=False)
            self._cpu_cache_bytes -= evicted.embedding.numel() * evicted.embedding.element_size()
        return record

    def load_batch(
        self, episodes: Sequence[FewShotEpisode], device: str | torch.device
    ) -> SamImageEmbeddingBatch:
        records = [self.load(episode, device=device) for episode in episodes]
        if not records:
            raise ValueError("Cannot load an empty SAM embedding batch.")
        input_size = records[0].input_size
        original_size = records[0].original_size
        for record in records[1:]:
            if record.input_size != input_size or record.original_size != original_size:
                raise ValueError(
                    "All cached SAM embeddings in a batch must share input_size/original_size."
                )
        return SamImageEmbeddingBatch(
            embeddings=torch.cat([record.embedding for record in records], dim=0),
            input_size=input_size,
            original_size=original_size,
            image_ids=tuple(record.image_id for record in records),
        )


def ensure_sam_embedding_caches(
    episodes: Sequence[FewShotEpisode],
    embedding_cache: SamImageEmbeddingCache,
    log_every: int = 100,
) -> None:
    total = 0
    for episode in episodes:
        embedding_cache.ensure(episode)
        total += 1
        if total == 1 or total % max(1, int(log_every)) == 0:
            print(f"[sam-embed] cached {total} episodes (latest: {episode.image_id})", flush=True)


def set_predictor_image_embedding(predictor, record: SamImageEmbeddingRecord) -> None:
    predictor.reset_image()
    predictor.features = record.embedding.to(device=predictor.device, dtype=torch.float32)
    predictor.original_size = tuple(int(value) for value in record.original_size)
    predictor.input_size = tuple(int(value) for value in record.input_size)
    predictor.is_image_set = True


def set_predictor_image_embedding_batch(predictor, batch: SamImageEmbeddingBatch) -> None:
    predictor.reset_image()
    predictor.features = batch.embeddings.to(device=predictor.device, dtype=torch.float32)
    predictor.original_size = tuple(int(value) for value in batch.original_size)
    predictor.input_size = tuple(int(value) for value in batch.input_size)
    predictor.is_image_set = True


def safe_slug(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))
    return cleaned.strip("_").lower()
