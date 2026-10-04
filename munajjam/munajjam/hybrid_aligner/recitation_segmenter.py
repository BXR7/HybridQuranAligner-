"""Lazy, fail-closed breath segmentation contracts and adapter."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from munajjam.exceptions import (
    InvalidProviderOutputError,
    ModelUnavailableError,
    NoBreathGroupsError,
    SegmenterError,
)
from munajjam.hybrid_aligner.types import AudioBuffer, BreathGroup


class AudioLoader(Protocol):
    def __call__(self, audio: Any, sample_rate: int) -> AudioBuffer: ...


class SegmenterBackend(Protocol):
    def __call__(self, audio: AudioBuffer) -> list[tuple[float, float, float | None]]: ...


RECITATION_SEGMENTER_REPOSITORY = "obadx/recitation-segmenter-v2"
RECITATION_SEGMENTER_REVISION = "5ee90364e7090ea6eb9dffe80353bed06996a196"


@dataclass(frozen=True, slots=True)
class _SegmenterRuntime:
    torch: Any
    model: Any
    processor: Any
    device: Any
    dtype: Any
    segment_recitations: Callable[..., Any]
    clean_speech_intervals: Callable[..., Any]


class TransformersRecitationSegmenterBackend:
    """Pinned adapter for the author's ``recitations-segmenter`` API.

    Dependencies and weights are loaded only on the first inference call. The
    model card's public revision is immutable; the library's documented
    ``segment_recitations``/``clean_speech_intervals`` API performs frame
    decoding and interval post-processing. No energy/VAD fallback is used.
    """

    def __init__(
        self,
        *,
        model_id: str = RECITATION_SEGMENTER_REPOSITORY,
        revision: str = RECITATION_SEGMENTER_REVISION,
        device: str | None = None,
        cache_dir: str | None = None,
        runtime_loader: Callable[[], _SegmenterRuntime] | None = None,
        min_silence_duration_ms: int = 30,
        min_speech_duration_ms: int = 30,
        pad_duration_ms: int = 30,
    ) -> None:
        if model_id != RECITATION_SEGMENTER_REPOSITORY:
            raise ValueError("only the pinned Quran recitation segmenter is supported")
        if revision != RECITATION_SEGMENTER_REVISION:
            raise ValueError("recitation segmenter revision must be the pinned immutable SHA")
        if device not in (None, "cpu", "cuda"):
            raise ValueError("segmenter device must be cpu, cuda, or None")
        if min(min_silence_duration_ms, min_speech_duration_ms, pad_duration_ms) < 0:
            raise ValueError("segmenter cleanup durations must be non-negative")
        self.model_id = model_id
        self.revision = revision
        self.device = device
        self.cache_dir = cache_dir
        self.min_silence_duration_ms = min_silence_duration_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.pad_duration_ms = pad_duration_ms
        self._runtime_loader = runtime_loader or self._load_runtime
        self._runtime: _SegmenterRuntime | None = None
        self._lock = threading.Lock()

    def _load_runtime(self) -> _SegmenterRuntime:
        try:
            import torch
            from recitations_segmenter import clean_speech_intervals, segment_recitations
            from transformers import AutoFeatureExtractor, AutoModelForAudioFrameClassification
        except Exception as exc:
            raise ModelUnavailableError(
                "install the pinned munajjam[segmenter] extra to use recitation-segmenter-v2"
            ) from exc
        try:
            if self.device == "cuda" and not torch.cuda.is_available():
                raise ModelUnavailableError(
                    "CUDA was requested for recitation segmentation but is unavailable"
                )
            device = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
            dtype = torch.float32
            load_args = {"revision": self.revision}
            if self.cache_dir is not None:
                load_args["cache_dir"] = self.cache_dir
            processor = AutoFeatureExtractor.from_pretrained(self.model_id, **load_args)
            model = AutoModelForAudioFrameClassification.from_pretrained(self.model_id, **load_args)
            model.to(device=device, dtype=dtype)
            model.eval()
        except Exception as exc:
            raise ModelUnavailableError(
                f"failed to load pinned segmenter {self.model_id}@{self.revision}"
            ) from exc
        return _SegmenterRuntime(
            torch, model, processor, device, dtype, segment_recitations, clean_speech_intervals
        )

    def _get_runtime(self) -> _SegmenterRuntime:
        if self._runtime is None:
            with self._lock:
                if self._runtime is None:
                    self._runtime = self._runtime_loader()
        return self._runtime

    @staticmethod
    def _as_intervals(value: Any) -> list[tuple[float, float, None]]:
        if hasattr(value, "detach"):
            value = value.detach().cpu().tolist()
        elif hasattr(value, "tolist"):
            value = value.tolist()
        if not isinstance(value, (list, tuple)):
            raise InvalidProviderOutputError("segmenter library returned malformed intervals")
        intervals: list[tuple[float, float, None]] = []
        for interval in value:
            if hasattr(interval, "tolist"):
                interval = interval.tolist()
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                raise InvalidProviderOutputError("segmenter library returned malformed intervals")
            intervals.append((interval[0], interval[1], None))
        return intervals

    def __call__(self, audio: AudioBuffer) -> list[tuple[float, float, float | None]]:
        if audio.sample_rate != 16_000:
            raise InvalidProviderOutputError("recitation-segmenter-v2 requires 16 kHz audio")
        runtime = self._get_runtime()
        try:
            waveform = runtime.torch.as_tensor(
                np.asarray(audio.samples, dtype=np.float32), dtype=runtime.torch.float32
            )
            outputs = runtime.segment_recitations(
                [waveform],
                runtime.model,
                runtime.processor,
                batch_size=1,
                device=runtime.device,
                dtype=runtime.dtype,
                sample_rate=16_000,
                max_duration_ms=19_995,
            )
            if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
                raise InvalidProviderOutputError(
                    "segmenter library must return one output for one waveform"
                )
            output = outputs[0]
            speech_intervals = getattr(output, "speech_intervals", None)
            is_complete = getattr(output, "is_complete", None)
            if speech_intervals is None or not isinstance(is_complete, (bool, np.bool_)):
                raise InvalidProviderOutputError(
                    "segmenter output is missing interval/completion data"
                )
            cleaned = runtime.clean_speech_intervals(
                speech_intervals,
                bool(is_complete),
                min_silence_duration_ms=self.min_silence_duration_ms,
                min_speech_duration_ms=self.min_speech_duration_ms,
                pad_duration_ms=self.pad_duration_ms,
                return_seconds=True,
            )
            return self._as_intervals(getattr(cleaned, "clean_speech_intervals", None))
        except (InvalidProviderOutputError, ModelUnavailableError):
            raise
        except Exception as exc:
            raise SegmenterError("recitation segmenter inference failed") from exc

    def close(self) -> None:
        runtime, self._runtime = self._runtime, None
        if runtime is not None:
            model = getattr(runtime, "model", None)
            if model is not None and hasattr(model, "to"):
                try:
                    model.to("cpu")
                except Exception:
                    pass


class EnergyBreathSegmenter:
    """Deterministic RMS-energy segmentation for real recitation audio.

    This is intentionally a signal-processing backend, not a neural-model
    claim. It finds contiguous voiced regions separated by minimum pauses.
    """

    def __init__(
        self, *, frame_ms: float = 25.0, min_pause_ms: float = 250.0, threshold_db: float = -25.0
    ) -> None:
        self.frame_ms = frame_ms
        self.min_pause_ms = min_pause_ms
        self.threshold_db = threshold_db

    def __call__(self, audio: AudioBuffer) -> list[tuple[float, float, float | None]]:
        frame = max(1, round(audio.sample_rate * self.frame_ms / 1000.0))
        samples = np.asarray(audio.samples, dtype=np.float32)
        count = max(1, int(np.ceil(len(samples) / frame)))
        rms = np.array(
            [
                float(
                    np.sqrt(
                        np.mean(np.square(samples[i * frame : min(len(samples), (i + 1) * frame)]))
                    )
                )
                for i in range(count)
            ]
        )
        db = 20.0 * np.log10(np.maximum(rms, 1e-8))
        voiced = db >= self.threshold_db
        pause_frames = max(1, round(self.min_pause_ms / self.frame_ms))
        boundaries = np.flatnonzero(voiced)
        if boundaries.size == 0:
            return []
        groups: list[tuple[float, float, float | None]] = []
        start = int(boundaries[0])
        previous = start
        for index in boundaries[1:]:
            index = int(index)
            if index - previous >= pause_frames:
                end = min(len(samples), (previous + 1) * frame)
                groups.append((start * frame / audio.sample_rate, end / audio.sample_rate, None))
                start = index
            previous = index
        end = min(len(samples), (previous + 1) * frame)
        groups.append((start * frame / audio.sample_rate, end / audio.sample_rate, None))
        return groups


@dataclass(frozen=True, slots=True)
class SegmenterConfig:
    sample_rate: int = 16_000
    max_duration_sec: float = 30 * 60
    max_samples: int = 16_000 * 30 * 60
    max_groups: int = 2_000
    min_group_duration_sec: float = 0.02
    max_group_duration_sec: float = 120.0

    def __post_init__(self) -> None:
        values = (
            self.sample_rate,
            self.max_duration_sec,
            self.max_samples,
            self.max_groups,
            self.min_group_duration_sec,
            self.max_group_duration_sec,
        )
        if any(isinstance(v, bool) or float(v) <= 0 for v in values):
            raise ValueError("segmenter limits must be positive")
        if self.min_group_duration_sec > self.max_group_duration_sec:
            raise ValueError("minimum group duration exceeds maximum")


class QuranRecitationSegmenter:
    """Segment speech intervals without importing or loading ML runtimes at construction."""

    def __init__(
        self,
        config: SegmenterConfig | None = None,
        *,
        loader: AudioLoader | None = None,
        backend_factory: Callable[[], SegmenterBackend] | None = None,
    ) -> None:
        self.config = config or SegmenterConfig()
        self._loader = loader or self._default_loader
        self._backend_factory = backend_factory or self._default_backend
        self._backend: SegmenterBackend | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _default_loader(audio: Any, sample_rate: int) -> AudioBuffer:
        if isinstance(audio, AudioBuffer):
            if audio.sample_rate != sample_rate:
                raise SegmenterError("AudioBuffer must already use the configured sample rate")
            return audio
        try:
            array = np.asarray(audio, dtype=np.float32)
        except Exception as exc:
            raise SegmenterError("audio must be an AudioBuffer or numeric mono samples") from exc
        return AudioBuffer(array, sample_rate)

    @staticmethod
    def _default_backend() -> SegmenterBackend:
        return TransformersRecitationSegmenterBackend()

    def _get_backend(self) -> SegmenterBackend:
        if self._backend is None:
            with self._lock:
                if self._backend is None:
                    try:
                        self._backend = self._backend_factory()
                    except ModelUnavailableError:
                        raise
                    except Exception as exc:
                        raise ModelUnavailableError(
                            "failed to initialize recitation segmenter"
                        ) from exc
        return self._backend

    def segment(self, audio: Any) -> tuple[BreathGroup, ...]:
        try:
            buffer = self._loader(audio, self.config.sample_rate)
        except SegmenterError:
            raise
        except Exception as exc:
            raise SegmenterError("failed to load audio") from exc
        if not np.all(np.isfinite(buffer.samples)):
            raise SegmenterError("audio contains non-finite samples")
        if (
            len(buffer.samples) > self.config.max_samples
            or buffer.duration > self.config.max_duration_sec
        ):
            raise SegmenterError("audio exceeds segmenter resource limits")
        try:
            raw = self._get_backend()(buffer)
        except (SegmenterError, ModelUnavailableError):
            raise
        except Exception as exc:
            raise SegmenterError("segmenter inference failed") from exc
        if not raw:
            raise NoBreathGroupsError("segmenter returned no speech groups")
        groups: list[BreathGroup] = []
        previous_end = 0.0
        if len(raw) > self.config.max_groups:
            raise InvalidProviderOutputError("segmenter returned too many groups")
        for item in raw:
            if not isinstance(item, (tuple, list)) or len(item) not in (2, 3):
                raise InvalidProviderOutputError(
                    "segment interval must contain start, end, and optional score"
                )
            start, end = item[0], item[1]
            score = item[2] if len(item) == 3 else None
            try:
                group = BreathGroup(start, end, score)
            except (TypeError, ValueError) as exc:
                raise InvalidProviderOutputError("segmenter returned an invalid group") from exc
            if group.end > buffer.duration or group.start < previous_end:
                raise InvalidProviderOutputError("groups must be ordered and inside audio")
            if (
                group.end - group.start < self.config.min_group_duration_sec
                or group.end - group.start > self.config.max_group_duration_sec
            ):
                raise InvalidProviderOutputError("group duration violates configured limits")
            groups.append(group)
            previous_end = group.end
        return tuple(groups)

    def close(self) -> None:
        backend, self._backend = self._backend, None
        if backend is not None and hasattr(backend, "close"):
            backend.close()  # type: ignore[attr-defined]
