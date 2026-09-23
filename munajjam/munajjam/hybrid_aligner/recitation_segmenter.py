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
        raise ModelUnavailableError(
            "recitation segmenter runtime is optional; inject a backend or install the approved runtime"
        )

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
