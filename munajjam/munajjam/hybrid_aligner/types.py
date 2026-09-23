"""Backend-neutral contracts for the opt-in tripartite Quran aligner."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from munajjam.exceptions import AlignmentError


def _finite(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True, slots=True)
class AudioBuffer:
    samples: Any
    sample_rate: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_rate, bool)
            or not isinstance(self.sample_rate, int)
            or self.sample_rate <= 0
        ):
            raise ValueError("sample_rate must be a positive integer")
        if getattr(self.samples, "ndim", 1) != 1:
            raise ValueError("audio samples must be a one-dimensional mono array")
        if len(self.samples) == 0:
            raise ValueError("audio samples cannot be empty")

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sample_rate


@dataclass(frozen=True, slots=True)
class BreathGroup:
    start: float
    end: float
    score: float | None = None

    def __post_init__(self) -> None:
        start, end = _finite(self.start, "start"), _finite(self.end, "end")
        if start < 0 or end <= start:
            raise ValueError("BreathGroup must satisfy 0 <= start < end")
        if self.score is not None and (
            not math.isfinite(float(self.score)) or not 0 <= float(self.score) <= 1
        ):
            raise ValueError("score must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class AlignmentSpan:
    token: str
    start: float
    end: float
    confidence: float | None = None
    provenance: str = "unknown"

    def __post_init__(self) -> None:
        _finite(self.start, "start")
        _finite(self.end, "end")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("alignment span must satisfy 0 <= start < end")
        if not self.token:
            raise ValueError("alignment token cannot be empty")
        if self.confidence is not None and not 0 <= float(self.confidence) <= 1:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class PhonemeEmission:
    token_id: int
    start_frame: int
    end_frame: int
    score: float

    def __post_init__(self) -> None:
        for name, value in (
            ("token_id", self.token_id),
            ("start_frame", self.start_frame),
            ("end_frame", self.end_frame),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.token_id < 0 or self.start_frame < 0 or self.end_frame <= self.start_frame:
            raise ValueError("invalid phoneme emission bounds")
        if not math.isfinite(float(self.score)):
            raise ValueError("score must be finite")


@dataclass(frozen=True, slots=True)
class HybridAlignmentResult:
    breath_groups: tuple[BreathGroup, ...]
    spans: tuple[AlignmentSpan, ...] = ()
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, duration: float) -> None:
        if not math.isfinite(duration) or duration <= 0:
            raise AlignmentError("invalid audio duration")
        previous = 0.0
        for group in self.breath_groups:
            if group.start < previous or group.end > duration:
                raise AlignmentError("breath group outside audio bounds")
            previous = group.end
        previous = 0.0
        for span in self.spans:
            if span.start < previous or span.end > duration:
                raise AlignmentError("alignment span outside audio bounds")
            previous = span.end
