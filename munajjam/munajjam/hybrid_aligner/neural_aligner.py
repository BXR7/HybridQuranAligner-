"""Capability-gated Zipformer reference alignment boundary."""
from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from munajjam.exceptions import InvalidProviderOutputError, ModelUnavailableError
from munajjam.hybrid_aligner.types import AudioBuffer, BreathGroup, PhonemeEmission


@dataclass(frozen=True, slots=True)
class ZipformerEvidence:
    repository: str
    revision: str
    approved: bool
    vocabulary_size: int = 251
    blank_id: int = 250
    sample_rate: int = 16_000
    feature_kind: str = "kaldi-fbank"
    token_table_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.repository != "Quran-Lab/zipformer_p-arabic-v3":
            raise ValueError("unsupported Zipformer repository")
        if len(self.revision) != 40 or any(c not in "0123456789abcdef" for c in self.revision.lower()):
            raise ValueError("revision must be an immutable git SHA")
        if not self.approved:
            raise ValueError("gated Zipformer evidence is not approved")
        for name, value in (("vocabulary_size", self.vocabulary_size), ("blank_id", self.blank_id), ("sample_rate", self.sample_rate)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.blank_id < self.vocabulary_size:
            raise ValueError("blank_id must be inside vocabulary")
        if self.feature_kind != "kaldi-fbank":
            raise ValueError("unsupported Zipformer feature contract")


class ZipformerBackend(Protocol):
    def __call__(self, audio: AudioBuffer, group: BreathGroup) -> list[PhonemeEmission]: ...


class ZipformerNeuralAligner:
    """Only exposes a provider after operator-supplied evidence is validated."""

    def __init__(self, evidence: ZipformerEvidence | None = None, *, backend_factory: Callable[[], ZipformerBackend] | None = None) -> None:
        self.evidence = evidence
        self._backend_factory = backend_factory
        self._backend: ZipformerBackend | None = None
        self._lock = threading.Lock()

    def _get_backend(self) -> ZipformerBackend:
        if self.evidence is None or self._backend_factory is None:
            raise ModelUnavailableError("Zipformer is gated or has no verified runtime evidence")
        if self._backend is None:
            with self._lock:
                if self._backend is None:
                    try:
                        self._backend = self._backend_factory()
                    except Exception as exc:
                        raise ModelUnavailableError("failed to initialize Zipformer backend") from exc
        return self._backend

    def align(self, audio: AudioBuffer, group: BreathGroup) -> tuple[PhonemeEmission, ...]:
        self._get_backend()
        if self.evidence is None or audio.sample_rate != self.evidence.sample_rate:
            raise InvalidProviderOutputError("audio sample rate does not match Zipformer evidence")
        emissions = self._backend(audio, group)  # type: ignore[misc]
        previous = 0
        result: list[PhonemeEmission] = []
        for emission in emissions:
            if not isinstance(emission, PhonemeEmission) or emission.token_id >= self.evidence.vocabulary_size:
                raise InvalidProviderOutputError("Zipformer returned invalid token evidence")
            if emission.start_frame < previous:
                raise InvalidProviderOutputError("Zipformer emissions are not monotonic")
            result.append(emission)
            previous = emission.end_frame
        return tuple(result)

    def close(self) -> None:
        backend, self._backend = self._backend, None
        if backend is not None and hasattr(backend, "close"):
            backend.close()  # type: ignore[attr-defined]
