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
    """Validated operator-supplied evidence for the gated Zipformer model.

    Fields:
        repository: HuggingFace/Git repository ID. Must be
            ``Quran-Lab/zipformer_p-arabic-v3``.
        revision: Immutable git commit SHA (40 hex characters).
        approved: Operator attestation that the artifact is approved for use.
            Must be ``True``.
        vocabulary_size: Number of tokens in the model vocabulary.
        blank_id: Token ID reserved for the CTC/graphical blank.
        sample_rate: Audio sample rate the model expects (Hz).
        feature_kind: Feature extraction contract identifier.
            Must be ``"kaldi-fbank"``.
        token_table_sha256: SHA-256 hex digest of the model ``tokens.txt``.
            Required for non-authoritative deployments; when provided it is
            validated for format and length.
        phoneme_mapping_required: Whether a phoneme-to-token mapping table is
            required by this artifact. Defaults to ``False``.

    Requirements:
        - ``tokens.txt`` must contain exactly ``vocabulary_size`` ``piece id``
          entries.  Entries are parsed by their explicit integer ID rather
          than line position; the artifact includes ``<blank> 250``.
        - When ``phoneme_mapping_required`` is ``True`, a ``phonemes.txt``
          mapping file must accompany ``tokens.txt`` and map each token to its
          IPA phoneme.  The mapping must not cross breath boundaries.
        - The evidence contract does not require a particular device. The pinned ONNX
          backend requests CUDA then CPU through ONNX Runtime's provider fallback.
    """

    repository: str
    revision: str
    approved: bool
    vocabulary_size: int = 251
    blank_id: int = 250
    sample_rate: int = 16_000
    feature_kind: str = "kaldi-fbank"
    token_table_sha256: str | None = None
    phoneme_mapping_required: bool = False

    def __post_init__(self) -> None:
        if self.repository != "Quran-Lab/zipformer_p-arabic-v3":
            raise ValueError("unsupported Zipformer repository")
        if len(self.revision) != 40 or not all(
            c in "0123456789abcdef" for c in self.revision.lower()
        ):
            raise ValueError("revision must be an immutable git SHA")
        if not self.approved:
            raise ValueError("gated Zipformer evidence is not approved")
        for name, value in (
            ("vocabulary_size", self.vocabulary_size),
            ("blank_id", self.blank_id),
            ("sample_rate", self.sample_rate),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.blank_id < self.vocabulary_size:
            raise ValueError("blank_id must be inside vocabulary")
        if self.feature_kind != "kaldi-fbank":
            raise ValueError("unsupported Zipformer feature contract")
        if self.token_table_sha256 is not None:
            sha = self.token_table_sha256.strip().lower()
            if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                raise ValueError("token_table_sha256 must be a 64-char hex digest")
        if not isinstance(self.phoneme_mapping_required, bool):
            raise ValueError("phoneme_mapping_required must be a bool")


class ZipformerBackend(Protocol):
    def __call__(self, audio: AudioBuffer, group: BreathGroup) -> list[PhonemeEmission]: ...


class ZipformerNeuralAligner:
    """Only exposes a provider after operator-supplied evidence is validated."""

    def __init__(
        self,
        evidence: ZipformerEvidence | None = None,
        *,
        backend_factory: Callable[[], ZipformerBackend] | None = None,
    ) -> None:
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
                        raise ModelUnavailableError(
                            "failed to initialize Zipformer backend"
                        ) from exc
        return self._backend

    def align(self, audio: AudioBuffer, group: BreathGroup) -> tuple[PhonemeEmission, ...]:
        self._get_backend()
        if self.evidence is None or audio.sample_rate != self.evidence.sample_rate:
            raise InvalidProviderOutputError("audio sample rate does not match Zipformer evidence")
        emissions = self._backend(audio, group)  # type: ignore[misc]
        previous = 0
        result: list[PhonemeEmission] = []
        for emission in emissions:
            if not isinstance(emission, PhonemeEmission):
                raise InvalidProviderOutputError("Zipformer returned non-PhonemeEmission evidence")
            if emission.token_id < 0 or emission.token_id >= self.evidence.vocabulary_size:
                raise InvalidProviderOutputError("Zipformer token ID outside vocabulary")
            if emission.token_id == self.evidence.blank_id:
                raise InvalidProviderOutputError("Zipformer emission contains blank token")
            if emission.start_frame < 0 or emission.end_frame <= emission.start_frame:
                raise InvalidProviderOutputError("Zipformer emission has invalid frame bounds")
            if emission.start_frame < previous:
                raise InvalidProviderOutputError("Zipformer emissions are not monotonic")
            result.append(emission)
            previous = emission.end_frame
        return tuple(result)

    def close(self) -> None:
        backend, self._backend = self._backend, None
        if backend is not None and hasattr(backend, "close"):
            backend.close()  # type: ignore[attr-defined]
