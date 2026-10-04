"""
Comprehensive unit tests for the opt-in HybridQuranAligner (Issue #120).

All tests are fully deterministic using injected fakes — no network, no model
downloads, no torch/transformers/whisperx imports.

Real-model smoke tests remain opt-in (see conftest.py markers).
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import textwrap
from unittest.mock import MagicMock

import numpy as np
import pytest

from munajjam.exceptions import (
    AlignmentError,
    InvalidProviderOutputError,
    ModelUnavailableError,
    NoBreathGroupsError,
    SegmenterError,
)
from munajjam.hybrid_aligner import (
    AlignmentSpan,
    AudioBuffer,
    BreathGroup,
    CanonicalQuranReferenceProvider,
    CanonicalReferenceRequest,
    HybridAlignmentResult,
    HybridQuranAligner,
    PhonemeEmission,
    QuranRecitationSegmenter,
    SegmenterConfig,
    TransformersRecitationSegmenterBackend,
    Wav2Vec2Config,
    Wav2Vec2ForcedAligner,
    ZipformerEvidence,
    ZipformerNeuralAligner,
)

HEAVY_MODULES = [
    "torch",
    "transformers",
    "whisperx",
    "onnxruntime",
    "sherpa_onnx",
    "faster_whisper",
]

SAMPLE_RATE = 16_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def audio(duration: float = 2.0) -> AudioBuffer:
    """Deterministic sine-wave AudioBuffer."""
    n = int(duration * SAMPLE_RATE)
    t = np.linspace(0, duration, n, endpoint=False)
    samples = (np.sin(2 * math.pi * 440 * t) * 0.5).astype(np.float32)
    return AudioBuffer(samples=samples, sample_rate=SAMPLE_RATE)


def silent_audio(duration: float = 2.0) -> AudioBuffer:
    n = int(duration * SAMPLE_RATE)
    return AudioBuffer(samples=np.zeros(n, dtype=np.float32), sample_rate=SAMPLE_RATE)


def make_evidence(**kwargs) -> ZipformerEvidence:
    base = {
        "repository": "Quran-Lab/zipformer_p-arabic-v3",
        "revision": "0123456789abcdef0123456789abcdef01234567",
        "approved": True,
        "vocabulary_size": 251,
        "blank_id": 250,
        "sample_rate": 16_000,
        "feature_kind": "kaldi-fbank",
    }
    base.update(kwargs)
    return ZipformerEvidence(**base)


class _FakeZipformerBackend:
    """Fake Zipformer backend returning canned emissions."""

    def __init__(self, emissions):
        self.emissions = emissions
        self.calls = 0

    def __call__(self, audio, group):
        self.calls += 1
        return self.emissions

    def close(self):
        pass


def valid_zipformer_aligner(emissions=None, backend_factory=None):
    if backend_factory is None:
        emissions = emissions or []

        def make_backend():
            return _FakeZipformerBackend(emissions)

        backend_factory = make_backend
    return ZipformerNeuralAligner(make_evidence(), backend_factory=backend_factory)


def make_logits_provider(logits):
    return lambda samples, sr: logits


def make_wav2vec_aligner(logits_provider=None, config=None):
    if config is None:
        config = Wav2Vec2Config()
    return Wav2Vec2ForcedAligner(logits_provider=logits_provider, config=config)


def blank_dominant_logits(frames, blank_id, vocab, strength=100.0):
    logits = np.full((frames, vocab), -strength, dtype=np.float32)
    logits[:, blank_id] = 0.0
    return logits


def uniform_logits(frames, vocab):
    """Maximally ambiguous — all logits equal."""
    return np.zeros((frames, vocab), dtype=np.float32)


def _aligned_logits(frames, blank_id, tokens):
    """Logits where each target token dominates in its own slice.

    The blank token is always at least as likely as any target token, but each
    target token gets a strong positive logit during its designated slice, so
    the CTC DP can trace a path through the full extended state sequence.
    """
    vocab = max(blank_id, max(tokens)) + 1
    logits = np.full((frames, vocab), -1.0, dtype=np.float32)
    logits[:, blank_id] = 0.0
    n = len(tokens)
    per = max(1, frames // (n + 1))
    offset = per
    for tok in tokens:
        logits[offset : offset + per, tok] = 10.0
        offset += per
    return logits


# ---------------------------------------------------------------------------
# A. Breath / recitation segmentation
# ---------------------------------------------------------------------------


class TestBreathGroupContract:
    def test_construction_validates_start_before_end(self):
        with pytest.raises(ValueError):
            BreathGroup(1.0, 0.5)

    def test_construction_rejects_non_finite(self):
        with pytest.raises(ValueError):
            BreathGroup(0.0, float("nan"))
        with pytest.raises(ValueError):
            BreathGroup(float("inf"), 1.0)

    def test_construction_rejects_negative(self):
        with pytest.raises(ValueError):
            BreathGroup(-0.1, 0.5)

    def test_construction_rejects_zero_length(self):
        with pytest.raises(ValueError):
            BreathGroup(0.5, 0.5)

    def test_construction_accepts_valid(self):
        g = BreathGroup(0.3, 0.7)
        assert g.start == 0.3
        assert g.end == 0.7
        assert g.score is None

    def test_construction_rejects_nan_start(self):
        with pytest.raises(ValueError):
            BreathGroup(float("nan"), 0.5)

    def test_construction_rejects_nan_end(self):
        with pytest.raises(ValueError):
            BreathGroup(0.1, float("nan"))

    def test_construction_rejects_inf_start(self):
        with pytest.raises(ValueError):
            BreathGroup(float("inf"), 0.5)

    def test_construction_rejects_inf_end(self):
        with pytest.raises(ValueError):
            BreathGroup(0.1, float("inf"))


class TestAudioBufferValidation:
    def test_rejects_empty_audio(self):
        with pytest.raises(ValueError, match="empty"):
            AudioBuffer(samples=np.array([], dtype=np.float32), sample_rate=16000)

    def test_rejects_zero_sample_rate(self):
        with pytest.raises(ValueError):
            AudioBuffer(samples=np.zeros(100, dtype=np.float32), sample_rate=0)

    def test_rejects_negative_sample_rate(self):
        with pytest.raises(ValueError):
            AudioBuffer(samples=np.zeros(100, dtype=np.float32), sample_rate=-1)

    def test_rejects_multichannel(self):
        bad = np.zeros((100, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="mono"):
            AudioBuffer(samples=bad, sample_rate=16000)

    def test_duration(self):
        buf = audio(1.0)
        assert buf.duration == pytest.approx(1.0, rel=1e-2)
        assert buf.sample_rate == SAMPLE_RATE

    def test_explicit_sample_rate(self):
        assert SAMPLE_RATE == 16_000

    def test_explicit_channel(self):
        buf = audio(1.0)
        assert buf.samples.ndim == 1  # mono


class TestEmptyAndSilentAudio:
    """Empty audio → explicit error; silent audio → explicit error."""

    def test_empty_audio_raises(self):
        with pytest.raises(ValueError, match="empty"):
            AudioBuffer(samples=np.array([], dtype=np.float32), sample_rate=16000)

    def test_silent_audio_rejected_by_segmenter(self):
        """Silent (all-zero) audio must not produce a whole-file interval."""
        called = [False]

        def backend_factory():
            called[0] = True
            return lambda a: []  # backend returns empty

        seg = QuranRecitationSegmenter(backend_factory=backend_factory)
        buf = silent_audio(2.0)
        with pytest.raises(NoBreathGroupsError):
            seg.segment(buf)
        assert called[0] is True

    def test_no_whole_file_interval_for_silent_audio(self):
        """Legacy silence fallback must not silently create a whole-file interval."""
        seg = QuranRecitationSegmenter(backend_factory=lambda: lambda a: [])
        with pytest.raises(NoBreathGroupsError):
            seg.segment(silent_audio(2.0))


class TestSegmenterOutputValidation:
    """Segmenter rejects invalid backend output: negative, reversed, overlapping,
    duplicate, unsorted, NaN, Infinity, zero-length, out-of-bounds intervals."""

    @pytest.mark.parametrize(
        "raw,err",
        [
            # valid ordered, non-overlapping, in-bounds
            ([(0.1, 0.5), (0.6, 0.9), (1.0, 1.5)], None),
            # overlapping
            ([(0.1, 0.5), (0.3, 0.7)], InvalidProviderOutputError),
            # duplicate
            ([(0.1, 0.5), (0.1, 0.5)], InvalidProviderOutputError),
            # unsorted (reversed)
            ([(0.6, 0.9), (0.1, 0.5)], InvalidProviderOutputError),
            # negative start
            ([(0.1, 0.5), (-0.1, 0.3)], InvalidProviderOutputError),
            # NaN
            ([(0.1, 0.5), (0.1, float("nan"))], InvalidProviderOutputError),
            # out-of-bounds end
            ([(0.1, 0.5), (0.6, 99.0)], InvalidProviderOutputError),
            # zero-length
            ([(0.1, 0.5), (0.6, 0.6)], InvalidProviderOutputError),
            # reversed start > end
            ([(0.1, 0.5), (0.7, 0.3)], InvalidProviderOutputError),
        ],
    )
    def test_interval_validation(self, raw, err):
        seg = QuranRecitationSegmenter(
            loader=lambda v, r: v,
            backend_factory=lambda: lambda a: list(raw),
        )
        buf = audio(3.0)
        if err:
            with pytest.raises(err):
                seg.segment(buf)
        else:
            result = seg.segment(buf)
            assert len(result) == len(raw)

    def test_empty_backend_output_raises(self):
        seg = QuranRecitationSegmenter(
            loader=lambda v, r: v,
            backend_factory=lambda: lambda a: [],
        )
        with pytest.raises(NoBreathGroupsError):
            seg.segment(audio(1.0))

    def test_backend_returning_non_tuple_rejected(self):
        seg = QuranRecitationSegmenter(
            loader=lambda v, r: v,
            backend_factory=lambda: lambda a: [object()],
        )
        with pytest.raises(InvalidProviderOutputError):
            seg.segment(audio(1.0))

    def test_backend_returning_dict_rejected(self):
        seg = QuranRecitationSegmenter(
            loader=lambda v, r: v,
            backend_factory=lambda: lambda a: [{"start": 0.1, "end": 0.5}],
        )
        with pytest.raises(InvalidProviderOutputError):
            seg.segment(audio(1.0))

    def test_too_many_groups_rejected(self):
        seg = QuranRecitationSegmenter(
            config=SegmenterConfig(max_groups=2),
            loader=lambda v, r: v,
            backend_factory=lambda: lambda a: [(0.0, 0.1), (0.2, 0.3), (0.4, 0.5)],
        )
        with pytest.raises(InvalidProviderOutputError):
            seg.segment(audio(1.0))


class TestSegmenterResourceLimits:
    """Resource limits are checked BEFORE backend/model allocation."""

    def test_duration_limit_prevents_backend_allocation(self):
        calls = [0]

        def factory():
            calls[0] += 1
            return lambda a: [(0.1, 0.5)]

        seg = QuranRecitationSegmenter(
            config=SegmenterConfig(max_duration_sec=0.05),
            backend_factory=factory,
        )
        with pytest.raises(SegmenterError):
            seg.segment(audio(1.0))
        assert calls[0] == 0, "backend must not be allocated when duration exceeds limit"

    def test_sample_count_limit_prevents_backend_allocation(self):
        calls = [0]

        def factory():
            calls[0] += 1
            return lambda a: [(0.1, 0.5)]

        seg = QuranRecitationSegmenter(
            config=SegmenterConfig(max_samples=10),
            backend_factory=factory,
        )
        with pytest.raises(SegmenterError):
            seg.segment(audio(1.0))
        assert calls[0] == 0

    def test_duration_below_limit_allows_backend(self):
        calls = [0]

        def factory():
            calls[0] += 1
            return lambda a: [(0.1, 0.5)]

        seg = QuranRecitationSegmenter(
            config=SegmenterConfig(max_duration_sec=2.0, sample_rate=16000),
            backend_factory=factory,
        )
        seg.segment(audio(1.0))
        assert calls[0] == 1

    def test_group_count_limit(self):
        seg = QuranRecitationSegmenter(
            config=SegmenterConfig(max_groups=2),
            backend_factory=lambda: lambda a: [(0.0, 0.1), (0.2, 0.3), (0.4, 0.5)],
        )
        with pytest.raises(InvalidProviderOutputError):
            seg.segment(audio(1.0))

    def test_group_duration_limit(self):
        seg = QuranRecitationSegmenter(
            config=SegmenterConfig(max_group_duration_sec=0.1),
            backend_factory=lambda: lambda a: [(0.0, 0.5)],
        )
        with pytest.raises(InvalidProviderOutputError):
            seg.segment(audio(1.0))


class TestSegmenterLazyBackend:
    """Backend factory is lazy — only called when segment() is invoked."""

    def test_backend_not_called_on_construction(self):
        calls = [0]
        QuranRecitationSegmenter(backend_factory=lambda: _CountingBackend(calls))
        assert calls[0] == 0

    def test_backend_called_once_and_cached(self):
        calls = [0]

        def factory():
            calls[0] += 1
            return _CountingBackend(None)

        seg = QuranRecitationSegmenter(backend_factory=factory)
        seg.segment(audio(1.0))
        seg.segment(audio(1.0))
        assert calls[0] == 1, "backend must be cached after first construction"


class _CountingBackend:
    def __init__(self, counter):
        if counter is not None:
            counter[0] += 1

    def __call__(self, audio):
        return [(0.1, 0.5)]


class TestSegmenterInjection:
    """Segmenter boundary is injectable."""

    def test_injected_backend_used(self):
        seg = QuranRecitationSegmenter(backend_factory=lambda: _SilentBackend())
        assert seg._backend_factory is not None

    def test_injected_loader(self):
        called = [False]

        def loader(path, sr):
            called[0] = True
            return audio(1.0)

        seg = QuranRecitationSegmenter(loader=loader)
        buf = seg._loader("fake.wav", SAMPLE_RATE)
        assert called[0]
        assert isinstance(buf, AudioBuffer)

    def test_default_backend_is_pinned_and_lazy(self):
        seg = QuranRecitationSegmenter()
        backend = seg._get_backend()
        assert isinstance(backend, TransformersRecitationSegmenterBackend)
        assert backend._runtime is None


class _SilentBackend:
    def __call__(self, audio):
        return []


# ---------------------------------------------------------------------------
# B. Zipformer reference alignment
# ---------------------------------------------------------------------------


class TestZipformerEvidence:
    def test_valid_evidence(self):
        e = make_evidence()
        assert e.repository == "Quran-Lab/zipformer_p-arabic-v3"
        assert e.approved is True

    def test_wrong_repository(self):
        with pytest.raises(ValueError, match="repository"):
            make_evidence(repository="other/repo")

    def test_unapproved_fails(self):
        with pytest.raises(ValueError, match="approved"):
            make_evidence(approved=False)

    def test_mutable_revision_fails(self):
        with pytest.raises(ValueError, match="revision"):
            make_evidence(revision="v1.0.0")
        with pytest.raises(ValueError, match="revision"):
            make_evidence(revision="short")

    def test_revision_uppercase_accepted(self):
        rev = "0123456789abcdef0123456789abcdef01234567"
        e = make_evidence(revision=rev)
        assert len(e.revision) == 40

    def test_invalid_vocab_size(self):
        with pytest.raises(ValueError):
            make_evidence(vocabulary_size=0)
        with pytest.raises(ValueError):
            make_evidence(vocabulary_size=-1)
        with pytest.raises(ValueError):
            make_evidence(vocabulary_size=True)  # bool rejected

    def test_invalid_blank_id(self):
        with pytest.raises(ValueError, match="blank_id"):
            make_evidence(blank_id=-1)
        with pytest.raises(ValueError, match="blank_id"):
            make_evidence(blank_id=251)  # >= vocab_size

    def test_sample_rate_validation(self):
        with pytest.raises(ValueError):
            make_evidence(sample_rate=0)

    def test_feature_type_validation(self):
        with pytest.raises(ValueError, match="feature"):
            make_evidence(feature_kind="mel")
        e = make_evidence(feature_kind="kaldi-fbank")
        assert e.feature_kind == "kaldi-fbank"

    def test_token_table_sha256_validation(self):
        e = make_evidence(token_table_sha256="a" * 64)
        assert e.token_table_sha256 == "a" * 64
        with pytest.raises(ValueError):
            make_evidence(token_table_sha256="short")
        with pytest.raises(ValueError):
            make_evidence(token_table_sha256="z" * 64)

    def test_phoneme_mapping_required(self):
        e = make_evidence(phoneme_mapping_required=True)
        assert e.phoneme_mapping_required is True
        e2 = make_evidence()
        assert e2.phoneme_mapping_required is False

    def test_tokens_txt_documented_in_docstring(self):
        docstring = ZipformerEvidence.__doc__ or ""
        assert "tokens.txt" in docstring
        assert "blank_id" in docstring


class TestZipformerProvider:
    """Backend-neutral, injectable, lazy construction."""

    def test_no_evidence_raises_model_unavailable(self):
        aligner = ZipformerNeuralAligner()
        with pytest.raises(ModelUnavailableError, match="gated"):
            aligner._get_backend()

    def test_no_backend_factory_raises_model_unavailable(self):
        aligner = ZipformerNeuralAligner(make_evidence())
        with pytest.raises(ModelUnavailableError):
            aligner._get_backend()

    def test_backend_factory_lazy(self):
        calls = [0]

        def factory():
            calls[0] += 1
            return _FakeZipformerBackend([])

        aligner = ZipformerNeuralAligner(make_evidence(), backend_factory=factory)
        assert calls[0] == 0
        aligner._get_backend()
        assert calls[0] == 1
        aligner._get_backend()
        assert calls[0] == 1, "backend must be cached"

    def test_backend_factory_exception_wrapped(self):
        def factory():
            raise RuntimeError("model download failed")

        aligner = ZipformerNeuralAligner(make_evidence(), backend_factory=factory)
        with pytest.raises(ModelUnavailableError, match="failed to initialize"):
            aligner._get_backend()

    def test_close_unload(self):
        aligner = valid_zipformer_aligner(
            emissions=[PhonemeEmission(token_id=1, start_frame=0, end_frame=5, score=0.9)]
        )
        aligner.align(audio(1.0), BreathGroup(0.0, 0.5))
        aligner.close()
        assert aligner._backend is None

    def test_align_validates_emissions(self):
        aligner = valid_zipformer_aligner(
            emissions=[
                PhonemeEmission(token_id=1, start_frame=0, end_frame=3, score=0.95),
                PhonemeEmission(token_id=2, start_frame=3, end_frame=8, score=0.90),
            ]
        )
        result = aligner.align(audio(1.0), BreathGroup(0.0, 0.5))
        assert len(result) == 2

    def test_align_rejects_non_phoneme_emission(self):
        backend = _FakeZipformerBackend([object()])
        aligner = ZipformerNeuralAligner(make_evidence(), backend_factory=lambda: backend)
        with pytest.raises(InvalidProviderOutputError):
            aligner.align(audio(1.0), BreathGroup(0.0, 0.5))

    def test_align_rejects_out_of_vocab_token(self):
        backend = _FakeZipformerBackend(
            [PhonemeEmission(token_id=300, start_frame=0, end_frame=1, score=0.9)]
        )
        aligner = ZipformerNeuralAligner(make_evidence(), backend_factory=lambda: backend)
        with pytest.raises(InvalidProviderOutputError):
            aligner.align(audio(1.0), BreathGroup(0.0, 0.5))

    def test_align_rejects_blank_token_in_emissions(self):
        backend = _FakeZipformerBackend(
            [PhonemeEmission(token_id=250, start_frame=0, end_frame=1, score=0.9)]
        )
        aligner = ZipformerNeuralAligner(make_evidence(), backend_factory=lambda: backend)
        with pytest.raises(InvalidProviderOutputError):
            aligner.align(audio(1.0), BreathGroup(0.0, 0.5))

    def test_align_rejects_non_monotonic_frames(self):
        backend = _FakeZipformerBackend(
            [
                PhonemeEmission(token_id=1, start_frame=5, end_frame=10, score=0.9),
                PhonemeEmission(token_id=2, start_frame=3, end_frame=8, score=0.9),
            ]
        )
        aligner = ZipformerNeuralAligner(make_evidence(), backend_factory=lambda: backend)
        with pytest.raises(InvalidProviderOutputError):
            aligner.align(audio(1.0), BreathGroup(0.0, 0.5))

    def test_align_rejects_negative_frame_at_construction(self):
        """PhonemeEmission constructor rejects negative start_frame."""
        with pytest.raises(ValueError):
            PhonemeEmission(token_id=1, start_frame=-1, end_frame=3, score=0.9)

    def test_align_rejects_reversed_frame_bounds_at_construction(self):
        """PhonemeEmission constructor rejects end_frame <= start_frame."""
        with pytest.raises(ValueError):
            PhonemeEmission(token_id=1, start_frame=5, end_frame=5, score=0.9)

    def test_align_rejects_non_integer_token_id(self):
        with pytest.raises(ValueError):
            PhonemeEmission(token_id=1.5, start_frame=0, end_frame=1, score=0.9)

    def test_align_rejects_non_integer_frames(self):
        with pytest.raises(ValueError):
            PhonemeEmission(token_id=1, start_frame=0.5, end_frame=1, score=0.9)

    def test_align_rejects_non_finite_score(self):
        with pytest.raises(ValueError):
            PhonemeEmission(token_id=1, start_frame=0, end_frame=1, score=float("nan"))

    def test_align_accepts_valid_phoneme_emission(self):
        backend = _FakeZipformerBackend(
            [PhonemeEmission(token_id=1, start_frame=0, end_frame=10, score=0.95)]
        )
        aligner = ZipformerNeuralAligner(make_evidence(), backend_factory=lambda: backend)
        result = aligner.align(audio(1.0), BreathGroup(0.0, 0.5))
        assert len(result) == 1

    def test_align_sample_rate_mismatch(self):
        aligner = ZipformerNeuralAligner(
            make_evidence(sample_rate=8000),
            backend_factory=lambda: _FakeZipformerBackend([]),
        )
        buf = AudioBuffer(samples=np.zeros(100, dtype=np.float32), sample_rate=16000)
        with pytest.raises(InvalidProviderOutputError):
            aligner.align(buf, BreathGroup(0.0, 0.5))

    def test_align_rejects_provider_runtime_failure(self):
        class FailingBackend:
            def __call__(self, audio, group):
                raise RuntimeError("backend crashed")

            def close(self):
                pass

        aligner = ZipformerNeuralAligner(make_evidence(), backend_factory=lambda: FailingBackend())
        with pytest.raises(RuntimeError, match="backend crashed"):
            aligner.align(audio(1.0), BreathGroup(0.0, 0.5))


# ---------------------------------------------------------------------------
# C. Wav2Vec2 CTC forced alignment
# ---------------------------------------------------------------------------


class TestWav2Vec2Config:
    def test_valid_defaults(self):
        c = Wav2Vec2Config()
        assert c.blank_id == 0
        assert c.post_roll_sec == 0.20
        assert c.max_frames == 20_000

    def test_post_roll_must_be_positive(self):
        with pytest.raises(ValueError):
            Wav2Vec2Config(post_roll_sec=0)

    def test_post_roll_finite(self):
        with pytest.raises(ValueError):
            Wav2Vec2Config(post_roll_sec=float("nan"))

    def test_max_limits_positive(self):
        with pytest.raises(ValueError):
            Wav2Vec2Config(max_frames=0)

    def test_blank_id_non_negative(self):
        with pytest.raises(ValueError):
            Wav2Vec2Config(blank_id=-1)


class TestWav2Vec2LazyProvider:
    def test_no_provider_raises_model_unavailable(self):
        aligner = Wav2Vec2ForcedAligner()
        with pytest.raises(ModelUnavailableError):
            aligner._get_provider()

    def test_provider_factory_cached(self):
        provider = MagicMock(return_value=np.zeros((10, 3), dtype=np.float32))
        aligner = Wav2Vec2ForcedAligner(logits_provider=provider)
        p1 = aligner._get_provider()
        p2 = aligner._get_provider()
        assert p1 is p2

    def test_provider_injection(self):
        provider = MagicMock(return_value=np.zeros((10, 3), dtype=np.float32))
        aligner = Wav2Vec2ForcedAligner(logits_provider=provider)
        assert aligner._get_provider() is provider

    def test_close_resets_provider(self):
        provider = MagicMock(return_value=np.zeros((10, 3), dtype=np.float32))
        aligner = Wav2Vec2ForcedAligner(logits_provider=provider)
        aligner._get_provider()
        aligner.close()
        assert aligner._provider is None
        assert aligner._provider_factory is None


class TestWav2Vec2LogitsValidation:
    def test_empty_logits(self):
        aligner = make_wav2vec_aligner(
            logits_provider=make_logits_provider(np.zeros((0, 3), dtype=np.float32))
        )
        with pytest.raises(InvalidProviderOutputError):
            aligner.align_group(audio(1.0), BreathGroup(0.1, 0.5), [1], ["a"])

    def test_nan_logits(self):
        bad = np.full((10, 3), float("nan"), dtype=np.float32)
        aligner = make_wav2vec_aligner(logits_provider=make_logits_provider(bad))
        with pytest.raises(InvalidProviderOutputError):
            aligner.align_group(audio(1.0), BreathGroup(0.1, 0.5), [1], ["a"])

    def test_inf_logits(self):
        bad = np.full((10, 3), float("inf"), dtype=np.float32)
        aligner = make_wav2vec_aligner(logits_provider=make_logits_provider(bad))
        with pytest.raises(InvalidProviderOutputError):
            aligner.align_group(audio(1.0), BreathGroup(0.1, 0.5), [1], ["a"])

    def test_wrong_rank_logits(self):
        bad = np.zeros((10, 3, 1), dtype=np.float32)
        aligner = make_wav2vec_aligner(logits_provider=make_logits_provider(bad))
        with pytest.raises(InvalidProviderOutputError):
            aligner.align_group(audio(1.0), BreathGroup(0.1, 0.5), [1], ["a"])

    def test_empty_vocab_logits(self):
        bad = np.zeros((10, 0), dtype=np.float32)
        aligner = make_wav2vec_aligner(logits_provider=make_logits_provider(bad))
        with pytest.raises(InvalidProviderOutputError):
            aligner.align_group(audio(1.0), BreathGroup(0.1, 0.5), [1], ["a"])


class TestWav2Vec2CTC:
    """CTC-specific alignment behaviour."""

    def test_blank_transitions_handled(self):
        aligner = make_wav2vec_aligner(
            logits_provider=make_logits_provider(_aligned_logits(16, 0, [1, 2])),
            config=Wav2Vec2Config(blank_id=0, max_frames=100, max_tokens=100),
        )
        spans = aligner.align_group(audio(1.0), BreathGroup(0.1, 0.5), [1, 2], ["a", "b"])
        assert len(spans) == 2
        assert spans[0].token == "a"
        assert spans[1].token == "b"

    def test_repeated_ctc_tokens(self):
        """Two identical token IDs in the target must map to distinct text."""
        aligner = make_wav2vec_aligner(
            logits_provider=make_logits_provider(_aligned_logits(16, 0, [1, 1])),
            config=Wav2Vec2Config(blank_id=0, max_frames=100, max_tokens=100),
        )
        spans = aligner.align_group(audio(1.0), BreathGroup(0.1, 0.5), [1, 1], ["a", "b"])
        assert len(spans) == 2
        assert spans[0].token == "a"
        assert spans[1].token == "b"

    def test_impossible_target_raises(self):
        """A single frame cannot align two tokens → AlignmentError."""
        aligner = make_wav2vec_aligner(
            logits_provider=make_logits_provider(np.full((1, 3), 1.0, dtype=np.float32)),
            config=Wav2Vec2Config(blank_id=0, max_frames=100, max_tokens=100),
        )
        with pytest.raises(AlignmentError):
            aligner.align_group(audio(1.0), BreathGroup(0.1, 0.5), [1, 2], ["a", "b"])

    def test_impossible_target_no_guessed_timestamps(self):
        """Impossible alignment must not produce guessed timestamps."""
        aligner = make_wav2vec_aligner(
            logits_provider=make_logits_provider(np.full((1, 3), 1.0, dtype=np.float32)),
        )
        with pytest.raises((AlignmentError, InvalidProviderOutputError)):
            aligner.align_group(audio(1.0), BreathGroup(0.1, 0.5), [1, 2], ["a", "b"])

    def test_oov_token_in_target_rejected(self):
        aligner = make_wav2vec_aligner(
            logits_provider=make_logits_provider(np.zeros((10, 3), dtype=np.float32)),
        )
        with pytest.raises(InvalidProviderOutputError):
            aligner.align_group(audio(1.0), BreathGroup(0.1, 0.5), [5], ["x"])

    def test_empty_target_rejected(self):
        aligner = make_wav2vec_aligner(
            logits_provider=make_logits_provider(np.zeros((10, 3), dtype=np.float32)),
        )
        with pytest.raises(InvalidProviderOutputError):
            aligner.align_group(audio(1.0), BreathGroup(0.1, 0.5), [], [])

    def test_parallel_mismatch_rejected(self):
        aligner = make_wav2vec_aligner(
            logits_provider=make_logits_provider(np.zeros((10, 3), dtype=np.float32)),
        )
        with pytest.raises(InvalidProviderOutputError):
            aligner.align_group(audio(1.0), BreathGroup(0.1, 0.5), [1, 2], ["a"])

    def test_group_end_exceeds_audio_duration(self):
        aligner = make_wav2vec_aligner(
            logits_provider=make_logits_provider(_aligned_logits(16, 0, [1, 2])),
            config=Wav2Vec2Config(blank_id=0, max_frames=100, max_tokens=100),
        )
        buf = audio(0.3)
        with pytest.raises(InvalidProviderOutputError):
            aligner.align_group(buf, BreathGroup(0.1, 0.5), [1, 2], ["a", "b"])


class TestWav2Vec2FrameConversion:
    def test_end_times_exclusive(self):
        aligner = make_wav2vec_aligner(
            logits_provider=make_logits_provider(_aligned_logits(16, 0, [1, 2])),
            config=Wav2Vec2Config(blank_id=0, max_frames=100, max_tokens=100),
        )
        spans = aligner.align_group(
            audio(2.0),
            BreathGroup(0.5, 1.0),
            [1, 2],
            ["a", "b"],
            next_group_start=1.5,
        )
        for span in spans:
            assert span.end > span.start

    def test_end_times_inside_audio_bounds(self):
        aligner = make_wav2vec_aligner(
            logits_provider=make_logits_provider(_aligned_logits(16, 0, [1, 2])),
            config=Wav2Vec2Config(blank_id=0, max_frames=100, max_tokens=100),
        )
        buf = audio(2.0)
        spans = aligner.align_group(
            buf,
            BreathGroup(0.5, 1.0),
            [1, 2],
            ["a", "b"],
        )
        for span in spans:
            assert span.end <= buf.duration

    def test_uses_correct_local_audio_origin(self):
        """Timestamps must be relative to group.start."""
        aligner = make_wav2vec_aligner(
            logits_provider=make_logits_provider(_aligned_logits(16, 0, [1, 2])),
            config=Wav2Vec2Config(blank_id=0, max_frames=100, max_tokens=100),
        )
        spans = aligner.align_group(
            audio(2.0),
            BreathGroup(0.5, 1.0),
            [1, 2],
            ["a", "b"],
        )
        for span in spans:
            assert span.start >= 0.5

    def test_frame_count_verified(self):
        """Frame-to-time uses the actual returned frame count."""
        captured = [None]

        def provider(samples, sr):
            logits = _aligned_logits(16, 0, [1, 2])
            captured[0] = logits.shape[0]
            return logits

        aligner = make_wav2vec_aligner(logits_provider=provider)
        spans = aligner.align_group(
            audio(2.0),
            BreathGroup(0.5, 1.0),
            [1, 2],
            ["a", "b"],
        )
        assert captured[0] == 16
        assert len(spans) == 2

    def test_token_span_preserves_all_consecutive_ctc_frames(self):
        """A token's timing spans its full CTC run, not only its final frame."""
        logits = np.full((20, 2), -2.0, dtype=np.float32)
        logits[:, 0] = 0.0
        logits[5:12, 1] = 8.0
        aligner = make_wav2vec_aligner(
            logits_provider=make_logits_provider(logits),
            config=Wav2Vec2Config(blank_id=0, max_frames=100, max_tokens=100),
        )
        spans = aligner.align_group(audio(1.0), BreathGroup(0.1, 0.5), [1], ["a"])
        assert len(spans) == 1
        assert spans[0].start == pytest.approx(0.25)
        assert spans[0].end == pytest.approx(0.46)
        assert spans[0].end - spans[0].start > 0.2


class TestWav2Vec2PostRoll:
    def test_post_roll_positive(self):
        assert Wav2Vec2Config().post_roll_sec > 0

    def test_post_roll_finite(self):
        assert math.isfinite(Wav2Vec2Config().post_roll_sec)

    def test_post_roll_bounded_by_audio_end(self):
        """Post-roll beyond audio end must be clipped to audio duration."""
        aligner = make_wav2vec_aligner(
            logits_provider=make_logits_provider(_aligned_logits(8, 0, [1, 2])),
            config=Wav2Vec2Config(blank_id=0, post_roll_sec=10.0, max_frames=1000, max_tokens=1000),
        )
        buf = audio(1.0)
        spans = aligner.align_group(
            buf,
            BreathGroup(0.8, 0.9),
            [1, 2],
            ["a", "b"],
        )
        for span in spans:
            assert span.end <= buf.duration

    def test_post_roll_clipped_at_next_breath(self):
        """Post-roll must not extend into the next breath group."""
        aligner = make_wav2vec_aligner(
            logits_provider=make_logits_provider(_aligned_logits(32, 0, [1, 2])),
            config=Wav2Vec2Config(blank_id=0, post_roll_sec=10.0, max_frames=1000, max_tokens=1000),
        )
        buf = audio(5.0)
        next_start = 1.0
        spans = aligner.align_group(
            buf,
            BreathGroup(0.5, 0.8),
            [1, 2],
            ["a", "b"],
            next_group_start=next_start,
        )
        for span in spans:
            assert span.end <= next_start

    def test_post_roll_retention(self):
        """Trailing phonetic decay may be retained inside post-roll window."""
        aligner = make_wav2vec_aligner(
            logits_provider=make_logits_provider(_aligned_logits(32, 0, [1, 2])),
            config=Wav2Vec2Config(blank_id=0, post_roll_sec=0.5, max_frames=1000, max_tokens=1000),
        )
        buf = audio(5.0)
        spans = aligner.align_group(
            buf,
            BreathGroup(0.5, 0.8),
            [1, 2],
            ["a", "b"],
        )
        for span in spans:
            assert span.end <= buf.duration

    def test_tiny_post_roll_window_raises(self):
        """Post-roll so small as to yield an empty window raises."""
        aligner = make_wav2vec_aligner(
            logits_provider=make_logits_provider(np.zeros((1, 3), dtype=np.float32)),
            config=Wav2Vec2Config(blank_id=0, post_roll_sec=1e-9, max_frames=100, max_tokens=100),
        )
        with pytest.raises((InvalidProviderOutputError, AlignmentError)):
            aligner.align_group(
                audio(5.0),
                BreathGroup(0.99999, 1.0),
                [1],
                ["a"],
                next_group_start=0.99998,
            )


class TestWav2Vec2PerBreathIsolation:
    def test_no_token_state_crosses_breath_boundaries(self):
        aligner = make_wav2vec_aligner(
            logits_provider=make_logits_provider(_aligned_logits(32, 0, [1, 2, 3, 4])),
            config=Wav2Vec2Config(blank_id=0, max_frames=1000, max_tokens=1000),
        )
        buf = audio(4.0)
        spans1 = aligner.align_group(
            buf,
            BreathGroup(0.0, 1.0),
            [1, 2],
            ["a", "b"],
            next_group_start=2.0,
        )
        spans2 = aligner.align_group(
            buf,
            BreathGroup(2.0, 3.0),
            [3, 4],
            ["c", "d"],
        )
        assert len(spans1) == 2
        assert len(spans2) == 2
        assert all(s.end <= 2.0 for s in spans1)
        assert all(s.start >= 2.0 for s in spans2)

    def test_no_trellis_state_crosses_breath_boundaries(self):
        """Trellis is built per-breath; different slices → different spans."""
        aligner = make_wav2vec_aligner(
            logits_provider=make_logits_provider(_aligned_logits(16, 0, [1, 2])),
            config=Wav2Vec2Config(blank_id=0, max_frames=100, max_tokens=100),
        )
        buf = audio(4.0)
        spans1 = aligner.align_group(
            buf,
            BreathGroup(0.0, 1.0),
            [1, 2],
            ["a", "b"],
            next_group_start=2.0,
        )
        spans2 = aligner.align_group(
            buf,
            BreathGroup(2.0, 3.0),
            [1, 2],
            ["a", "b"],
        )
        assert spans1[0].start != spans2[0].start

    def test_adversarial_logits_in_another_breath_group(self):
        """Logits in one breath group must NOT influence another."""
        group1_logits = _aligned_logits(16, 0, [1, 2])
        # Group 2 uses vocab 5 (tokens 3, 4), so logits must have vocab >= 5
        group2_logits = _aligned_logits(16, 0, [3, 4])

        call_count = [0]

        def provider(samples, sr):
            call_count[0] += 1
            if call_count[0] == 1:
                return group1_logits
            return group2_logits

        aligner = Wav2Vec2ForcedAligner(
            logits_provider=provider,
            config=Wav2Vec2Config(blank_id=0, max_frames=1000, max_tokens=1000),
        )
        buf = audio(4.0)
        spans1 = aligner.align_group(
            buf,
            BreathGroup(0.0, 1.0),
            [1, 2],
            ["a", "b"],
            next_group_start=2.0,
        )
        spans2 = aligner.align_group(
            buf,
            BreathGroup(2.0, 3.0),
            [3, 4],
            ["c", "d"],
        )
        assert len(spans1) == 2
        assert len(spans2) == 2
        assert spans1[0].token == "a"
        assert spans2[0].token == "c"


class TestWav2Vec2NoFabrication:
    def test_partial_audio_no_fabricated_spans(self):
        """Partial audio (zero frames) → no spans."""
        aligner = make_wav2vec_aligner(
            logits_provider=make_logits_provider(np.zeros((0, 3), dtype=np.float32))
        )
        with pytest.raises((InvalidProviderOutputError, AlignmentError)):
            aligner.align_group(audio(0.1), BreathGroup(0.0, 0.00001), [1], ["a"])

    def test_unrelated_audio_no_fabricated_spans(self):
        """Logits that cannot align target → no spans, raised error."""
        # 1 frame for 2 tokens → impossible
        aligner = make_wav2vec_aligner(
            logits_provider=make_logits_provider(np.full((1, 3), 1.0, dtype=np.float32))
        )
        with pytest.raises((AlignmentError, InvalidProviderOutputError)):
            aligner.align_group(audio(1.0), BreathGroup(0.1, 0.5), [1, 2], ["a", "b"])

    def test_ambiguous_audio_no_fabricated_spans(self):
        """Flat logits with insufficient frames → no fabricated spans."""
        aligner = make_wav2vec_aligner(
            logits_provider=make_logits_provider(np.zeros((1, 3), dtype=np.float32))
        )
        with pytest.raises((AlignmentError, InvalidProviderOutputError)):
            aligner.align_group(audio(1.0), BreathGroup(0.1, 0.5), [1, 2], ["a", "b"])

    def test_no_fabrication_when_ctc_fails(self):
        """When CTC backtrace fails, no spans should be returned."""
        aligner = make_wav2vec_aligner(
            logits_provider=make_logits_provider(np.full((1, 3), 1.0, dtype=np.float32))
        )
        with pytest.raises(AlignmentError):
            aligner.align_group(audio(1.0), BreathGroup(0.1, 0.5), [1, 2], ["a", "b"])


# ---------------------------------------------------------------------------
# D. Hybrid orchestration
# ---------------------------------------------------------------------------


class TestHybridPipeline:
    """Full pipeline orchestration with injected providers."""

    def _make_aligner(
        self,
        segmenter_groups=None,
        logits=None,
        reference_aligner=None,
    ):
        if segmenter_groups is None:
            segmenter_groups = [(0.0, 0.5), (0.6, 1.0)]
        if logits is None:
            logits = _aligned_logits(32, 0, [1, 2, 3, 4])
        seg = QuranRecitationSegmenter(
            backend_factory=lambda: lambda a: list(segmenter_groups),
        )
        fa = Wav2Vec2ForcedAligner(
            logits_provider=make_logits_provider(logits),
            config=Wav2Vec2Config(blank_id=0, max_frames=1000, max_tokens=1000),
        )
        return HybridQuranAligner(
            segmenter=seg,
            reference_aligner=reference_aligner,
            forced_aligner=fa,
        )

    def test_full_pipeline_orchestration(self):
        aligner = self._make_aligner()
        buf = audio(2.0)
        result = aligner.align(buf, [([1, 2], ["a", "b"]), ([3, 4], ["c", "d"])])
        assert len(result.breath_groups) == 2
        assert len(result.spans) == 4
        assert result.metadata["provider"] == "wav2vec2-ctc"
        aligner.close()

    def test_canonical_reference_is_tokenized_per_physical_breath(self):
        class _CanonicalTokenizerProvider:
            def __call__(self, samples, sample_rate):
                return _aligned_logits(32, 0, [1])

            def encode_text(self, text):
                assert "".join(text.split())
                return [1], ["canonical-token"]

        seg = QuranRecitationSegmenter(
            backend_factory=lambda: lambda a: [(0.0, 0.5)],
        )
        forced = Wav2Vec2ForcedAligner(
            logits_provider=_CanonicalTokenizerProvider(),
            config=Wav2Vec2Config(blank_id=0, max_frames=1000, max_tokens=1000),
        )
        pipeline = HybridQuranAligner(
            segmenter=seg,
            forced_aligner=forced,
            canonical_reference_provider=CanonicalQuranReferenceProvider(),
        )
        result = pipeline.align(audio(2.0), references=[CanonicalReferenceRequest(1, 1, 1, "hafs")])
        assert len(result.spans) == 1
        assert result.spans[0].token == "canonical-token"
        assert result.metadata["canonical_reference_groups"] == 1
        reference = result.metadata["canonical_references"][0]
        assert reference["riwaya"] == "hafs"
        assert reference["parts"][0]["surah_id"] == 1
        assert len(reference["source_file_sha256"]) == 64
        pipeline.close()

    def test_canonical_stage_order_is_zipformer_then_tokenizer_then_ctc(self):
        stages = []

        class _TrackingTokenizerProvider:
            def encode_text(self, text):
                stages.append("tokenizer")
                return [1], ["q"]

            def __call__(self, samples, sample_rate):
                stages.append("ctc")
                return _aligned_logits(32, 0, [1])

        segmenter = QuranRecitationSegmenter(
            backend_factory=lambda: _TrackingBackend(stages, "segmentation"),
        )
        reference_aligner = ZipformerNeuralAligner(
            make_evidence(), backend_factory=lambda: _TrackingRef(stages)
        )
        forced = Wav2Vec2ForcedAligner(
            logits_provider=_TrackingTokenizerProvider(),
            config=Wav2Vec2Config(blank_id=0, max_frames=1000, max_tokens=1000),
        )
        pipeline = HybridQuranAligner(
            segmenter=segmenter,
            reference_aligner=reference_aligner,
            forced_aligner=forced,
            canonical_reference_provider=CanonicalQuranReferenceProvider(),
        )
        pipeline.align(
            audio(2.0),
            references=[
                CanonicalReferenceRequest(1, 1, 1, "hafs"),
                CanonicalReferenceRequest(1, 2, 2, "hafs"),
            ],
        )
        assert stages == [
            "segmentation",
            "reference",
            "tokenizer",
            "ctc",
            "reference",
            "tokenizer",
            "ctc",
        ]
        pipeline.close()

    def test_exact_stage_ordering(self):
        """Pipeline must run: segmentation → reference evidence → CTC → validation."""
        stages = []

        seg = QuranRecitationSegmenter(
            backend_factory=lambda: _TrackingBackend(stages, "segmentation"),
        )
        ref = ZipformerNeuralAligner(
            make_evidence(),
            backend_factory=lambda: _TrackingRef(stages),
        )
        fa = Wav2Vec2ForcedAligner(
            logits_provider=_TrackingLogits(stages, _aligned_logits(32, 0, [1, 2, 3, 4])),
            config=Wav2Vec2Config(blank_id=0, max_frames=1000, max_tokens=1000),
        )
        pipeline = HybridQuranAligner(
            segmenter=seg,
            reference_aligner=ref,
            forced_aligner=fa,
        )
        buf = audio(2.0)
        pipeline.align(buf, [([1, 2], ["a", "b"]), ([3, 4], ["c", "d"])])
        # Per-group ordering: reference evidence THEN ctc, for each breath group
        expected = ["segmentation", "reference", "ctc", "reference", "ctc"]
        assert stages == expected
        pipeline.close()

    def test_one_target_per_breath(self):
        aligner = self._make_aligner()
        buf = audio(2.0)
        with pytest.raises(InvalidProviderOutputError):
            aligner.align(buf, [([1, 2], ["a", "b"])])  # 1 target for 2 groups
        aligner.close()

    def test_four_breath_groups_require_four_caller_supplied_targets(self):
        groups = [(0.0, 0.3), (0.4, 0.7), (0.8, 1.1), (1.2, 1.5)]
        aligner = self._make_aligner(segmenter_groups=groups)
        buf = audio(2.0)
        targets = [([index + 1], [f"t{index}"]) for index in range(4)]
        result = aligner.align(buf, targets)
        assert len(result.breath_groups) == 4
        assert {span.token for span in result.spans} <= {f"t{index}" for index in range(4)}
        with pytest.raises(InvalidProviderOutputError, match="one reference target"):
            aligner.align(buf, targets[:3])
        aligner.close()

    def test_stage_failure_propagates(self):
        """Segmenter failure must propagate, not be silently swallowed."""
        seg = QuranRecitationSegmenter(backend_factory=lambda: _SilentBackend())
        aligner = HybridQuranAligner(
            segmenter=seg,
            forced_aligner=Wav2Vec2ForcedAligner(
                logits_provider=make_logits_provider(np.zeros((5, 3), dtype=np.float32)),
            ),
        )
        with pytest.raises((NoBreathGroupsError, SegmenterError)):
            aligner.align(audio(1.0), [])

    def test_stage_failure_provenance(self):
        """Stage failures carry identifiable provenance."""
        seg = QuranRecitationSegmenter(backend_factory=lambda: _SilentBackend())
        aligner = HybridQuranAligner(
            segmenter=seg,
            forced_aligner=Wav2Vec2ForcedAligner(
                logits_provider=make_logits_provider(np.zeros((5, 3), dtype=np.float32)),
            ),
        )
        with pytest.raises(NoBreathGroupsError, match="no speech groups"):
            aligner.align(audio(1.0), [])

    def test_results_have_provenance(self):
        aligner = self._make_aligner()
        buf = audio(2.0)
        result = aligner.align(buf, [([1, 2], ["a", "b"]), ([3, 4], ["c", "d"])])
        for span in result.spans:
            assert span.provenance is not None
            assert span.provenance != ""
        aligner.close()

    def test_result_timestamps_finite(self):
        aligner = self._make_aligner()
        buf = audio(2.0)
        result = aligner.align(buf, [([1, 2], ["a", "b"]), ([3, 4], ["c", "d"])])
        for span in result.spans:
            assert math.isfinite(span.start)
            assert math.isfinite(span.end)
        aligner.close()

    def test_result_timestamps_monotonic(self):
        aligner = self._make_aligner()
        buf = audio(2.0)
        result = aligner.align(buf, [([1, 2], ["a", "b"]), ([3, 4], ["c", "d"])])
        # Spans may have gaps between breath groups, but within the result
        # each span must start at or after the previous span's end.
        for i in range(1, len(result.spans)):
            assert result.spans[i].start >= result.spans[i - 1].end
        result.validate(buf.duration)  # validate passes → monotonic per contract
        aligner.close()

    def test_result_intervals_bounded(self):
        aligner = self._make_aligner()
        buf = audio(2.0)
        result = aligner.align(buf, [([1, 2], ["a", "b"]), ([3, 4], ["c", "d"])])
        for span in result.spans:
            assert span.start >= 0
            assert span.end <= buf.duration
        aligner.close()

    def test_no_silent_fallback_to_greedy(self):
        """Without reference_aligner, reference_count must be 0."""
        seg = QuranRecitationSegmenter(
            backend_factory=lambda: lambda a: [(0.0, 0.5)],
        )
        fa = Wav2Vec2ForcedAligner(
            logits_provider=make_logits_provider(_aligned_logits(16, 0, [1, 2])),
            config=Wav2Vec2Config(blank_id=0, max_frames=1000, max_tokens=1000),
        )
        aligner = HybridQuranAligner(segmenter=seg, forced_aligner=fa)
        buf = audio(2.0)
        result = aligner.align(buf, [([1, 2], ["a", "b"])])
        assert result.metadata["reference_groups"] == 0
        assert result.metadata["zipformer_role"] == "not_configured"
        assert result.metadata["zipformer_evidence_fused_into_final_spans"] is None
        aligner.close()

    def test_reference_stage_tracks_usage(self):
        """Expose Zipformer reference evidence without claiming final-span fusion."""
        aligner = self._make_aligner(
            reference_aligner=ZipformerNeuralAligner(
                make_evidence(),
                backend_factory=lambda: _FakeZipformerBackend([PhonemeEmission(17, 2, 4, 0.9)]),
            ),
        )
        buf = audio(2.0)
        result = aligner.align(buf, [([1, 2], ["a", "b"]), ([3, 4], ["c", "d"])])
        assert result.metadata["reference_groups"] == 2
        assert result.metadata["zipformer_role"] == "unaligned_phoneme_emissions"
        assert result.metadata["zipformer_reference_alignment_completed"] is False
        assert (
            result.metadata["zipformer_reference_alignment_status"]
            == "blocked_missing_gated_phoneme_map"
        )
        assert result.metadata["zipformer_evidence_fused_into_final_spans"] is False
        assert result.metadata["zipformer_and_wav2vec2_vocabularies_are_independent"] is True
        assert result.metadata["zipformer_reference_evidence"][0]["emissions"][0] == {
            "token_id": 17,
            "start_frame": 2,
            "end_frame": 4,
            "score": 0.9,
        }
        assert {span.token for span in result.spans} <= {"a", "b", "c", "d"}
        aligner.close()

    def test_repeated_calls(self):
        """Provider caching across repeated pipeline calls."""
        calls = [0]

        def provider(samples, sr):
            calls[0] += 1
            return _aligned_logits(32, 0, [1, 2, 3, 4])

        aligner = HybridQuranAligner(
            segmenter=QuranRecitationSegmenter(
                backend_factory=lambda: lambda a: [(0.0, 0.5), (0.6, 1.0)],
            ),
            forced_aligner=Wav2Vec2ForcedAligner(
                logits_provider=provider,
                config=Wav2Vec2Config(blank_id=0, max_frames=1000, max_tokens=1000),
            ),
        )
        buf = audio(2.0)
        aligner.align(buf, [([1, 2], ["a", "b"]), ([3, 4], ["c", "d"])])
        aligner.align(buf, [([1, 2], ["a", "b"]), ([3, 4], ["c", "d"])])
        assert calls[0] >= 4  # 2 groups × 2 calls
        aligner.close()

    def test_close(self):
        aligner = self._make_aligner()
        aligner.close()  # should not raise


class _TrackingBackend:
    def __init__(self, stages, label):
        self.stages = stages
        self.label = label

    def __call__(self, audio):
        self.stages.append(self.label)
        return [(0.0, 0.5), (0.6, 1.0)]


class _TrackingRef:
    def __init__(self, stages):
        self.stages = stages

    def __call__(self, audio, group):
        self.stages.append("reference")
        return []  # empty emissions

    def close(self):
        pass


class _TrackingLogits:
    def __init__(self, stages, logits):
        self.stages = stages
        self.logits = logits

    def __call__(self, samples, sr):
        self.stages.append("ctc")
        return self.logits


class TestHybridResultValidation:
    def test_validate_passes(self):
        groups = (BreathGroup(0.0, 0.5),)
        spans = (AlignmentSpan("a", 0.1, 0.4, 0.9, "wav2vec2-ctc"),)
        result = HybridAlignmentResult(groups, spans)
        result.validate(1.0)

    def test_validate_rejects_non_finite_start(self):
        """AlignmentSpan constructor rejects NaN."""
        with pytest.raises(ValueError):
            AlignmentSpan("a", float("nan"), 0.4, 0.9, "wav2vec2-ctc")

    def test_validate_rejects_non_monotonic(self):
        groups = (BreathGroup(0.0, 0.5),)
        spans = (
            AlignmentSpan("a", 0.1, 0.5, 0.9, "wav2vec2-ctc"),
            AlignmentSpan("b", 0.2, 0.4, 0.9, "wav2vec2-ctc"),  # start < previous end
        )
        result = HybridAlignmentResult(groups, spans)
        with pytest.raises(AlignmentError):
            result.validate(1.0)

    def test_validate_rejects_out_of_bounds(self):
        groups = (BreathGroup(0.0, 0.5),)
        spans = (AlignmentSpan("a", 0.1, 2.0, 0.9, "wav2vec2-ctc"),)
        result = HybridAlignmentResult(groups, spans)
        with pytest.raises(AlignmentError):
            result.validate(1.0)

    def test_validate_rejects_zero_duration(self):
        with pytest.raises(AlignmentError):
            HybridAlignmentResult((), ()).validate(0.0)

    def test_validate_rejects_non_finite_duration(self):
        with pytest.raises(AlignmentError):
            HybridAlignmentResult((), ()).validate(float("nan"))


# ---------------------------------------------------------------------------
# E. Import isolation
# ---------------------------------------------------------------------------


class TestImportIsolation:
    """Importing munajjam must NOT import heavy neural dependencies."""

    def test_top_level_import_clean(self):
        import munajjam  # noqa: F401

        for mod in HEAVY_MODULES:
            assert mod not in sys.modules, f"heavy module {mod} loaded on top-level import"

    def test_hybrid_aligner_import_clean(self):
        import munajjam.hybrid_aligner  # noqa: F401

        for mod in HEAVY_MODULES:
            assert mod not in sys.modules, f"heavy module {mod} loaded on hybrid import"

    def test_subprocess_import_isolation(self):
        """Subprocess import-isolation check: heavy modules must not be pulled in."""
        script = textwrap.dedent(
            """\
            import sys
            # Inject a finder that raises if a heavy module is imported
            HEAVY = {"torch", "transformers", "whisperx", "onnxruntime",
                     "sherpa_onnx", "faster_whisper"}
            class Blocker:
                def find_spec(self, name, path=None, target=None):
                    if name in HEAVY:
                        raise ImportError(f"heavy module {name} must not be imported")
            sys.meta_path.insert(0, Blocker())
            import munajjam
            import munajjam.hybrid_aligner
            print("IMPORT_OK")
            """
        )
        env = {
            **os.environ,
            "PYTHONPATH": os.path.join(os.path.dirname(__file__), "..", "..", "munajjam"),
        }
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, f"import failed:\n{result.stderr}"
        assert "IMPORT_OK" in result.stdout

    def test_no_cuda_selection(self):
        """No CUDA-related module should be loaded after import."""
        import munajjam.hybrid_aligner  # noqa: F401

        cuda_mods = [m for m in sys.modules if "cuda" in m.lower()]
        cuda_mods = [m for m in cuda_mods if m not in ("ctypes", "typing", "numpy")]
        assert cuda_mods == [], f"CUDA modules loaded: {cuda_mods}"


# ---------------------------------------------------------------------------
# E. Security
# ---------------------------------------------------------------------------


class TestSecurity:
    """Static checks for secrets and eager imports."""

    def test_no_hardcoded_secrets(self):
        import pathlib

        pkg_dir = pathlib.Path(__file__).parents[3] / "munajjam" / "munajjam" / "hybrid_aligner"
        secret_patterns = ["api_key", "apikey", "password", "token", "secret"]
        for f in pkg_dir.glob("**/*.py"):
            content = f.read_text()
            for pat in secret_patterns:
                assert f'"{pat}="' not in content, f"possible hardcoded secret in {f}"
                assert f"'{pat}='" not in content, f"possible hardcoded secret in {f}"


# ---------------------------------------------------------------------------
# Legacy regression
# ---------------------------------------------------------------------------


class TestLegacyRegression:
    """Existing Munajjam Aligner behavior must remain unchanged."""

    def test_existing_aligner_importable(self):
        from munajjam.core import Aligner  # noqa: F401

    def test_legacy_strategies_present(self):
        from munajjam.core import Aligner

        for strategy in ["greedy", "dp", "hybrid", "auto"]:
            aligner = Aligner("__test__", strategy=strategy, energy_snap=False)
            assert aligner.strategy == strategy

    def test_legacy_strategies_unchanged(self):
        """All four legacy strategies must still be available."""
        from munajjam.core import Aligner

        for strategy in ["greedy", "dp", "hybrid", "auto"]:
            aligner = Aligner("test.wav", strategy=strategy, energy_snap=False)
            assert aligner.strategy == strategy
