from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from munajjam.exceptions import InvalidProviderOutputError, ModelUnavailableError, QuranDataError
from munajjam.hybrid_aligner import (
    RECITATION_SEGMENTER_REVISION,
    AudioBuffer,
    CanonicalQuranReferenceProvider,
    CanonicalReferenceRequest,
    HybridQuranAligner,
    QuranRecitationSegmenter,
    TransformersRecitationSegmenterBackend,
    VerifiedSpecialPhrase,
    Wav2Vec2Config,
    Wav2Vec2ForcedAligner,
)
from munajjam.hybrid_aligner.recitation_segmenter import _SegmenterRuntime


class _FakeTorch:
    float32 = "float32"

    @staticmethod
    def as_tensor(value, dtype=None):
        return np.asarray(value, dtype=np.float32)


class _FakeProvider:
    def __init__(self):
        self.encoded = []

    def encode_text(self, text):
        self.encoded.append(text)
        return [1], ["q"]

    def __call__(self, samples, sample_rate):
        logits = np.full((16, 2), -1.0, dtype=np.float32)
        logits[:, 0] = 0.0
        logits[5:9, 1] = 10.0
        return logits


def _audio(duration=1.0):
    return AudioBuffer(np.ones(int(16_000 * duration), dtype=np.float32) * 0.1, 16_000)


def test_segmenter_backend_loads_lazily_and_uses_authors_cleaner():
    calls = {"load": 0, "segment": 0, "clean": 0}

    def segment_recitations(waves, model, processor, **kwargs):
        calls["segment"] += 1
        assert len(waves) == 1
        assert kwargs["sample_rate"] == 16_000
        assert kwargs["max_duration_ms"] == 19_995
        assert kwargs["device"] == "cpu"
        return [SimpleNamespace(speech_intervals=[(1000, 5000)], is_complete=True)]

    def clean_speech_intervals(intervals, is_complete, **kwargs):
        calls["clean"] += 1
        assert is_complete is True
        assert kwargs == {
            "min_silence_duration_ms": 30,
            "min_speech_duration_ms": 30,
            "pad_duration_ms": 30,
            "return_seconds": True,
        }
        return SimpleNamespace(clean_speech_intervals=[(0.05, 0.4)])

    runtime = _SegmenterRuntime(
        _FakeTorch(),
        object(),
        object(),
        "cpu",
        "float32",
        segment_recitations,
        clean_speech_intervals,
    )

    def load_runtime():
        calls["load"] += 1
        return runtime

    backend = TransformersRecitationSegmenterBackend(runtime_loader=load_runtime)
    assert backend._runtime is None
    assert backend.revision == RECITATION_SEGMENTER_REVISION
    assert backend(_audio()) == [(0.05, 0.4, None)]
    assert backend(_audio()) == [(0.05, 0.4, None)]
    assert calls == {"load": 1, "segment": 2, "clean": 2}


def test_production_segmenter_default_is_not_energy_vad():
    backend = QuranRecitationSegmenter._default_backend()
    assert isinstance(backend, TransformersRecitationSegmenterBackend)
    assert backend._runtime is None


def test_segmenter_backend_rejects_wrong_sample_rate_before_loading():
    loads = []
    backend = TransformersRecitationSegmenterBackend(runtime_loader=lambda: loads.append(1))
    with pytest.raises(InvalidProviderOutputError, match="16 kHz"):
        backend(AudioBuffer(np.ones(100, dtype=np.float32), 8_000))
    assert not loads


def test_hafs_canonical_reference_contains_source_and_text_digests():
    result = CanonicalQuranReferenceProvider().get_reference(
        CanonicalReferenceRequest(1, 1, 2, "hafs")
    )
    assert [part.ayah_number for part in result.parts] == [1, 2]
    assert all(part.kind == "ayah" for part in result.parts)
    assert result.source_file == "munajjam.data/quran_hafs.json"
    assert len(result.source_file_sha256) == 64
    assert len(result.text_sha256) == 64
    assert result.text == " ".join(part.text for part in result.parts)


def test_basmalah_uses_bundled_canonical_source_and_is_not_duplicated_for_fatiha():
    provider = CanonicalQuranReferenceProvider()
    standalone = provider.get_reference(
        CanonicalReferenceRequest(2, 1, 1, "hafs", include_basmalah=True)
    )
    assert [part.kind for part in standalone.parts] == ["basmalah", "ayah"]
    assert standalone.parts[0].source.endswith("#1:1")
    assert standalone.parts[0].source_sha256 == standalone.source_file_sha256

    fatiha = provider.get_reference(
        CanonicalReferenceRequest(1, 1, 2, "hafs", include_basmalah=True)
    )
    assert [part.kind for part in fatiha.parts] == ["ayah", "ayah"]


def test_istiadhah_requires_a_sourced_phrase_and_preserves_order():
    provider = CanonicalQuranReferenceProvider()
    request = CanonicalReferenceRequest(1, 1, 1, include_istiadhah=True)
    with pytest.raises(ModelUnavailableError, match="source and SHA-256"):
        provider.get_reference(request)

    phrase = VerifiedSpecialPhrase.from_source(
        "istiadhah",
        "أعوذ بالله من الشيطان الرجيم",
        "caller-provided canonical istiadhah source",
    )
    result = provider.get_reference(
        CanonicalReferenceRequest(1, 1, 1, include_istiadhah=True, istiadhah_phrase=phrase)
    )
    assert [part.kind for part in result.parts] == ["istiadhah", "ayah"]
    assert result.parts[0].source == phrase.source
    assert result.parts[0].source_sha256 == phrase.sha256


def test_special_phrase_rejects_tampered_hash_or_wrong_phrase_type():
    phrase = "أعوذ بالله من الشيطان الرجيم"
    with pytest.raises(ValueError, match="SHA-256"):
        VerifiedSpecialPhrase("istiadhah", phrase, "source", "0" * 64)
    with pytest.raises(ValueError, match="does not match"):
        VerifiedSpecialPhrase.from_source("istiadhah", "بسم الله الرحمن الرحيم", "source")
    with pytest.raises(ValueError, match="does not match"):
        VerifiedSpecialPhrase.from_source("istiadhah", f"{phrase} extra text", "source")
    with pytest.raises(ValueError, match="VerifiedSpecialPhrase"):
        CanonicalReferenceRequest(
            1,
            1,
            1,
            include_istiadhah=True,
            istiadhah_phrase="not-a-phrase",  # type: ignore[arg-type]
        )


def test_missing_warsh_placeholder_fails_closed_without_losing_ayah_numbering():
    with pytest.raises(QuranDataError, match="2:254"):
        CanonicalQuranReferenceProvider().get_reference(
            CanonicalReferenceRequest(2, 254, 254, "warsh")
        )


def test_wav2vec_factory_and_canonical_tokenizer_are_lazy():
    provider = _FakeProvider()
    factories = []

    def factory():
        factories.append(1)
        return provider

    aligner = Wav2Vec2ForcedAligner(
        logits_provider_factory=factory,
        config=Wav2Vec2Config(blank_id=0),
    )
    assert factories == []
    assert aligner.encode_text("canonical Quran text") == ([1], ["q"])
    assert factories == [1]
    assert provider.encoded == ["canonical Quran text"]


def test_production_construction_does_not_download_or_load_heavy_models(monkeypatch):
    from munajjam.hybrid_aligner.model_manager import ModelManager

    def unexpected_resolve(*args, **kwargs):
        raise AssertionError("artifact resolution must be deferred")

    monkeypatch.setattr(ModelManager, "resolve", unexpected_resolve)
    aligner = HybridQuranAligner.from_pretrained(
        device="cpu", cache_dir="/tmp/hybrid-models", allow_download=False
    )
    assert aligner.segmenter._backend is None
    assert aligner.reference_aligner._backend is None
    assert aligner.forced_aligner._provider is None
    segmenter_backend = aligner.segmenter._backend_factory()
    assert segmenter_backend.device == "cpu"
    assert segmenter_backend.cache_dir == "/tmp/hybrid-models"
    aligner.close()
