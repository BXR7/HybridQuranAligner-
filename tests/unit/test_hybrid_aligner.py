import numpy as np
import pytest
from munajjam.exceptions import (
    AlignmentError,
    InvalidProviderOutputError,
    ModelUnavailableError,
    NoBreathGroupsError,
)
from munajjam.hybrid_aligner import (
    AudioBuffer,
    BreathGroup,
    QuranRecitationSegmenter,
    Wav2Vec2ForcedAligner,
    ZipformerEvidence,
    ZipformerNeuralAligner,
)


def audio(seconds=1.0):
    return AudioBuffer(np.zeros(int(16_000 * seconds), dtype=np.float32), 16_000)


def test_segmenter_is_lazy_and_validates_groups():
    calls = []
    segmenter = QuranRecitationSegmenter(
        loader=lambda value, rate: value,
        backend_factory=lambda: calls.append("load") or (lambda value: [(0.1, 0.5, 0.9)]),
    )
    assert calls == []
    assert segmenter.segment(audio())[0] == BreathGroup(0.1, 0.5, 0.9)
    assert calls == ["load"]


def test_segmenter_rejects_empty_output():
    segmenter = QuranRecitationSegmenter(loader=lambda value, rate: value, backend_factory=lambda: lambda value: [])
    with pytest.raises(NoBreathGroupsError):
        segmenter.segment(audio())


def test_segmenter_rejects_overlap():
    segmenter = QuranRecitationSegmenter(loader=lambda value, rate: value, backend_factory=lambda: lambda value: [(0.1, 0.4), (0.3, 0.5)])
    with pytest.raises(InvalidProviderOutputError):
        segmenter.segment(audio())


def test_zipformer_is_fail_closed_without_evidence():
    with pytest.raises(ModelUnavailableError):
        ZipformerNeuralAligner().align(audio(), BreathGroup(0.1, 0.5))


def test_zipformer_evidence_rejects_unapproved_artifact():
    with pytest.raises(ValueError):
        ZipformerEvidence("Quran-Lab/zipformer_p-arabic-v3", "0" * 40, False)


def test_wav2vec_ctc_stays_inside_breath_group():
    def logits(samples, rate):
        values = np.full((8, 3), -8.0)
        values[:, 0] = 0.0
        values[2:5, 1] = 8.0
        return values

    aligner = Wav2Vec2ForcedAligner(logits_provider=logits)
    spans = aligner.align_group(audio(), BreathGroup(0.2, 0.6), [1], ["ا"])
    assert len(spans) == 1
    assert 0.2 <= spans[0].start < spans[0].end <= 0.8


def test_wav2vec_rejects_out_of_vocabulary_tokens():
    aligner = Wav2Vec2ForcedAligner(logits_provider=lambda samples, rate: np.zeros((4, 2)))
    with pytest.raises(InvalidProviderOutputError):
        aligner.align_group(audio(), BreathGroup(0.1, 0.4), [2], ["bad"])


def test_wav2vec_rejects_impossible_target():
    aligner = Wav2Vec2ForcedAligner(logits_provider=lambda samples, rate: np.zeros((1, 2)))
    with pytest.raises(AlignmentError):
        aligner.align_group(audio(), BreathGroup(0.1, 0.4), [1, 1], ["a", "b"])
