"""Opt-in hybrid alignment API with dependency-free imports."""
from munajjam.hybrid_aligner.hybrid_pipeline import HybridQuranAligner
from munajjam.hybrid_aligner.neural_aligner import ZipformerEvidence, ZipformerNeuralAligner
from munajjam.hybrid_aligner.recitation_segmenter import QuranRecitationSegmenter, SegmenterConfig
from munajjam.hybrid_aligner.types import (
    AlignmentSpan,
    AudioBuffer,
    BreathGroup,
    HybridAlignmentResult,
)
from munajjam.hybrid_aligner.wav2vec2_aligner import Wav2Vec2Config, Wav2Vec2ForcedAligner

__all__ = [
    "AlignmentSpan",
    "AudioBuffer",
    "BreathGroup",
    "HybridAlignmentResult",
    "HybridQuranAligner",
    "QuranRecitationSegmenter",
    "SegmenterConfig",
    "Wav2Vec2Config",
    "Wav2Vec2ForcedAligner",
    "ZipformerEvidence",
    "ZipformerNeuralAligner",
]
