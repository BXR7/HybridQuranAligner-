"""Opt-in hybrid alignment API with dependency-free imports."""

from munajjam.hybrid_aligner.hybrid_pipeline import HybridQuranAligner
from munajjam.hybrid_aligner.model_manager import ModelManager, ModelSpec
from munajjam.hybrid_aligner.neural_aligner import ZipformerEvidence, ZipformerNeuralAligner
from munajjam.hybrid_aligner.recitation_segmenter import QuranRecitationSegmenter, SegmenterConfig
from munajjam.hybrid_aligner.types import (
    AlignmentSpan,
    AudioBuffer,
    BreathGroup,
    HybridAlignmentResult,
    PhonemeEmission,
)
from munajjam.hybrid_aligner.wav2vec2_aligner import (
    WAV2VEC2_REPOSITORY,
    WAV2VEC2_REVISION,
    TransformersWav2Vec2LogitsProvider,
    Wav2Vec2Config,
    Wav2Vec2ForcedAligner,
    load_audio_file,
)

__all__ = [
    "AlignmentSpan",
    "AudioBuffer",
    "BreathGroup",
    "HybridAlignmentResult",
    "HybridQuranAligner",
    "ModelManager",
    "ModelSpec",
    "PhonemeEmission",
    "QuranRecitationSegmenter",
    "SegmenterConfig",
    "Wav2Vec2Config",
    "Wav2Vec2ForcedAligner",
    "TransformersWav2Vec2LogitsProvider",
    "WAV2VEC2_REPOSITORY",
    "WAV2VEC2_REVISION",
    "load_audio_file",
    "ZipformerEvidence",
    "ZipformerNeuralAligner",
]
