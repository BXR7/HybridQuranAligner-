"""Opt-in orchestration for the tripartite hybrid aligner."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from munajjam.exceptions import InvalidProviderOutputError, ModelUnavailableError
from munajjam.hybrid_aligner.model_manager import ModelManager, ModelSpec
from munajjam.hybrid_aligner.neural_aligner import ZipformerEvidence, ZipformerNeuralAligner
from munajjam.hybrid_aligner.recitation_segmenter import (
    EnergyBreathSegmenter,
    QuranRecitationSegmenter,
)
from munajjam.hybrid_aligner.types import AlignmentSpan, AudioBuffer, HybridAlignmentResult
from munajjam.hybrid_aligner.wav2vec2_aligner import (
    TransformersWav2Vec2LogitsProvider,
    Wav2Vec2ForcedAligner,
    load_audio_file,
)


class HybridQuranAligner:
    """Explicit, opt-in pipeline; existing Munajjam strategies remain unchanged."""

    def __init__(
        self,
        *,
        segmenter: QuranRecitationSegmenter | None = None,
        reference_aligner: ZipformerNeuralAligner | None = None,
        forced_aligner: Wav2Vec2ForcedAligner | None = None,
    ) -> None:
        self.segmenter = segmenter or QuranRecitationSegmenter()
        self.reference_aligner = reference_aligner
        self.forced_aligner = forced_aligner or Wav2Vec2ForcedAligner()

    @classmethod
    def from_pretrained(
        cls,
        *,
        zipformer_evidence: ZipformerEvidence,
        zipformer_backend_factory: Any,
        cache_dir: str | None = None,
        device: str | None = None,
        allow_download: bool = True,
    ) -> HybridQuranAligner:
        """Construct only a fully specified real runtime; never install mocks.

        The Zipformer backend factory is deliberately required because its
        gated artifact's executable interface must be supplied by the verified
        artifact runtime. Missing access therefore fails closed.
        """
        if zipformer_backend_factory is None:
            raise ModelUnavailableError("a verified Zipformer backend is required")
        manager = ModelManager(cache_dir)
        # The manager validates metadata and caches the exact artifact set. It
        # never fabricates an ONNX interface from filenames.
        manager.resolve(
            ModelSpec(
                repository=zipformer_evidence.repository,
                revision=zipformer_evidence.revision,
                files=(
                    "config.json",
                    "tokens.txt",
                    "phoneme_units.json",
                    "ordered_quran_phonemes.json",
                ),
            ),
            allow_download=allow_download,
        )
        segmenter = QuranRecitationSegmenter(
            loader=lambda audio, rate: load_audio_file(audio, rate),
            backend_factory=lambda: EnergyBreathSegmenter(),
        )
        reference = ZipformerNeuralAligner(
            zipformer_evidence,
            backend_factory=zipformer_backend_factory,
        )
        provider = TransformersWav2Vec2LogitsProvider(device=device, cache_dir=cache_dir)
        forced = Wav2Vec2ForcedAligner(logits_provider=provider)
        return cls(segmenter=segmenter, reference_aligner=reference, forced_aligner=forced)

    def align(
        self, audio: Any, targets: Sequence[tuple[Sequence[int], Sequence[str]]]
    ) -> HybridAlignmentResult:
        buffer = (
            audio
            if isinstance(audio, AudioBuffer)
            else self.segmenter._loader(audio, self.segmenter.config.sample_rate)
        )
        groups = self.segmenter.segment(buffer)
        if len(targets) != len(groups):
            raise InvalidProviderOutputError("one reference target is required per breath group")
        spans: list[AlignmentSpan] = []
        reference_count = 0
        for index, (group, target) in enumerate(zip(groups, targets, strict=True)):
            token_ids, token_texts = target
            if self.reference_aligner is not None:
                self.reference_aligner.align(buffer, group)
                reference_count += 1
            next_start = groups[index + 1].start if index + 1 < len(groups) else None
            spans.extend(
                self.forced_aligner.align_group(
                    buffer, group, token_ids, token_texts, next_group_start=next_start
                )
            )
        result = HybridAlignmentResult(
            groups,
            tuple(spans),
            metadata={"reference_groups": reference_count, "provider": "wav2vec2-ctc"},
        )
        result.validate(buffer.duration)
        return result

    def close(self) -> None:
        self.segmenter.close()
        if self.reference_aligner is not None:
            self.reference_aligner.close()
        self.forced_aligner.close()
