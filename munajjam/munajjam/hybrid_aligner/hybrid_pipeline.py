"""Opt-in orchestration for the tripartite hybrid aligner."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from munajjam.exceptions import InvalidProviderOutputError
from munajjam.hybrid_aligner.neural_aligner import ZipformerNeuralAligner
from munajjam.hybrid_aligner.recitation_segmenter import QuranRecitationSegmenter
from munajjam.hybrid_aligner.types import AlignmentSpan, AudioBuffer, HybridAlignmentResult
from munajjam.hybrid_aligner.wav2vec2_aligner import Wav2Vec2ForcedAligner


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
