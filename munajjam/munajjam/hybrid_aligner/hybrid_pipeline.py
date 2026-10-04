"""Opt-in orchestration for the tripartite hybrid aligner."""

from __future__ import annotations

from collections.abc import Callable, Sequence
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
from munajjam.hybrid_aligner.zipformer_backend import (
    ZIPFORMER_BLANK_ID,
    ZIPFORMER_FRAME_DURATION_SEC,
    ZIPFORMER_MODEL,
    ZIPFORMER_REPOSITORY,
    ZIPFORMER_REVISION,
    ZIPFORMER_TOKEN_SHA256,
    ZipformerOnnxBackend,
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
        zipformer_evidence: ZipformerEvidence | None = None,
        zipformer_backend_factory: Callable[[], Any] | None = None,
        cache_dir: str | None = None,
        device: str | None = None,
        allow_download: bool = True,
    ) -> HybridQuranAligner:
        """Construct only a fully specified real runtime; never install mocks.

        An explicit backend factory remains available for deterministic tests,
        but normal production construction instantiates ``ZipformerOnnxBackend``
        from the verified local artifact directory.
        """
        manager = ModelManager(cache_dir)
        zip_dir = manager.resolve(
            ModelSpec(
                repository=ZIPFORMER_REPOSITORY,
                revision=ZIPFORMER_REVISION,
                files=(
                    "config.json",
                    "tokens.txt",
                    "phoneme_units.json",
                    "ordered_quran_phonemes.json",
                    "quran_text2phoneme.json",
                    "packing_front.json",
                    "packing_back.json",
                    "decode_with_confidence.py",
                    "export_quran_streaming_onnx.py",
                    "quran_per_eval.py",
                    ZIPFORMER_MODEL,
                ),
                hashes={"tokens.txt": ZIPFORMER_TOKEN_SHA256},
            ),
            allow_download=allow_download,
        )
        evidence = zipformer_evidence or ZipformerEvidence(
            repository=ZIPFORMER_REPOSITORY,
            revision=ZIPFORMER_REVISION,
            approved=True,
            vocabulary_size=251,
            blank_id=ZIPFORMER_BLANK_ID,
            sample_rate=16_000,
            feature_kind="kaldi-fbank",
            token_table_sha256=ZIPFORMER_TOKEN_SHA256,
        )
        if evidence.repository != ZIPFORMER_REPOSITORY or evidence.revision != ZIPFORMER_REVISION:
            raise ModelUnavailableError(
                "Zipformer evidence does not match the pinned production artifact"
            )
        backend_factory = zipformer_backend_factory or (lambda: ZipformerOnnxBackend(zip_dir))
        segmenter = QuranRecitationSegmenter(
            loader=lambda audio, rate: load_audio_file(audio, rate),
            backend_factory=lambda: EnergyBreathSegmenter(),
        )
        reference = ZipformerNeuralAligner(
            evidence,
            backend_factory=backend_factory,
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
        reference_evidence: list[dict[str, Any]] = []
        for index, (group, target) in enumerate(zip(groups, targets, strict=True)):
            token_ids, token_texts = target
            if self.reference_aligner is not None:
                emissions = self.reference_aligner.align(buffer, group)
                reference_evidence.append(
                    {
                        "group_index": index,
                        "role": "reference_phoneme_evidence_not_fused",
                        "frame_duration_sec": ZIPFORMER_FRAME_DURATION_SEC,
                        "emissions": [
                            {
                                "token_id": item.token_id,
                                "start_frame": item.start_frame,
                                "end_frame": item.end_frame,
                                "score": item.score,
                            }
                            for item in emissions
                        ],
                    }
                )
            next_start = groups[index + 1].start if index + 1 < len(groups) else None
            spans.extend(
                self.forced_aligner.align_group(
                    buffer, group, token_ids, token_texts, next_group_start=next_start
                )
            )
        result = HybridAlignmentResult(
            groups,
            tuple(spans),
            metadata={
                "provider": "wav2vec2-ctc",
                "zipformer_role": (
                    "canonical_phoneme_reference_evidence"
                    if self.reference_aligner is not None
                    else "not_configured"
                ),
                "zipformer_evidence_fused_into_final_spans": (
                    False if self.reference_aligner is not None else None
                ),
                "reference_groups": len(reference_evidence),
                "zipformer_reference_evidence": reference_evidence,
                "zipformer_and_wav2vec2_vocabularies_are_independent": True,
            },
        )
        result.validate(buffer.duration)
        return result

    def close(self) -> None:
        self.segmenter.close()
        if self.reference_aligner is not None:
            self.reference_aligner.close()
        self.forced_aligner.close()
