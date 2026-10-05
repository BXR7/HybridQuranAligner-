"""Opt-in orchestration for the tripartite hybrid aligner."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from munajjam.exceptions import InvalidProviderOutputError, ModelUnavailableError
from munajjam.hybrid_aligner.canonical_zipformer import (
    align_zipformer_to_reference,
    alignment_evidence,
)
from munajjam.hybrid_aligner.model_manager import ModelManager, ModelSpec
from munajjam.hybrid_aligner.neural_aligner import ZipformerEvidence, ZipformerNeuralAligner
from munajjam.hybrid_aligner.recitation_segmenter import (
    QuranRecitationSegmenter,
    TransformersRecitationSegmenterBackend,
)
from munajjam.hybrid_aligner.reference import (
    CanonicalQuranReferenceProvider,
    CanonicalReference,
    CanonicalReferenceRequest,
)
from munajjam.hybrid_aligner.types import AlignmentSpan, AudioBuffer, HybridAlignmentResult
from munajjam.hybrid_aligner.wav2vec2_aligner import (
    TransformersWav2Vec2LogitsProvider,
    Wav2Vec2ForcedAligner,
    load_audio_file,
)
from munajjam.hybrid_aligner.zipformer_backend import (
    PUBLIC_ZIPFORMER_MODEL,
    PUBLIC_ZIPFORMER_REPOSITORY,
    PUBLIC_ZIPFORMER_REVISION,
    PUBLIC_ZIPFORMER_TOKENS,
    ZIPFORMER_BLANK_ID,
    ZIPFORMER_FRAME_DURATION_SEC,
    ZIPFORMER_TOKEN_SHA256,
    SherpaZipformerBackend,
)


def _load_production_audio(audio: Any, sample_rate: int) -> AudioBuffer:
    """Load paths or wrap samples without forcing model imports at construction."""
    if isinstance(audio, AudioBuffer):
        return audio
    if isinstance(audio, (str, Path)):
        return load_audio_file(audio, sample_rate)
    try:
        samples = np.asarray(audio, dtype=np.float32)
    except Exception as exc:
        raise InvalidProviderOutputError(
            "audio must be a path, AudioBuffer, or mono samples"
        ) from exc
    return AudioBuffer(samples=samples, sample_rate=sample_rate)


class HybridQuranAligner:
    """Explicit, opt-in pipeline; existing Munajjam strategies remain unchanged.

    ``references`` assigns an explicit canonical verse range to each detected
    physical breath group. The aligner never infers verse numbers from duration,
    and production construction rejects legacy caller-supplied token pairs.
    """

    def __init__(
        self,
        *,
        segmenter: QuranRecitationSegmenter | None = None,
        reference_aligner: ZipformerNeuralAligner | None = None,
        forced_aligner: Wav2Vec2ForcedAligner | None = None,
        canonical_reference_provider: CanonicalQuranReferenceProvider | None = None,
        require_canonical_references: bool = False,
    ) -> None:
        self.segmenter = segmenter or QuranRecitationSegmenter()
        self.reference_aligner = reference_aligner
        self.forced_aligner = forced_aligner or Wav2Vec2ForcedAligner()
        self.canonical_reference_provider = canonical_reference_provider
        self.require_canonical_references = require_canonical_references

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
        """Construct a production pipeline with pinned, lazy model factories.

        The method itself performs no downloads and imports no heavyweight ML
        runtime. Zipformer weights, the segmenter model, and the Wav2Vec2 model
        are loaded on the first inference that needs each component.
        """
        manager = ModelManager(cache_dir)
        zipformer_spec = ModelSpec(
            repository=PUBLIC_ZIPFORMER_REPOSITORY,
            revision=PUBLIC_ZIPFORMER_REVISION,
            files=(PUBLIC_ZIPFORMER_TOKENS, PUBLIC_ZIPFORMER_MODEL),
            hashes={PUBLIC_ZIPFORMER_TOKENS: ZIPFORMER_TOKEN_SHA256},
        )
        evidence = zipformer_evidence or ZipformerEvidence(
            repository=PUBLIC_ZIPFORMER_REPOSITORY,
            revision=PUBLIC_ZIPFORMER_REVISION,
            approved=True,
            vocabulary_size=251,
            blank_id=ZIPFORMER_BLANK_ID,
            sample_rate=16_000,
            feature_kind="kaldi-fbank",
            token_table_sha256=ZIPFORMER_TOKEN_SHA256,
        )
        if (
            evidence.repository != PUBLIC_ZIPFORMER_REPOSITORY
            or evidence.revision != PUBLIC_ZIPFORMER_REVISION
        ):
            raise ModelUnavailableError(
                "Zipformer evidence does not match the pinned production artifact"
            )

        def make_zipformer_backend() -> Any:
            if zipformer_backend_factory is not None:
                return zipformer_backend_factory()
            zip_dir = manager.resolve(zipformer_spec, allow_download=allow_download)
            return SherpaZipformerBackend(zip_dir)

        segmenter = QuranRecitationSegmenter(
            loader=_load_production_audio,
            backend_factory=lambda: TransformersRecitationSegmenterBackend(
                device=device, cache_dir=cache_dir
            ),
        )
        reference = ZipformerNeuralAligner(evidence, backend_factory=make_zipformer_backend)
        forced = Wav2Vec2ForcedAligner(
            logits_provider_factory=lambda: TransformersWav2Vec2LogitsProvider(
                device=device, cache_dir=cache_dir
            )
        )
        return cls(
            segmenter=segmenter,
            reference_aligner=reference,
            forced_aligner=forced,
            canonical_reference_provider=CanonicalQuranReferenceProvider(),
            require_canonical_references=True,
        )

    @staticmethod
    def _reference_metadata(reference: CanonicalReference) -> dict[str, Any]:
        return {
            "riwaya": reference.riwaya,
            "parts": [
                {
                    "kind": part.kind,
                    "surah_id": part.surah_id,
                    "ayah_number": part.ayah_number,
                    "source": part.source,
                    "source_sha256": part.source_sha256,
                }
                for part in reference.parts
            ],
            "source_file": reference.source_file,
            "source_file_sha256": reference.source_file_sha256,
            "text_sha256": reference.text_sha256,
        }

    def align(
        self,
        audio: Any,
        targets: Sequence[tuple[Sequence[int], Sequence[str]]] | None = None,
        *,
        references: Sequence[CanonicalReferenceRequest] | None = None,
    ) -> HybridAlignmentResult:
        if (targets is None) == (references is None):
            raise InvalidProviderOutputError(
                "provide exactly one of canonical references or legacy token targets"
            )
        if self.require_canonical_references and references is None:
            raise InvalidProviderOutputError(
                "production alignment requires canonical verse references, not caller token IDs"
            )
        if references is not None and self.canonical_reference_provider is None:
            raise ModelUnavailableError("no canonical Quran reference provider is configured")

        buffer = (
            audio
            if isinstance(audio, AudioBuffer)
            else self.segmenter._loader(audio, self.segmenter.config.sample_rate)
        )
        groups = self.segmenter.segment(buffer)
        selected_references: list[CanonicalReference] | None = None
        legacy_targets: list[tuple[Sequence[int], Sequence[str]]] | None = None
        if references is not None:
            if len(references) != len(groups):
                raise InvalidProviderOutputError(
                    "one canonical verse reference is required per breath group"
                )
            assert self.canonical_reference_provider is not None
            selected_references = [
                self.canonical_reference_provider.get_reference(request) for request in references
            ]
        else:
            assert targets is not None
            if len(targets) != len(groups):
                raise InvalidProviderOutputError(
                    "one reference target is required per breath group"
                )
            legacy_targets = list(targets)

        spans: list[AlignmentSpan] = []
        reference_evidence: list[dict[str, Any]] = []
        fused_groups = 0
        for index, group in enumerate(groups):
            canonical = selected_references[index] if selected_references is not None else None
            group_alignment = None
            if self.reference_aligner is not None:
                emissions = self.reference_aligner.align(buffer, group)
                evidence = {
                    "group_index": index,
                    "frame_duration_sec": ZIPFORMER_FRAME_DURATION_SEC,
                    "canonical_reference": (
                        self._reference_metadata(canonical) if canonical is not None else None
                    ),
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
                is_public = (
                    canonical is not None
                    and self.reference_aligner.evidence is not None
                    and self.reference_aligner.evidence.repository == PUBLIC_ZIPFORMER_REPOSITORY
                )
                if is_public:
                    group_alignment = align_zipformer_to_reference(
                        emissions, self.reference_aligner.token_table(), canonical
                    )
                    evidence["role"] = "canonical_character_dp_fusion"
                    evidence["alignment"] = alignment_evidence(
                        group_alignment, group, frame_duration_sec=ZIPFORMER_FRAME_DURATION_SEC
                    )
                    fused_groups += 1
                else:
                    evidence["role"] = "unaligned_phoneme_emissions"
                reference_evidence.append(evidence)
            if canonical is not None:
                token_ids, token_texts = self.forced_aligner.encode_text(canonical.text)
            else:
                assert legacy_targets is not None
                token_ids, token_texts = legacy_targets[index]
            next_start = groups[index + 1].start if index + 1 < len(groups) else None
            group_spans = list(
                self.forced_aligner.align_group(
                    buffer, group, token_ids, token_texts, next_group_start=next_start
                )
            )
            if group_alignment is not None:
                group_spans = [
                    AlignmentSpan(
                        span.token,
                        span.start,
                        span.end,
                        span.confidence,
                        "wav2vec2-ctc+zipformer-character-dp",
                    )
                    for span in group_spans
                ]
            spans.extend(group_spans)

        warnings: list[str] = []
        public_fusion_complete = (
            self.reference_aligner is not None
            and selected_references is not None
            and fused_groups == len(groups)
        )
        if self.reference_aligner is not None and not public_fusion_complete:
            warnings.append(
                "Public Zipformer emissions are available, but canonical phoneme-target DP "
                "fusion is not enabled until an authoritative token-target provider is supplied."
            )
        result = HybridAlignmentResult(
            groups,
            tuple(spans),
            metadata={
                "provider": "wav2vec2-ctc",
                "reference_groups": len(reference_evidence),
                "canonical_reference_groups": len(selected_references or ()),
                "canonical_references": (
                    [self._reference_metadata(item) for item in selected_references]
                    if selected_references is not None
                    else []
                ),
                "zipformer_role": (
                    "unaligned_phoneme_emissions"
                    if self.reference_aligner is not None
                    else "not_configured"
                ),
                "zipformer_reference_alignment_completed": public_fusion_complete,
                "zipformer_reference_alignment_status": (
                    (
                        "canonical_character_dp_verified"
                        if public_fusion_complete
                        else "public_model_alignment_pending"
                        if self.reference_aligner.evidence is not None
                        and self.reference_aligner.evidence.repository
                        == PUBLIC_ZIPFORMER_REPOSITORY
                        else "blocked_gated_artifact_access"
                    )
                    if self.reference_aligner is not None
                    else "not_configured"
                ),
                "zipformer_evidence_fused_into_final_spans": (
                    public_fusion_complete if self.reference_aligner is not None else None
                ),
                "zipformer_reference_evidence": reference_evidence,
                "zipformer_and_wav2vec2_vocabularies_are_independent": True,
                "warnings": warnings,
            },
        )
        result.validate(buffer.duration)
        return result

    def close(self) -> None:
        self.segmenter.close()
        if self.reference_aligner is not None:
            self.reference_aligner.close()
        self.forced_aligner.close()
