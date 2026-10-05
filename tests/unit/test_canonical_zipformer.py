from __future__ import annotations

import numpy as np
import pytest
from munajjam.core.arabic import normalize_arabic
from munajjam.exceptions import InvalidProviderOutputError
from munajjam.hybrid_aligner import (
    AlignmentSpan,
    AudioBuffer,
    BreathGroup,
    CanonicalQuranReferenceProvider,
    CanonicalReferenceRequest,
    PhonemeEmission,
    ZipformerEvidence,
    ZipformerNeuralAligner,
    align_zipformer_to_reference,
)
from munajjam.hybrid_aligner.canonical_zipformer import alignment_evidence
from munajjam.hybrid_aligner.zipformer_backend import ZipformerOnnxBackend


def _table(*pairs: tuple[int, str]) -> list[str]:
    result = [f"unused-{i}" for i in range(251)]
    result[250] = "<blank>"
    for index, token in pairs:
        result[index] = token
    return result


def _reference():
    return CanonicalQuranReferenceProvider().get_reference(
        CanonicalReferenceRequest(1, 1, 1, "hafs")
    )


def test_public_table_parser_accepts_composites_and_blank():
    entries = [f"unused-{i} {i}" for i in range(251)]
    entries[46] = "بَ 46"
    entries[148] = "ااۜ 148"
    entries[250] = "<blank> 250"
    parsed = ZipformerOnnxBackend._parse_token_table("\n".join(entries))
    assert parsed[46] == "بَ"
    assert parsed[148] == "ااۜ"
    assert parsed[250] == "<blank>"


def test_normalization_decodes_composite_symbols_into_canonical_space():
    table = _table((46, "بَ"), (148, "ااۜ"))
    assert normalize_arabic(table[46]) == "ب"
    assert normalize_arabic(table[148]) == "اا"
    assert "ۜ" not in normalize_arabic(table[148])


def test_exact_canonical_match_preserves_frames_scores_and_provenance():
    reference = _reference()
    normalized = normalize_arabic(reference.text).replace(" ", "")
    table = [f"unused-{i}" for i in range(251)]
    table[250] = "<blank>"
    token_by_char = {}
    for char in normalized:
        if char not in token_by_char:
            token_by_char[char] = len(token_by_char)
            table[token_by_char[char]] = char
    emissions = [
        PhonemeEmission(token_by_char[char], i, i + 1, 0.9)
        for i, char in enumerate(normalized)
    ]
    result = align_zipformer_to_reference(emissions, table, reference)
    assert result.accepted
    assert result.normalized_cost == 0
    assert result.match_coverage == 1
    evidence = alignment_evidence(
        result, BreathGroup(0.0, 2.0), frame_duration_sec=0.04
    )
    assert evidence["algorithm"] == "zipformer-character-dp-v1"
    assert all(cell["operation"] == "MATCH" for cell in evidence["cells"])
    first = evidence["cells"][0]
    assert first["token_id"] == emissions[0].token_id
    assert first["start_frame"] == 0
    assert first["canonical_provenance"]["source"]


def test_dp_exposes_match_substitute_insert_delete():
    from dataclasses import replace

    reference = _reference()
    tiny = replace(
        reference, text="ابت", parts=(replace(reference.parts[0], text="ابت"),)
    )
    table = _table((1, "ا"), (2, "ب"), (3, "ث"), (4, "ج"))
    emissions = [
        PhonemeEmission(1, 0, 1, 0.9),
        PhonemeEmission(4, 1, 2, 0.8),
        PhonemeEmission(3, 2, 3, 0.7),
    ]
    result = align_zipformer_to_reference(
        emissions, table, tiny, quality_threshold=1.0, coverage_threshold=0.0
    )
    operations = {cell.operation for cell in result.cells}
    assert {"MATCH", "SUBSTITUTE"} <= operations
    longer = replace(
        reference, text="ابتج", parts=(replace(reference.parts[0], text="ابتج"),)
    )
    deletion_result = align_zipformer_to_reference(
        emissions, table, longer, quality_threshold=1.0, coverage_threshold=0.0
    )
    assert "DELETE" in {cell.operation for cell in deletion_result.cells}
    insertion_result = align_zipformer_to_reference(
        emissions + [PhonemeEmission(2, 3, 4, 0.7)],
        table,
        tiny,
        quality_threshold=1.0,
        coverage_threshold=0.0,
    )
    assert "INSERT" in {cell.operation for cell in insertion_result.cells}


def test_low_quality_alignment_fails_closed():
    with pytest.raises(InvalidProviderOutputError, match="quality below threshold"):
        align_zipformer_to_reference(
            [PhonemeEmission(1, 0, 1, 0.1)], _table((1, "ب")), _reference()
        )


def test_public_evidence_requires_token_table_before_fusion():
    class Backend:
        def __call__(self, audio, group):
            return [PhonemeEmission(1, 0, 1, 0.9)]

    evidence = ZipformerEvidence(
        "Alimalas/munajjam-onnx-models",
        "5dbab4db48a88f5a2a76ead282b2bc3d4b958ee0",
        True,
    )
    aligner = ZipformerNeuralAligner(evidence, backend_factory=Backend)
    with pytest.raises(Exception, match="token table"):
        aligner.token_table()


def test_public_character_dp_fuses_into_hybrid_result():
    class Segmenter:
        def _loader(self, value, sample_rate):
            return value

        def segment(self, value):
            return (BreathGroup(0.0, 1.0),)

        def close(self):
            pass

    class Backend:
        def __init__(self):
            table = [f"unused-{i}" for i in range(251)]
            table[1], table[2], table[3], table[250] = "ا", "ب", "ت", "<blank>"
            self.token_table = tuple(table)

        def __call__(self, audio, group):
            return [PhonemeEmission(i, i - 1, i, 0.9) for i in (1, 2, 3)]

    class ForcedAligner:
        def encode_text(self, text):
            return [1, 2, 3], ["ا", "ب", "ت"]

        def align_group(self, *args, **kwargs):
            return (AlignmentSpan("ا", 0.1, 0.2, 0.9, "wav2vec2-ctc"),)

        def close(self):
            pass

    class Provider:
        def get_reference(self, request):
            from munajjam.hybrid_aligner.reference import (
                CanonicalReference,
                CanonicalReferencePart,
            )

            part = CanonicalReferencePart("ayah", "ابت", "unit-test", "a" * 64, 1, 1)
            return CanonicalReference(
                "hafs", (part,), "unit", "b" * 64, "ابت", "c" * 64
            )

    from munajjam.hybrid_aligner.hybrid_pipeline import HybridQuranAligner

    ref = ZipformerNeuralAligner(
        ZipformerEvidence(
            "Alimalas/munajjam-onnx-models",
            "5dbab4db48a88f5a2a76ead282b2bc3d4b958ee0",
            True,
        ),
        backend_factory=Backend,
    )
    pipeline = HybridQuranAligner(
        segmenter=Segmenter(),
        reference_aligner=ref,
        forced_aligner=ForcedAligner(),
        canonical_reference_provider=Provider(),
        require_canonical_references=True,
    )
    result = pipeline.align(
        AudioBuffer(np.zeros(16_000, dtype=np.float32), 16_000),
        references=[CanonicalReferenceRequest(1, 1, 1)],
    )
    assert result.metadata["zipformer_reference_alignment_completed"] is True
    assert result.metadata["zipformer_evidence_fused_into_final_spans"] is True
    assert (
        result.metadata["zipformer_reference_alignment_status"]
        == "canonical_character_dp_verified"
    )
    assert result.spans[0].provenance == "wav2vec2-ctc+zipformer-character-dp"
    pipeline.close()
