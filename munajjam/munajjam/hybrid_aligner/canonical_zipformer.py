"""Canonical character-level DP fusion for public Zipformer emissions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from munajjam.core.arabic import normalize_arabic
from munajjam.exceptions import InvalidProviderOutputError
from munajjam.hybrid_aligner.reference import CanonicalReference
from munajjam.hybrid_aligner.types import BreathGroup, PhonemeEmission


@dataclass(frozen=True, slots=True)
class ZipformerObservedUnit:
    emission_index: int
    token_id: int
    raw_symbol: str
    normalized_symbol: str
    start_frame: int
    end_frame: int
    score: float
    unit_index: int
    unit: str


@dataclass(frozen=True, slots=True)
class CanonicalUnit:
    index: int
    unit: str
    source: str
    source_sha256: str
    part_kind: str
    surah_id: int | None
    ayah_number: int | None


@dataclass(frozen=True, slots=True)
class ZipformerDPCell:
    operation: str
    observed: ZipformerObservedUnit | None
    canonical: CanonicalUnit | None
    cost: float

    @property
    def token_id(self) -> int | None:
        return self.observed.token_id if self.observed else None

    @property
    def start_frame(self) -> int | None:
        return self.observed.start_frame if self.observed else None

    @property
    def end_frame(self) -> int | None:
        return self.observed.end_frame if self.observed else None


@dataclass(frozen=True, slots=True)
class ZipformerDPAlignment:
    cells: tuple[ZipformerDPCell, ...]
    total_cost: float
    normalized_cost: float
    match_coverage: float
    observed_units: int
    canonical_units: int
    quality_threshold: float
    coverage_threshold: float

    @property
    def accepted(self) -> bool:
        return (
            self.normalized_cost <= self.quality_threshold
            and self.match_coverage >= self.coverage_threshold
        )


@dataclass(frozen=True, slots=True)
class _DPState:
    cost: float
    matches: int
    cells: tuple[ZipformerDPCell, ...]


def _canonical_units(reference: CanonicalReference) -> tuple[CanonicalUnit, ...]:
    units: list[CanonicalUnit] = []
    for part in reference.parts:
        normalized = normalize_arabic(part.text).replace(" ", "")
        for char in normalized:
            units.append(
                CanonicalUnit(
                    index=len(units),
                    unit=char,
                    source=part.source,
                    source_sha256=part.source_sha256,
                    part_kind=part.kind,
                    surah_id=part.surah_id,
                    ayah_number=part.ayah_number,
                )
            )
    if not units:
        raise InvalidProviderOutputError("canonical reference has no alignable characters")
    return tuple(units)


def _observed_units(
    emissions: Sequence[PhonemeEmission], token_table: Sequence[str]
) -> tuple[ZipformerObservedUnit, ...]:
    units: list[ZipformerObservedUnit] = []
    for emission_index, emission in enumerate(emissions):
        if emission.token_id < 0 or emission.token_id >= len(token_table):
            raise InvalidProviderOutputError("Zipformer token ID is outside the token table")
        raw = token_table[emission.token_id]
        if raw == "<blank>":
            raise InvalidProviderOutputError("blank token cannot be fused as evidence")
        normalized = normalize_arabic(raw).replace(" ", "")
        for unit_index, unit in enumerate(normalized):
            units.append(
                ZipformerObservedUnit(
                    emission_index=emission_index,
                    token_id=emission.token_id,
                    raw_symbol=raw,
                    normalized_symbol=normalized,
                    start_frame=emission.start_frame,
                    end_frame=emission.end_frame,
                    score=float(emission.score),
                    unit_index=unit_index,
                    unit=unit,
                )
            )
    if not units:
        raise InvalidProviderOutputError("Zipformer emitted no alignable characters")
    return tuple(units)


def align_zipformer_to_reference(
    emissions: Sequence[PhonemeEmission],
    token_table: Sequence[str],
    reference: CanonicalReference,
    *,
    quality_threshold: float = 0.45,
    coverage_threshold: float = 0.70,
) -> ZipformerDPAlignment:
    """Align normalized public-token characters to canonical Quran characters.

    The DP has the four explicit operations required by the production contract:
    MATCH, SUBSTITUTE, INSERT, and DELETE. Token chunks are expanded only for
    comparison; every cell retains the original token ID, symbol, frames, score,
    and canonical provenance. A path is rejected when its normalized edit cost or
    canonical match coverage is below the explicit quality thresholds. The
    defaults are calibrated against the public Al-Fatiha sample: normalized edit
    cost <= 0.45 and canonical match coverage >= 0.70.
    """
    if not 0 <= quality_threshold <= 1 or not 0 <= coverage_threshold <= 1:
        raise ValueError("DP quality thresholds must be between zero and one")
    observed = _observed_units(emissions, token_table)
    canonical = _canonical_units(reference)
    rows, cols = len(observed), len(canonical)
    dp: list[list[_DPState | None]] = [[None] * (cols + 1) for _ in range(rows + 1)]
    dp[0][0] = _DPState(0.0, 0, ())
    for i in range(1, rows + 1):
        prev = dp[i - 1][0]
        assert prev is not None
        obs = observed[i - 1]
        dp[i][0] = _DPState(
            prev.cost + 1.0,
            prev.matches,
            prev.cells + (ZipformerDPCell("INSERT", obs, None, 1.0),),
        )
    for j in range(1, cols + 1):
        prev = dp[0][j - 1]
        assert prev is not None
        can = canonical[j - 1]
        dp[0][j] = _DPState(
            prev.cost + 1.0,
            prev.matches,
            prev.cells + (ZipformerDPCell("DELETE", None, can, 1.0),),
        )
    for i in range(1, rows + 1):
        for j in range(1, cols + 1):
            obs, can = observed[i - 1], canonical[j - 1]
            diagonal = dp[i - 1][j - 1]
            assert diagonal is not None
            same = obs.unit == can.unit
            candidates = [
                _DPState(
                    diagonal.cost + (0.0 if same else 1.0),
                    diagonal.matches + (1 if same else 0),
                    diagonal.cells
                    + (
                        ZipformerDPCell(
                            "MATCH" if same else "SUBSTITUTE", obs, can, 0.0 if same else 1.0
                        ),
                    ),
                )
            ]
            above = dp[i - 1][j]
            assert above is not None
            candidates.append(
                _DPState(
                    above.cost + 1.0,
                    above.matches,
                    above.cells + (ZipformerDPCell("INSERT", obs, None, 1.0),),
                )
            )
            left = dp[i][j - 1]
            assert left is not None
            candidates.append(
                _DPState(
                    left.cost + 1.0,
                    left.matches,
                    left.cells + (ZipformerDPCell("DELETE", None, can, 1.0),),
                )
            )
            # Prefer lower cost, then more exact matches, then diagonal paths.
            dp[i][j] = min(candidates, key=lambda state: (state.cost, -state.matches))
    final = dp[rows][cols]
    assert final is not None
    scale = float(max(rows, cols, 1))
    alignment = ZipformerDPAlignment(
        cells=final.cells,
        total_cost=final.cost,
        normalized_cost=final.cost / scale,
        match_coverage=final.matches / float(cols),
        observed_units=rows,
        canonical_units=cols,
        quality_threshold=quality_threshold,
        coverage_threshold=coverage_threshold,
    )
    if not alignment.accepted:
        raise InvalidProviderOutputError(
            "Zipformer canonical alignment quality below threshold: "
            f"normalized_cost={alignment.normalized_cost:.4f}, "
            f"match_coverage={alignment.match_coverage:.4f}"
        )
    return alignment


def alignment_evidence(
    alignment: ZipformerDPAlignment,
    group: BreathGroup,
    *,
    frame_duration_sec: float,
) -> dict[str, object]:
    """Serialize accepted DP evidence without losing token/timing provenance."""
    cells: list[dict[str, object]] = []
    for cell in alignment.cells:
        observed, canonical = cell.observed, cell.canonical
        item: dict[str, object] = {
            "operation": cell.operation,
            "cost": cell.cost,
            "canonical_unit": canonical.unit if canonical else None,
            "canonical_index": canonical.index if canonical else None,
            "canonical_provenance": (
                {
                    "source": canonical.source,
                    "source_sha256": canonical.source_sha256,
                    "part_kind": canonical.part_kind,
                    "surah_id": canonical.surah_id,
                    "ayah_number": canonical.ayah_number,
                }
                if canonical
                else None
            ),
        }
        if observed is not None:
            item.update(
                {
                    "token_id": observed.token_id,
                    "raw_symbol": observed.raw_symbol,
                    "normalized_symbol": observed.normalized_symbol,
                    "unit": observed.unit,
                    "start_frame": observed.start_frame,
                    "end_frame": observed.end_frame,
                    "start_time": group.start + observed.start_frame * frame_duration_sec,
                    "end_time": group.start + observed.end_frame * frame_duration_sec,
                    "score": observed.score,
                }
            )
        else:
            item.update(
                {
                    "token_id": None,
                    "raw_symbol": None,
                    "normalized_symbol": None,
                    "unit": None,
                    "start_frame": None,
                    "end_frame": None,
                    "start_time": None,
                    "end_time": None,
                    "score": None,
                }
            )
        cells.append(item)
    return {
        "algorithm": "zipformer-character-dp-v1",
        "operations": ["MATCH", "SUBSTITUTE", "INSERT", "DELETE"],
        "total_cost": alignment.total_cost,
        "normalized_cost": alignment.normalized_cost,
        "match_coverage": alignment.match_coverage,
        "quality_threshold": alignment.quality_threshold,
        "coverage_threshold": alignment.coverage_threshold,
        "observed_units": alignment.observed_units,
        "canonical_units": alignment.canonical_units,
        "cells": cells,
    }
