"""Canonical Quran text and verified phoneme-reference contracts.

The repository bundles Hafs and Warsh ayah text. Callers identify the verse
range belonging to each physical breath group; this module deliberately does
not guess verse boundaries from audio duration or segment count. Zipformer's
phoneme map is a separate gated artifact and is represented by a strict
provider contract rather than reconstructed from token IDs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib import resources
from typing import Literal, Protocol

from munajjam.core.arabic import BASMALA_PATTERN, ISTIADHA_PATTERN, normalize_arabic
from munajjam.data.quran import get_ayah
from munajjam.exceptions import ModelUnavailableError, QuranDataError

ReferenceKind = Literal["ayah", "istiadhah", "basmalah"]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True, slots=True)
class VerifiedSpecialPhrase:
    """An explicitly sourced non-ayah phrase with a content digest.

    Isti'adhah is not present in the bundled ayah JSON. It is therefore never
    synthesized by this provider: callers must supply the canonical phrase and
    its source/digest when they explicitly request it.
    """

    kind: Literal["istiadhah", "basmalah"]
    text: str
    source: str
    sha256: str

    def __post_init__(self) -> None:
        if self.kind not in ("istiadhah", "basmalah"):
            raise ValueError("special phrase kind must be istiadhah or basmalah")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("special phrase text must be non-empty")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("special phrase source must be explicit")
        actual = _sha256(self.text.encode("utf-8"))
        if self.sha256.lower() != actual:
            raise ValueError("special phrase SHA-256 does not match its text")
        normalized = normalize_arabic(self.text)
        pattern = ISTIADHA_PATTERN if self.kind == "istiadhah" else BASMALA_PATTERN
        if not pattern.fullmatch(normalized):
            raise ValueError(f"text does not match the declared {self.kind} phrase")

    @classmethod
    def from_source(
        cls, kind: Literal["istiadhah", "basmalah"], text: str, source: str
    ) -> VerifiedSpecialPhrase:
        """Build a phrase record while keeping its source and digest attached."""
        return cls(kind, text, source, _sha256(text.encode("utf-8")))


@dataclass(frozen=True, slots=True)
class CanonicalReferenceRequest:
    """Explicit Quran passage assigned by the caller to one physical breath."""

    surah_id: int
    ayah_start: int
    ayah_end: int | None = None
    riwaya: Literal["hafs", "warsh"] = "hafs"
    include_istiadhah: bool = False
    include_basmalah: bool = False
    istiadhah_phrase: VerifiedSpecialPhrase | None = None

    def __post_init__(self) -> None:
        for name, value in (("surah_id", self.surah_id), ("ayah_start", self.ayah_start)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.ayah_end is not None and (
            isinstance(self.ayah_end, bool)
            or not isinstance(self.ayah_end, int)
            or self.ayah_end < self.ayah_start
        ):
            raise ValueError("ayah_end must be an integer at least ayah_start")
        if self.surah_id > 114:
            raise ValueError("surah_id must be in 1..114")
        if self.riwaya not in ("hafs", "warsh"):
            raise ValueError("riwaya must be hafs or warsh")
        if not isinstance(self.include_istiadhah, bool) or not isinstance(
            self.include_basmalah, bool
        ):
            raise ValueError("special phrase flags must be bools")
        if self.istiadhah_phrase is not None and not isinstance(
            self.istiadhah_phrase, VerifiedSpecialPhrase
        ):
            raise ValueError("istiadhah_phrase must be a VerifiedSpecialPhrase")
        if self.istiadhah_phrase is not None and self.istiadhah_phrase.kind != "istiadhah":
            raise ValueError("istiadhah_phrase must have kind='istiadhah'")
        if self.istiadhah_phrase is not None and not self.include_istiadhah:
            raise ValueError("an istiadhah phrase requires include_istiadhah=True")


@dataclass(frozen=True, slots=True)
class CanonicalReferencePart:
    kind: ReferenceKind
    text: str
    source: str
    source_sha256: str
    surah_id: int | None = None
    ayah_number: int | None = None


@dataclass(frozen=True, slots=True)
class CanonicalReference:
    riwaya: str
    parts: tuple[CanonicalReferencePart, ...]
    source_file: str
    source_file_sha256: str
    text: str
    text_sha256: str


class CanonicalQuranReferenceProvider:
    """Look up exact, bundled Hafs/Warsh ayah text with deterministic provenance."""

    def get_reference(self, request: CanonicalReferenceRequest) -> CanonicalReference:
        if request.include_istiadhah and request.istiadhah_phrase is None:
            raise ModelUnavailableError(
                "Isti'adhah was requested, but no sourced canonical phrase was supplied; "
                "provide the phrase with its source and SHA-256"
            )

        filename = f"quran_{request.riwaya}.json"
        try:
            source_bytes = resources.files("munajjam.data").joinpath(filename).read_bytes()
        except Exception as exc:
            raise QuranDataError(f"bundled Quran reference {filename} is unavailable") from exc
        source_digest = _sha256(source_bytes)
        source_name = f"munajjam.data/{filename}"
        parts: list[CanonicalReferencePart] = []

        if request.include_istiadhah:
            phrase = request.istiadhah_phrase
            assert phrase is not None
            parts.append(
                CanonicalReferencePart("istiadhah", phrase.text, phrase.source, phrase.sha256)
            )

        ayah_end = request.ayah_end or request.ayah_start
        includes_fatiha_basmala = request.surah_id == 1 and request.ayah_start <= 1 <= ayah_end
        if request.include_basmalah and not includes_fatiha_basmala:
            basmalah = get_ayah(1, 1, request.riwaya)
            if basmalah is None or not basmalah.text.strip():
                raise QuranDataError(
                    f"bundled {request.riwaya} reference has no verified Basmalah source at 1:1"
                )
            if not BASMALA_PATTERN.fullmatch(normalize_arabic(basmalah.text)):
                raise QuranDataError(
                    f"bundled {request.riwaya} ayah 1:1 does not verify as Basmalah"
                )
            parts.append(
                CanonicalReferencePart(
                    "basmalah",
                    basmalah.text,
                    f"{source_name}#1:1",
                    source_digest,
                    surah_id=1,
                    ayah_number=1,
                )
            )

        for ayah_number in range(request.ayah_start, ayah_end + 1):
            ayah = get_ayah(request.surah_id, ayah_number, request.riwaya)
            if ayah is None or not ayah.text.strip():
                raise QuranDataError(
                    f"bundled {request.riwaya} Quran reference is unavailable for "
                    f"{request.surah_id}:{ayah_number}"
                )
            parts.append(
                CanonicalReferencePart(
                    "ayah",
                    ayah.text,
                    f"{source_name}#{request.surah_id}:{ayah_number}",
                    source_digest,
                    surah_id=request.surah_id,
                    ayah_number=ayah_number,
                )
            )

        text = " ".join(part.text for part in parts)
        if not text.strip():
            raise QuranDataError("canonical Quran reference is empty")
        return CanonicalReference(
            riwaya=request.riwaya,
            parts=tuple(parts),
            source_file=source_name,
            source_file_sha256=source_digest,
            text=text,
            text_sha256=_sha256(text.encode("utf-8")),
        )


@dataclass(frozen=True, slots=True)
class ZipformerPhonemeTarget:
    """Strict shape for phoneme targets decoded from the pinned gated artifact."""

    token_ids: tuple[int, ...]
    token_texts: tuple[str, ...]
    canonical_text_sha256: str
    artifact_file: str
    artifact_sha256: str
    revision: str = "506422c82a81c86e7ae74a5a2ab4641724bcd3b3"

    def __post_init__(self) -> None:
        if not isinstance(self.token_ids, tuple) or not isinstance(self.token_texts, tuple):
            raise ValueError("Zipformer targets must use immutable token ID/text tuples")
        if not self.token_ids or len(self.token_ids) != len(self.token_texts):
            raise ValueError("Zipformer target IDs and texts must be non-empty and parallel")
        if any(
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            or token_id >= 250
            for token_id in self.token_ids
        ):
            raise ValueError("Zipformer targets must be valid non-blank IDs below 250")
        if any(not isinstance(text, str) or not text for text in self.token_texts):
            raise ValueError("Zipformer target symbols must be non-empty strings")
        for name, value in (
            ("canonical_text_sha256", self.canonical_text_sha256),
            ("artifact_sha256", self.artifact_sha256),
        ):
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
                raise ValueError(f"{name} must be a 64-character SHA-256")
        if self.revision != "506422c82a81c86e7ae74a5a2ab4641724bcd3b3":
            raise ValueError("Zipformer phoneme target must use the pinned artifact revision")
        if self.artifact_file not in {
            "quran_text2phoneme.json",
            "ordered_quran_phonemes.json",
        }:
            raise ValueError("Zipformer targets must cite an authoritative Quran phoneme file")


class ZipformerReferenceTargetProvider(Protocol):
    """Adapter contract for the gated artifact's canonical text-to-phoneme map."""

    def get_target(self, reference: CanonicalReference) -> ZipformerPhonemeTarget: ...
