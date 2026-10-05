from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from munajjam.exceptions import InvalidProviderOutputError
from munajjam.hybrid_aligner import AudioBuffer, BreathGroup
from munajjam.hybrid_aligner.zipformer_backend import (
    PUBLIC_ZIPFORMER_MODEL,
    PUBLIC_ZIPFORMER_TOKENS,
    SherpaZipformerBackend,
)


class _Stream:
    def __init__(self) -> None:
        self.waveform = None
        self.finished = False

    def accept_waveform(self, sample_rate, waveform):
        self.waveform = (sample_rate, np.asarray(waveform))

    def input_finished(self):
        self.finished = True


class _Recognizer:
    def __init__(self, tokens, timestamps, probabilities):
        self.stream = _Stream()
        self._tokens = tokens
        self._timestamps = timestamps
        self._probabilities = probabilities
        self.decode_calls = 0

    def create_stream(self):
        return self.stream

    def is_ready(self, stream):
        return self.decode_calls == 0

    def decode_stream(self, stream):
        self.decode_calls += 1

    def tokens(self, stream):
        return self._tokens

    def timestamps(self, stream):
        return self._timestamps

    def ys_probs(self, stream):
        return self._probabilities


def _artifact(tmp_path: Path) -> Path:
    root = tmp_path / "public-model"
    (root / "model_zipformer").mkdir(parents=True, exist_ok=True)
    (root / PUBLIC_ZIPFORMER_MODEL).write_bytes(b"not loaded by the fake recognizer")
    (root / PUBLIC_ZIPFORMER_TOKENS).write_text(
        "\n".join([*(f"t{i} {i}" for i in range(250)), "<blank> 250"]),
        encoding="utf-8",
    )
    return root


def _audio() -> AudioBuffer:
    return AudioBuffer(np.zeros(16_000, dtype=np.float32), 16_000)


def test_public_backend_is_lazy_and_uses_sherpa_contract(tmp_path):
    calls = []
    recognizer = _Recognizer(["t7", "t8"], [0.08, 0.16], [0.8, 0.7])

    def factory(**kwargs):
        calls.append(kwargs)
        return recognizer

    backend = SherpaZipformerBackend(_artifact(tmp_path), recognizer_factory=factory)
    assert calls == []
    emissions = backend(_audio(), BreathGroup(0.0, 1.0))
    assert calls == [
        {
            "tokens": str(_artifact(tmp_path) / PUBLIC_ZIPFORMER_TOKENS),
            "model": str(_artifact(tmp_path) / PUBLIC_ZIPFORMER_MODEL),
        }
    ]
    assert recognizer.stream.finished is True
    assert recognizer.stream.waveform[0] == 16_000
    assert [item.token_id for item in emissions] == [7, 8]
    assert [(item.start_frame, item.end_frame) for item in emissions] == [
        (2, 4),
        (4, 5),
    ]
    assert [item.score for item in emissions] == [0.8, 0.7]


def test_public_backend_rejects_unknown_sherpa_token(tmp_path):
    recognizer = _Recognizer(["unknown"], [0.0], [0.9])
    backend = SherpaZipformerBackend(
        _artifact(tmp_path), recognizer_factory=lambda **kwargs: recognizer
    )
    with pytest.raises(InvalidProviderOutputError, match="unknown token"):
        backend(_audio(), BreathGroup(0.0, 1.0))


def test_public_backend_requires_artifacts(tmp_path):
    with pytest.raises(Exception, match="missing"):
        SherpaZipformerBackend(tmp_path)
