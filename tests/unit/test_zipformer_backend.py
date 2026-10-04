from __future__ import annotations

import numpy as np
import pytest
from munajjam.exceptions import InvalidProviderOutputError, ModelUnavailableError
from munajjam.hybrid_aligner import AudioBuffer, BreathGroup
from munajjam.hybrid_aligner.zipformer_backend import ZipformerOnnxBackend


class _Value:
    def __init__(self, name, shape, type_name="tensor(float)"):
        self.name = name
        self.shape = shape
        self.type = type_name


class _Session:
    def __init__(self, state_value=None):
        self.state_value = (
            state_value
            if state_value is not None
            else np.zeros((1, 2), dtype=np.float32)
        )
        self.calls = []

    def get_inputs(self):
        return [
            _Value("x", [1, 61, 80]),
            _Value("processed_lens", [1], "tensor(int32)"),
            _Value("state", [1, 2]),
        ]

    def get_outputs(self):
        return [
            _Value("logits", [1, 61, 251]),
            _Value("processed_lens", [1], "tensor(int32)"),
            _Value("state", [1, 2]),
        ]

    def run(self, output_names, feeds):
        self.calls.append(feeds)
        logits = np.full((1, 61, 251), -10.0, dtype=np.float32)
        logits[:, :, 250] = 10.0
        logits[:, 3, 12] = 20.0
        return [logits, np.array([48], dtype=np.int32), self.state_value + 1.0]


def _backend(monkeypatch, session=None):
    monkeypatch.setattr(
        ZipformerOnnxBackend,
        "_read_tokens",
        lambda self: [*(f"t{i}" for i in range(250)), "<blank>"],
    )
    monkeypatch.setattr(ZipformerOnnxBackend, "_read_json", lambda self, name: {})
    return ZipformerOnnxBackend(
        "/unused",
        session=session or _Session(),
        feature_extractor=lambda samples, sample_rate: np.zeros(
            (61, 80), dtype=np.float32
        ),
    )


def test_chunks_pad_to_verified_stream_window():
    chunks = ZipformerOnnxBackend._chunks(np.zeros((62, 80), dtype=np.float32))
    assert [chunk.shape for chunk in chunks] == [(1, 61, 80), (1, 61, 80)]


def test_token_table_uses_explicit_ids_not_line_positions():
    lines = ["ؙ 0", *(f"u{i} {i}" for i in range(1, 249))]
    lines.extend(["ۦۦۦۦۦۦ 249", "<blank> 250"])
    tokens = ZipformerOnnxBackend._parse_token_table("\n".join(reversed(lines)))
    assert tokens[0] == "ؙ"
    assert tokens[249] == "ۦۦۦۦۦۦ"
    assert tokens[250] == "<blank>"


def test_normalize_logits_supports_only_verified_token_axis():
    logits = np.zeros((1, 251, 4), dtype=np.float32)
    normalized = ZipformerOnnxBackend._normalize_logits(logits)
    assert normalized.shape == (1, 4, 251)
    with pytest.raises(InvalidProviderOutputError):
        ZipformerOnnxBackend._normalize_logits(np.zeros((1, 3, 4), dtype=np.float32))


def test_backend_propagates_state_and_decodes_real_shaped_output(monkeypatch):
    session = _Session()
    backend = _backend(monkeypatch, session)
    audio = AudioBuffer(np.zeros(16_000, dtype=np.float32), 16_000)
    emissions = backend(audio, BreathGroup(0.0, 1.0))
    assert len(emissions) == 2
    assert all(emission.token_id == 12 for emission in emissions)
    assert emissions[0].start_frame == 3
    assert session.calls[0]["state"].shape == (1, 2)
    assert np.all(session.calls[0]["state"] == 0)
    assert session.calls[0]["processed_lens"].dtype == np.int32


def test_backend_rejects_nonfinite_state(monkeypatch):
    session = _Session(np.full((1, 2), np.nan, dtype=np.float32))
    backend = _backend(monkeypatch, session)
    audio = AudioBuffer(np.zeros(16_000, dtype=np.float32), 16_000)
    with pytest.raises(InvalidProviderOutputError):
        backend(audio, BreathGroup(0.0, 1.0))


def test_constructor_fails_closed_on_wrong_token_table(monkeypatch):
    monkeypatch.setattr(ZipformerOnnxBackend, "_read_tokens", lambda self: ["wrong"])
    with pytest.raises(ModelUnavailableError):
        ZipformerOnnxBackend("/unused", session=_Session())
