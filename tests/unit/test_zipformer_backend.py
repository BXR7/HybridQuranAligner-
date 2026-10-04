from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from munajjam.exceptions import InvalidProviderOutputError, ModelUnavailableError
from munajjam.hybrid_aligner import AudioBuffer, BreathGroup
from munajjam.hybrid_aligner.zipformer_backend import (
    ZIPFORMER_BLANK_ID,
    ZIPFORMER_DECODE_CHUNK_LEN,
    ZIPFORMER_FRAME_DURATION_SEC,
    ZIPFORMER_INPUT_FRAMES,
    ZIPFORMER_OUTPUT_FRAMES,
    ZipformerOnnxBackend,
)


class _Value:
    def __init__(self, name, shape, type_name="tensor(float)"):
        self.name = name
        self.shape = shape
        self.type = type_name


class _Session:
    def __init__(
        self,
        *,
        state_type="tensor(float)",
        state_output="state",
        processed_output="processed_lens",
        output_shape=(1, ZIPFORMER_OUTPUT_FRAMES, 251),
        runtime_output_frames=ZIPFORMER_OUTPUT_FRAMES,
        runtime_batch=1,
        runtime_vocabulary_size=251,
        token_frames=None,
        nonfinite_logits=False,
        nonfinite_state=False,
        nonfinite_processed=False,
        processed_input_type="tensor(int32)",
        processed_input_shape=(1,),
        processed_output_type="tensor(int32)",
        processed_output_shape=(1,),
        acoustic_shape=(1, ZIPFORMER_INPUT_FRAMES, 80),
    ):
        self.state_type = state_type
        self.state_output = state_output
        self.processed_output = processed_output
        self.output_shape = list(output_shape)
        self.runtime_output_frames = runtime_output_frames
        self.runtime_batch = runtime_batch
        self.runtime_vocabulary_size = runtime_vocabulary_size
        self.token_frames = token_frames or []
        self.nonfinite_logits = nonfinite_logits
        self.nonfinite_state = nonfinite_state
        self.nonfinite_processed = nonfinite_processed
        self.processed_input_type = processed_input_type
        self.processed_input_shape = list(processed_input_shape)
        self.processed_output_type = processed_output_type
        self.processed_output_shape = list(processed_output_shape)
        self.acoustic_shape = list(acoustic_shape)
        self.calls = []

    def get_inputs(self):
        return [
            _Value("x", self.acoustic_shape),
            _Value(
                "processed_lens", self.processed_input_shape, self.processed_input_type
            ),
            _Value("state", [1, 2], self.state_type),
        ]

    def get_outputs(self):
        return [
            _Value("log_probs", self.output_shape),
            _Value(
                self.processed_output,
                self.processed_output_shape,
                self.processed_output_type,
            ),
            _Value(self.state_output, [1, 2], self.state_type),
        ]

    def run(self, output_names, feeds):
        self.calls.append(feeds)
        logits = np.full(
            (
                self.runtime_batch,
                self.runtime_output_frames,
                self.runtime_vocabulary_size,
            ),
            -10.0,
            dtype=np.float32,
        )
        if self.runtime_vocabulary_size > ZIPFORMER_BLANK_ID:
            logits[:, :, ZIPFORMER_BLANK_ID] = 10.0
        call_index = len(self.calls) - 1
        if call_index < len(self.token_frames):
            for frame_index, token_id in enumerate(self.token_frames[call_index]):
                logits[0, frame_index, token_id] = 20.0
        elif self.runtime_output_frames > 3 and self.runtime_vocabulary_size > 12:
            logits[0, 3, 12] = 20.0
        if self.nonfinite_logits:
            logits[0, 0, 0] = np.nan

        dtype = np.dtype(
            {
                "tensor(float)": np.float32,
                "tensor(float16)": np.float16,
                "tensor(int32)": np.int32,
                "tensor(int64)": np.int64,
            }[self.state_type]
        )
        state = np.asarray(feeds["state"] + 1, dtype=dtype)
        if self.nonfinite_state:
            state = np.full((1, 2), np.nan, dtype=np.float32)

        if self.nonfinite_processed:
            processed = np.array([np.nan], dtype=np.float32)
        else:
            processed = np.asarray(feeds["processed_lens"] + ZIPFORMER_DECODE_CHUNK_LEN)
        return [logits, processed, state]


class _KaggleMetadataSession:
    """Fake runtime with the exact symbolic metadata reported by Kaggle."""

    def get_inputs(self):
        return [
            _Value("x", ["N", 61, 80], "tensor(float)"),
            _Value("processed_lens", ["N"], "tensor(int64)"),
            _Value("cached_key_0", [256, "N", 128], "tensor(float)"),
            _Value("embed_states", ["N", 128, 3, 19], "tensor(float)"),
        ]

    def get_outputs(self):
        return [
            _Value("log_probs", ["N", "LogSoftmaxlog_probs_dim_1", 251]),
            _Value("new_processed_lens", ["N"], "tensor(int64)"),
            _Value(
                "new_cached_key_0",
                ["Slicenew_cached_key_0_dim_0", "N", 128],
                "tensor(float)",
            ),
            _Value(
                "new_embed_states",
                ["N", 128, "Slicenew_embed_states_dim_2", 19],
                "tensor(float)",
            ),
        ]


def _backend(monkeypatch, session=None, *, feature_count=61):
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
            (feature_count, 80), dtype=np.float32
        ),
    )


def _audio() -> AudioBuffer:
    return AudioBuffer(np.zeros(16_000, dtype=np.float32), 16_000)


def _decode(backend, session):
    return backend(_audio(), BreathGroup(0.0, 1.0))


def test_chunks_use_61_input_frames_and_exact_48_frame_starts():
    features = np.zeros((100, 80), dtype=np.float32)
    chunks = ZipformerOnnxBackend._chunks(features)
    assert ZipformerOnnxBackend._chunk_starts(len(features)) == [0, 48, 96]
    assert [chunk.shape for chunk in chunks] == [(1, 61, 80)] * 3
    assert ZIPFORMER_INPUT_FRAMES == 61
    assert ZIPFORMER_DECODE_CHUNK_LEN == 48


def test_final_window_is_zero_padded_after_real_features():
    features = np.arange(49 * 80, dtype=np.float32).reshape(49, 80)
    chunks = ZipformerOnnxBackend._chunks(features)
    assert len(chunks) == 2
    final_window = chunks[1][0]
    np.testing.assert_array_equal(final_window[0], features[48])
    assert np.count_nonzero(final_window[1:]) == 0


def test_fbank_contract_settings_scaling_and_final_frame(monkeypatch):
    captured = {}

    class _FrameOptions:
        pass

    class _MelOptions:
        pass

    class _FbankOptions:
        def __init__(self):
            self.frame_opts = _FrameOptions()
            self.mel_opts = _MelOptions()

    class _OnlineFbank:
        def __init__(self, options):
            captured["options"] = options
            self.num_frames_ready = 1

        def accept_waveform(self, sample_rate, waveform):
            captured["sample_rate"] = sample_rate
            captured["waveform"] = np.asarray(waveform, dtype=np.float32)

        def get_frame(self, index):
            assert index == 0
            return np.arange(80, dtype=np.float32)

    monkeypatch.setitem(
        sys.modules,
        "kaldi_native_fbank",
        SimpleNamespace(FbankOptions=_FbankOptions, OnlineFbank=_OnlineFbank),
    )
    samples = np.array([0.0, 0.25, -0.5], dtype=np.float32)
    features = ZipformerOnnxBackend._official_feature_extractor(samples, 16_000)
    options = captured["options"]
    assert options.frame_opts.samp_freq == 16_000
    assert options.frame_opts.frame_length_ms == 25.0
    assert options.frame_opts.frame_shift_ms == 10.0
    assert options.frame_opts.preemph_coeff == 0.97
    assert options.frame_opts.remove_dc_offset is True
    assert options.frame_opts.dither == 0.0
    assert options.frame_opts.snip_edges is False
    assert options.frame_opts.window_type == "povey"
    assert options.frame_opts.round_to_power_of_two is True
    assert options.mel_opts.num_bins == 80
    assert options.mel_opts.low_freq == 20.0
    assert options.mel_opts.high_freq == 0.0
    assert options.raw_energy is True
    assert options.use_energy is False
    assert options.energy_floor == 0.0
    assert captured["sample_rate"] == 16_000
    np.testing.assert_array_equal(captured["waveform"], samples * 32_768.0)
    assert features.shape == (1, 80)
    np.testing.assert_array_equal(features[0], np.arange(80, dtype=np.float32))


def test_installed_kaldi_fbank_is_deterministic_with_final_frame_enabled():
    pytest.importorskip("kaldi_native_fbank")
    sample_index = np.arange(16_000, dtype=np.float32)
    samples = np.sin(2 * np.pi * 440 * sample_index / 16_000).astype(np.float32)
    first = ZipformerOnnxBackend._official_feature_extractor(samples, 16_000)
    second = ZipformerOnnxBackend._official_feature_extractor(samples, 16_000)
    assert first.shape == (99, 80)
    assert np.all(np.isfinite(first))
    np.testing.assert_array_equal(first, second)


def test_token_table_uses_explicit_ids_not_line_positions_and_blank_250():
    lines = ["ؙ 0", *(f"u{i} {i}" for i in range(1, 249))]
    lines.extend(["ۦۦۦۦۦۦ 249", "<blank> 250"])
    tokens = ZipformerOnnxBackend._parse_token_table("\n".join(reversed(lines)))
    assert tokens[0] == "ؙ"
    assert tokens[249] == "ۦۦۦۦۦۦ"
    assert tokens[ZIPFORMER_BLANK_ID] == "<blank>"


@pytest.mark.parametrize(
    "table",
    [
        "token 0\ntoken 0",  # duplicate ID
        "token 1",  # missing ID 0
        "token nope",  # non-integer ID
        "token -1",  # negative ID
    ],
)
def test_token_table_rejects_invalid_or_missing_ids(table):
    with pytest.raises(ModelUnavailableError):
        ZipformerOnnxBackend._parse_token_table(table)


def test_normalize_logits_supports_verified_token_axis():
    logits = np.zeros((1, 251, ZIPFORMER_OUTPUT_FRAMES), dtype=np.float32)
    normalized = ZipformerOnnxBackend._normalize_logits(logits)
    assert normalized.shape == (1, ZIPFORMER_OUTPUT_FRAMES, 251)
    with pytest.raises(InvalidProviderOutputError):
        ZipformerOnnxBackend._normalize_logits(np.zeros((1, 3, 4), dtype=np.float32))


def test_backend_propagates_state_and_processed_lens_across_calls(monkeypatch):
    session = _Session()
    backend = _backend(monkeypatch, session, feature_count=100)
    emissions = _decode(backend, session)
    assert [feed["processed_lens"].tolist() for feed in session.calls] == [
        [0],
        [48],
        [96],
    ]
    assert [feed["state"].tolist() for feed in session.calls] == [
        [[0.0, 0.0]],
        [[1.0, 1.0]],
        [[2.0, 2.0]],
    ]
    assert len(emissions) == 2
    assert [item.start_frame for item in emissions] == [3, 15]
    assert session.calls[0]["processed_lens"].dtype == np.int32


@pytest.mark.parametrize(
    ("state_output", "processed_output"),
    [("state", "processed_lens"), ("new_state", "new_processed_lens")],
)
def test_backend_resolves_same_name_and_new_prefix_state_outputs(
    monkeypatch, state_output, processed_output
):
    session = _Session(state_output=state_output, processed_output=processed_output)
    backend = _backend(monkeypatch, session, feature_count=48)
    _decode(backend, session)
    assert session.calls[0]["state"].tolist() == [[0.0, 0.0]]


def test_backend_initializes_with_real_kaggle_symbolic_metadata(monkeypatch):
    session = _KaggleMetadataSession()
    backend = _backend(monkeypatch, session, feature_count=48)

    # This fixture validates real graph metadata and initialization only. It
    # deliberately does not fabricate a model response or claim real inference.
    assert backend._logit_output == "log_probs"
    assert backend._processed_input == "processed_lens"
    assert backend._processed_output == "new_processed_lens"
    assert backend._initial_processed_lens().dtype == np.int64
    assert backend._initial_processed_lens().shape == (1,)
    state = backend._initial_state()
    assert state["cached_key_0"].shape == (256, 1, 128)
    assert state["embed_states"].shape == (1, 128, 3, 19)
    assert backend._state_output_names == {
        "cached_key_0": "new_cached_key_0",
        "embed_states": "new_embed_states",
    }


@pytest.mark.parametrize(
    ("type_name", "dtype"),
    [
        ("tensor(float)", np.float32),
        ("tensor(float16)", np.float16),
        ("tensor(int32)", np.int32),
        ("tensor(int64)", np.int64),
    ],
)
def test_state_dtype_follows_graph_contract(monkeypatch, type_name, dtype):
    session = _Session(state_type=type_name)
    backend = _backend(monkeypatch, session, feature_count=48)
    _decode(backend, session)
    assert session.calls[0]["state"].dtype == np.dtype(dtype)


def test_graph_batch_symbol_n_is_resolved_to_batch_one(monkeypatch):
    session = _Session(acoustic_shape=["N", ZIPFORMER_INPUT_FRAMES, 80])
    backend = _backend(monkeypatch, session, feature_count=48)
    assert backend._x_input == "x"


def test_graph_with_fixed_non_one_batch_is_rejected(monkeypatch):
    with pytest.raises(ModelUnavailableError, match="acoustic input"):
        _backend(monkeypatch, _Session(acoustic_shape=[2, ZIPFORMER_INPUT_FRAMES, 80]))


def test_backend_rejects_nonfinite_logits(monkeypatch):
    session = _Session(nonfinite_logits=True)
    backend = _backend(monkeypatch, session, feature_count=48)
    with pytest.raises(InvalidProviderOutputError, match="logits"):
        _decode(backend, session)


def test_backend_rejects_nonfinite_state(monkeypatch):
    session = _Session(nonfinite_state=True)
    backend = _backend(monkeypatch, session, feature_count=48)
    with pytest.raises(InvalidProviderOutputError, match="state"):
        _decode(backend, session)


def test_backend_rejects_nonfinite_processed_lens(monkeypatch):
    session = _Session(nonfinite_processed=True)
    backend = _backend(monkeypatch, session, feature_count=48)
    with pytest.raises(InvalidProviderOutputError, match="processed_lens"):
        _decode(backend, session)


def test_backend_rejects_wrong_ctc_frame_count(monkeypatch):
    session = _Session(
        output_shape=["N", "dynamic_time", 251],
        runtime_output_frames=61,
    )
    backend = _backend(monkeypatch, session, feature_count=48)
    with pytest.raises(InvalidProviderOutputError, match="12 CTC frames"):
        _decode(backend, session)


@pytest.mark.parametrize(
    ("runtime_batch", "runtime_vocabulary_size", "message"),
    [
        (2, 251, "runtime shape \\[1,T,251\\]"),
        (1, 51, "unambiguous 251-token axis"),
    ],
)
def test_backend_rejects_wrong_runtime_batch_or_vocabulary(
    monkeypatch, runtime_batch, runtime_vocabulary_size, message
):
    session = _Session(
        output_shape=["N", "dynamic_time", 251],
        runtime_batch=runtime_batch,
        runtime_vocabulary_size=runtime_vocabulary_size,
    )
    backend = _backend(monkeypatch, session, feature_count=48)
    with pytest.raises(InvalidProviderOutputError, match=message):
        _decode(backend, session)


def test_backend_rejects_output_without_251_class_axis(monkeypatch):
    session = _Session(output_shape=(1, ZIPFORMER_OUTPUT_FRAMES, 51))
    with pytest.raises(ModelUnavailableError, match="unique 251-token axis"):
        _backend(monkeypatch, session)


def test_backend_rejects_ambiguous_structural_ctc_outputs(monkeypatch):
    session = _Session(output_shape=["N", "dynamic_time", 251])
    original_get_outputs = session.get_outputs
    session.get_outputs = lambda: [
        *original_get_outputs(),
        _Value("second_ctc_output", ["N", "another_time", 251]),
    ]
    with pytest.raises(ModelUnavailableError, match="second_ctc_output"):
        _backend(monkeypatch, session)


@pytest.mark.parametrize(
    ("onnx_type", "expected_dtype"),
    [("tensor(int32)", np.int32), ("tensor(int64)", np.int64)],
)
def test_processed_lens_dtype_is_derived_from_real_input_metadata(
    monkeypatch, onnx_type, expected_dtype
):
    session = _Session(
        processed_input_type=onnx_type,
        processed_input_shape=["N"],
        processed_output_type=onnx_type,
        processed_output_shape=["N"],
        processed_output="new_processed_lens",
    )
    backend = _backend(monkeypatch, session, feature_count=96)
    _decode(backend, session)
    assert [feed["processed_lens"].dtype for feed in session.calls] == [
        np.dtype(expected_dtype),
        np.dtype(expected_dtype),
    ]


def test_processed_lens_rejects_input_output_dtype_mismatch(monkeypatch):
    session = _Session(
        processed_input_type="tensor(int64)",
        processed_input_shape=["N"],
        processed_output_type="tensor(int32)",
        processed_output_shape=["N"],
    )
    with pytest.raises(ModelUnavailableError, match="dtypes differ"):
        _backend(monkeypatch, session)


def test_full_chunks_emit_12_frames_and_frame_grid_is_40ms(monkeypatch):
    session = _Session()
    backend = _backend(monkeypatch, session, feature_count=48)
    emissions = _decode(backend, session)
    assert len(emissions) == 1
    assert emissions[0].start_frame == 3
    assert ZIPFORMER_FRAME_DURATION_SEC == 0.04
    assert ZIPFORMER_OUTPUT_FRAMES * ZIPFORMER_FRAME_DURATION_SEC == pytest.approx(0.48)


def test_final_padded_chunk_trims_logits_to_real_advance(monkeypatch):
    first = [ZIPFORMER_BLANK_ID] * ZIPFORMER_OUTPUT_FRAMES
    second = [ZIPFORMER_BLANK_ID] * ZIPFORMER_OUTPUT_FRAMES
    first[3] = 12
    second[0] = 12
    session = _Session(token_frames=[first, second])
    backend = _backend(monkeypatch, session, feature_count=49)
    emissions = _decode(backend, session)
    # Final advance contains one real feature frame: one 40 ms-grid output is kept.
    assert len(emissions) == 2
    assert [item.start_frame for item in emissions] == [3, 12]


@pytest.mark.parametrize(
    ("first_tail", "second_head", "expected_count"),
    [
        (7, 7, 1),  # A | A collapses across the call boundary
        (7, ZIPFORMER_BLANK_ID, 2),  # A | blank | A remains distinct
        (ZIPFORMER_BLANK_ID, 7, 1),  # blank | A begins a new emission
    ],
)
def test_ctc_decoder_preserves_raw_state_across_chunk_boundaries(
    monkeypatch, first_tail, second_head, expected_count
):
    first = [ZIPFORMER_BLANK_ID] * ZIPFORMER_OUTPUT_FRAMES
    second = [ZIPFORMER_BLANK_ID] * ZIPFORMER_OUTPUT_FRAMES
    first[-1] = first_tail
    second[0] = second_head
    if second_head == ZIPFORMER_BLANK_ID:
        second[1] = 7
    session = _Session(token_frames=[first, second])
    backend = _backend(monkeypatch, session, feature_count=96)
    emissions = _decode(backend, session)
    token_emissions = [item for item in emissions if item.token_id == 7]
    assert len(token_emissions) == expected_count
    if first_tail == 7 and second_head == 7:
        assert token_emissions[0].start_frame == 11
        assert token_emissions[0].end_frame == 13
    if first_tail == 7 and second_head == ZIPFORMER_BLANK_ID:
        assert token_emissions[0].end_frame == 12
        assert token_emissions[1].start_frame == 13


@pytest.mark.parametrize(
    ("prefix", "expected_bounds"),
    [
        ([9, 9, ZIPFORMER_BLANK_ID], [(0, 2)]),  # A | A collapses
        ([9, ZIPFORMER_BLANK_ID, 9], [(0, 1), (2, 3)]),  # A | blank | A stays distinct
        ([ZIPFORMER_BLANK_ID, 9], [(1, 2)]),  # blank | A starts at frame one
        ([9, ZIPFORMER_BLANK_ID], [(0, 1)]),  # A | blank closes the run
    ],
)
def test_ctc_collapse_inside_chunk_handles_repeats_and_blanks(
    monkeypatch, prefix, expected_bounds
):
    frame_ids = [ZIPFORMER_BLANK_ID] * ZIPFORMER_OUTPUT_FRAMES
    frame_ids[: len(prefix)] = prefix
    session = _Session(token_frames=[frame_ids])
    backend = _backend(monkeypatch, session, feature_count=48)
    emissions = _decode(backend, session)
    assert [(item.start_frame, item.end_frame) for item in emissions] == expected_bounds


def test_constructor_fails_closed_on_wrong_token_table(monkeypatch):
    monkeypatch.setattr(ZipformerOnnxBackend, "_read_tokens", lambda self: ["wrong"])
    with pytest.raises(ModelUnavailableError):
        ZipformerOnnxBackend("/unused", session=_Session())
