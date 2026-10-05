"""Verified-artifact Zipformer ONNX backend.

The backend intentionally refuses to infer a runtime contract from a filename.
It requires the pinned artifact metadata and derives ONNX state shapes from the
actual graph. Optional dependencies are imported only when the backend loads.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from munajjam.exceptions import InvalidProviderOutputError, ModelUnavailableError
from munajjam.hybrid_aligner.types import AudioBuffer, BreathGroup, PhonemeEmission

ZIPFORMER_REPOSITORY = "Quran-Lab/zipformer_p-arabic-v3"
ZIPFORMER_REVISION = "506422c82a81c86e7ae74a5a2ab4641724bcd3b3"
ZIPFORMER_MODEL = "zipformer_p_arabic_v3.1.onnx"
ZIPFORMER_VOCABULARY_SIZE = 251
ZIPFORMER_BLANK_ID = 250
ZIPFORMER_TOKEN_SHA256 = "252c10687e442aa9291973065fae19fa39bcd681c4f5612ec496a647e20b43a1"
PUBLIC_ZIPFORMER_REPOSITORY = "Alimalas/munajjam-onnx-models"
PUBLIC_ZIPFORMER_REVISION = "5dbab4db48a88f5a2a76ead282b2bc3d4b958ee0"
PUBLIC_ZIPFORMER_MODEL = "model_zipformer/zipformer_p_arabic_v3.onnx"
PUBLIC_ZIPFORMER_TOKENS = "model_zipformer/tokens.txt"
# Pinned model-card contract at ZIPFORMER_REVISION: 61 input fbank frames,
# 48-frame (0.48 s) decode advance, and 12 CTC outputs per full advance.
ZIPFORMER_INPUT_FRAMES = 61
ZIPFORMER_DECODE_CHUNK_LEN = 48
ZIPFORMER_OUTPUT_FRAMES = 12
ZIPFORMER_FRAME_DURATION_SEC = 0.04


class ZipformerOnnxBackend:
    """Run the pinned streaming Zipformer artifact against one breath group.

    ``feature_extractor`` is an explicit seam for the artifact's official
    feature implementation. In production it must be supplied by the verified
    runtime package; tests may inject a deterministic extractor. No generic
    MFCC/log-mel fallback is used.
    """

    def __init__(
        self,
        artifact_dir: str | Path,
        *,
        model_name: str = ZIPFORMER_MODEL,
        feature_extractor: Callable[[np.ndarray, int], np.ndarray] | None = None,
        session: Any | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self._feature_extractor = feature_extractor or self._official_feature_extractor
        self._tokens = self._read_tokens()
        if len(self._tokens) != ZIPFORMER_VOCABULARY_SIZE:
            raise ModelUnavailableError("Zipformer tokens.txt does not contain 251 tokens")
        if self._tokens[ZIPFORMER_BLANK_ID] != "<blank>":
            raise ModelUnavailableError("Zipformer blank token metadata is invalid")
        self._config = self._read_json("config.json")
        self._session = session or self._load_session(self.artifact_dir / model_name)
        self._x_input = self._find_acoustic_input()
        self._processed_input, self._processed_output = self._find_processed_lens()
        self._state_output_names: dict[str, str] = {}
        self._state_inputs = self._find_state_inputs()
        self._state_outputs = self._find_state_outputs()
        if set(self._state_inputs) != set(self._state_outputs):
            raise ModelUnavailableError(
                "Zipformer ONNX graph does not expose one-to-one streaming state inputs/outputs"
            )
        self._logit_output = self._find_logit_output()

    def _read_json(self, name: str) -> dict[str, Any]:
        try:
            with (self.artifact_dir / name).open(encoding="utf-8") as handle:
                value = json.load(handle)
        except Exception as exc:
            raise ModelUnavailableError(f"unable to read Zipformer {name}") from exc
        if not isinstance(value, dict):
            raise ModelUnavailableError(f"Zipformer {name} must contain an object")
        return value

    def _read_tokens(self) -> list[str]:
        path = self.artifact_dir / "tokens.txt"
        try:
            tokens = self._parse_token_table(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ModelUnavailableError("unable to read Zipformer tokens.txt") from exc
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != ZIPFORMER_TOKEN_SHA256:
            raise ModelUnavailableError(
                "Zipformer tokens.txt SHA-256 does not match the pinned artifact"
            )
        return tokens

    @property
    def token_table(self) -> tuple[str, ...]:
        """Return the verified token table used by the loaded artifact."""
        return tuple(self._tokens)

    @staticmethod
    def _parse_token_table(text: str) -> list[str]:
        """Parse the artifact's ``piece id`` table, independent of line order."""
        entries: dict[int, str] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                piece, raw_id = line.rsplit(maxsplit=1)
                token_id = int(raw_id)
            except (ValueError, TypeError) as exc:
                raise ModelUnavailableError("malformed Zipformer token-table line") from exc
            if token_id < 0 or token_id in entries:
                raise ModelUnavailableError("Zipformer token IDs must be unique and non-negative")
            entries[token_id] = piece
        if sorted(entries) != list(range(len(entries))):
            raise ModelUnavailableError("Zipformer token IDs must be contiguous from zero")
        return [entries[index] for index in range(len(entries))]

    @staticmethod
    def _load_session(model_path: Path) -> Any:
        try:
            import onnxruntime as ort

            return ort.InferenceSession(
                str(model_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
            )
        except Exception as exc:
            raise ModelUnavailableError(
                "Zipformer ONNX Runtime is unavailable or failed to load"
            ) from exc

    @staticmethod
    def _shape(value: Any) -> tuple[int, ...] | None:
        shape = getattr(value, "shape", None)
        if not isinstance(shape, (list, tuple)):
            return None
        result: list[int] = []
        for dimension in shape:
            if isinstance(dimension, int) and dimension > 0:
                result.append(dimension)
            elif isinstance(dimension, str) and dimension.lower() in {"n", "b", "batch"}:
                result.append(1)
            else:
                return None
        return tuple(result)

    def _find_acoustic_input(self) -> str:
        candidates = []
        for item in self._session.get_inputs():
            shape = self._shape(item)
            if shape == (1, ZIPFORMER_INPUT_FRAMES, 80):
                candidates.append(item.name)
        if candidates != ["x"]:
            raise ModelUnavailableError(
                f"expected one acoustic input named x with shape [N,61,80], got {candidates}"
            )
        return candidates[0]

    def _find_processed_lens(self) -> tuple[str, str]:
        inputs = {item.name: item for item in self._session.get_inputs()}
        outputs = {item.name: item for item in self._session.get_outputs()}
        if "processed_lens" not in inputs:
            raise ModelUnavailableError("Zipformer graph is missing processed_lens input")
        output_name = "new_processed_lens" if "new_processed_lens" in outputs else "processed_lens"
        if output_name not in outputs:
            raise ModelUnavailableError("Zipformer graph is missing processed_lens output")
        input_value = inputs["processed_lens"]
        output_value = outputs[output_name]
        input_shape = self._shape(input_value)
        output_shape = self._shape(output_value)
        if input_shape is None or output_shape != input_shape:
            raise ModelUnavailableError("Zipformer processed_lens input/output shapes differ")
        input_dtype = self._numpy_dtype(input_value)
        output_dtype = self._numpy_dtype(output_value)
        if input_dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
            raise ModelUnavailableError("Zipformer processed_lens must use int32 or int64")
        if output_dtype != input_dtype:
            raise ModelUnavailableError("Zipformer processed_lens input/output dtypes differ")
        return "processed_lens", output_name

    def _find_state_inputs(self) -> dict[str, Any]:
        outputs = {item.name: item for item in self._session.get_outputs()}
        return {
            item.name: item
            for item in self._session.get_inputs()
            if item.name not in (self._x_input, "processed_lens")
            and (item.name in outputs or f"new_{item.name}" in outputs)
        }

    def _find_state_outputs(self) -> dict[str, Any]:
        outputs = {item.name: item for item in self._session.get_outputs()}
        result: dict[str, Any] = {}
        for name in self._state_inputs:
            output_name = name if name in outputs else f"new_{name}"
            if output_name not in outputs:
                raise ModelUnavailableError(f"Zipformer state output is missing for {name}")
            input_shape = self._shape(self._state_inputs[name])
            output_shape = self._metadata_shape(outputs[output_name])
            if input_shape is None or output_shape is None or len(output_shape) != len(input_shape):
                raise ModelUnavailableError(
                    f"Zipformer state output {output_name} has an incompatible rank"
                )
            if any(
                isinstance(output_dim, int) and output_dim != input_dim
                for input_dim, output_dim in zip(input_shape, output_shape, strict=True)
            ):
                raise ModelUnavailableError(
                    f"Zipformer state output {output_name} contradicts its input shape"
                )
            self._state_output_names[name] = output_name
            result[name] = outputs[output_name]
        return result

    @staticmethod
    def _metadata_shape(value: Any) -> tuple[int | str | None, ...] | None:
        """Return ONNX shape metadata while preserving symbolic dimensions."""
        shape = getattr(value, "shape", None)
        if not isinstance(shape, (list, tuple)):
            return None
        result: list[int | str | None] = []
        for dimension in shape:
            if dimension is None:
                result.append(None)
            elif isinstance(dimension, str) and dimension:
                result.append(dimension)
            elif isinstance(dimension, int) and not isinstance(dimension, bool) and dimension > 0:
                result.append(dimension)
            else:
                return None
        return tuple(result)

    def _logit_candidate_name(self) -> str:
        candidates = []
        for item in self._session.get_outputs():
            shape = self._metadata_shape(item)
            if shape is None or len(shape) != 3:
                continue
            vocabulary_axes = [
                axis
                for axis, dimension in enumerate(shape)
                if dimension == ZIPFORMER_VOCABULARY_SIZE
            ]
            batch_dimension = shape[0]
            batch_is_one_or_dynamic = (
                batch_dimension == 1 or batch_dimension is None or isinstance(batch_dimension, str)
            )
            if len(vocabulary_axes) == 1 and vocabulary_axes[0] != 0 and batch_is_one_or_dynamic:
                candidates.append(item.name)
        if len(candidates) != 1:
            raise ModelUnavailableError(
                "expected one rank-3 Zipformer output with a unique 251-token axis "
                f"and batch-first shape, got {candidates}"
            )
        return candidates[0]

    def _find_logit_output(self) -> str:
        return self._logit_candidate_name()

    @staticmethod
    def _official_feature_extractor(samples: np.ndarray, sample_rate: int) -> np.ndarray:
        """Compute the artifact's 16 kHz, 80-bin Kaldi/Povey fbank."""
        try:
            import kaldi_native_fbank as knf
        except Exception as exc:
            raise ModelUnavailableError(
                "Zipformer requires the installable kaldi-native-fbank runtime"
            ) from exc
        if sample_rate != 16_000:
            raise InvalidProviderOutputError("Zipformer fbank requires 16 kHz audio")
        options = knf.FbankOptions()
        options.frame_opts.samp_freq = 16_000
        options.frame_opts.frame_length_ms = 25.0
        options.frame_opts.frame_shift_ms = 10.0
        options.frame_opts.preemph_coeff = 0.97
        options.frame_opts.remove_dc_offset = True
        options.mel_opts.num_bins = 80
        options.mel_opts.low_freq = 20.0
        options.mel_opts.high_freq = 0.0
        options.frame_opts.dither = 0.0
        options.frame_opts.snip_edges = False
        options.frame_opts.window_type = "povey"
        options.frame_opts.round_to_power_of_two = True
        options.raw_energy = True
        options.use_energy = False
        options.energy_floor = 0.0
        fbank = knf.OnlineFbank(options)
        # kaldi-native-fbank follows Kaldi's int16-scaled waveform convention.
        fbank.accept_waveform(sample_rate, (samples.astype(np.float32) * 32768.0).tolist())
        array = np.asarray(
            [fbank.get_frame(index) for index in range(fbank.num_frames_ready)], dtype=np.float32
        )
        if array.ndim != 2 or array.shape[1] != 80:
            raise InvalidProviderOutputError(
                "Zipformer feature extractor returned a non-[frames,80] tensor"
            )
        return array

    def _initial_state(self) -> dict[str, np.ndarray]:
        state: dict[str, np.ndarray] = {}
        for name, item in self._state_inputs.items():
            shape = self._shape(item)
            if shape is None or any(d <= 0 for d in shape):
                raise ModelUnavailableError(
                    f"Zipformer state input {name} has dynamic/invalid shape {shape}"
                )
            state[name] = np.zeros(shape, dtype=self._numpy_dtype(item))
        return state

    @staticmethod
    def _numpy_dtype(value: Any) -> np.dtype:
        type_name = str(getattr(value, "type", "")).lower()
        dtype_by_onnx_type = {
            "tensor(float16)": np.dtype(np.float16),
            "tensor(float)": np.dtype(np.float32),
            "tensor(int32)": np.dtype(np.int32),
            "tensor(int64)": np.dtype(np.int64),
        }
        if type_name in dtype_by_onnx_type:
            return dtype_by_onnx_type[type_name]
        raise ModelUnavailableError(f"unsupported Zipformer ONNX tensor type: {type_name}")

    def _initial_processed_lens(self) -> np.ndarray:
        item = next(
            item for item in self._session.get_inputs() if item.name == self._processed_input
        )
        shape = self._shape(item)
        if shape is None:
            raise ModelUnavailableError("Zipformer processed_lens has a dynamic shape")
        dtype = self._numpy_dtype(item)
        if dtype not in (np.dtype(np.int32), np.dtype(np.int64)):
            raise ModelUnavailableError("Zipformer processed_lens must use int32 or int64")
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def _chunks(
        features: np.ndarray,
        *,
        input_frames: int = ZIPFORMER_INPUT_FRAMES,
        decode_chunk_len: int = ZIPFORMER_DECODE_CHUNK_LEN,
    ) -> list[np.ndarray]:
        if features.ndim != 2 or features.shape[1] != 80 or not np.all(np.isfinite(features)):
            raise InvalidProviderOutputError("Zipformer features must be finite [frames,80]")
        if input_frames <= decode_chunk_len:
            raise InvalidProviderOutputError(
                "Zipformer input window must exceed decode chunk length"
            )
        chunks: list[np.ndarray] = []
        for start in range(0, len(features), decode_chunk_len):
            chunk = features[start : start + input_frames]
            if len(chunk) < input_frames:
                chunk = np.pad(chunk, ((0, input_frames - len(chunk)), (0, 0)))
            chunks.append(chunk[None, ...].astype(np.float32, copy=False))
        return chunks

    @staticmethod
    def _chunk_starts(
        feature_count: int, *, decode_chunk_len: int = ZIPFORMER_DECODE_CHUNK_LEN
    ) -> list[int]:
        if feature_count <= 0 or decode_chunk_len <= 0:
            raise InvalidProviderOutputError("Zipformer needs positive feature and chunk lengths")
        return list(range(0, feature_count, decode_chunk_len))

    @staticmethod
    def _valid_output_frames(real_advance_frames: int) -> int:
        """Count non-padding outputs for a final partial 48-frame advance.

        The pinned model card specifies a 0.48-second advance for 12 output
        frames, i.e. one CTC frame per four 10 ms fbank frames. Padded output
        positions beyond the real final advance are not emitted as evidence.
        """
        if real_advance_frames <= 0:
            return 0
        if real_advance_frames > ZIPFORMER_DECODE_CHUNK_LEN:
            raise InvalidProviderOutputError("Zipformer decode advance exceeds 48 frames")
        return min(
            ZIPFORMER_OUTPUT_FRAMES,
            (real_advance_frames + 3) // 4,
        )

    def __call__(self, audio: AudioBuffer, group: BreathGroup) -> list[PhonemeEmission]:
        if audio.sample_rate != 16_000 or group.end > audio.duration:
            raise InvalidProviderOutputError(
                "Zipformer requires 16 kHz audio inside the breath bounds"
            )
        start = round(group.start * audio.sample_rate)
        end = round(group.end * audio.sample_rate)
        features = self._feature_extractor(np.asarray(audio.samples[start:end]), audio.sample_rate)
        chunks = self._chunks(features)
        starts = self._chunk_starts(len(features))
        if len(chunks) != len(starts):
            raise InvalidProviderOutputError("Zipformer chunk/start accounting is inconsistent")
        state = self._initial_state()
        processed_lens = self._initial_processed_lens()
        emissions: list[PhonemeEmission] = []
        frame_offset = 0
        previous_raw_token: int | None = None
        output_names = [item.name for item in self._session.get_outputs()]
        for chunk_start, chunk in zip(starts, chunks, strict=True):
            feeds = {self._x_input: chunk, self._processed_input: processed_lens, **state}
            outputs = self._session.run(output_names, feeds)
            by_name = dict(zip(output_names, outputs, strict=True))
            logits = np.asarray(by_name[self._logit_output], dtype=np.float32)
            logits = self._normalize_logits(logits)
            if logits.shape[0] != 1 or logits.shape[2] != ZIPFORMER_VOCABULARY_SIZE:
                raise InvalidProviderOutputError(
                    "Zipformer logits must have runtime shape [1,T,251]"
                )
            if logits.shape[1] != ZIPFORMER_OUTPUT_FRAMES:
                raise InvalidProviderOutputError(
                    "Zipformer must return 12 CTC frames for each 48-frame call"
                )
            if not np.all(np.isfinite(logits)):
                raise InvalidProviderOutputError("Zipformer logits contain non-finite values")
            real_advance = min(ZIPFORMER_DECODE_CHUNK_LEN, len(features) - chunk_start)
            valid_output_frames = self._valid_output_frames(real_advance)
            ids = np.argmax(logits, axis=-1)
            probs = self._softmax(logits)
            for local_frame, token_id in enumerate(ids[0, :valid_output_frames].tolist()):
                token_id = int(token_id)
                if token_id == ZIPFORMER_BLANK_ID:
                    previous_raw_token = ZIPFORMER_BLANK_ID
                    continue
                if token_id == previous_raw_token:
                    previous = emissions.pop()
                    emissions.append(
                        PhonemeEmission(
                            token_id,
                            previous.start_frame,
                            previous.end_frame + 1,
                            max(previous.score, float(probs[0, local_frame, token_id])),
                        )
                    )
                else:
                    emissions.append(
                        PhonemeEmission(
                            token_id,
                            frame_offset + local_frame,
                            frame_offset + local_frame + 1,
                            float(probs[0, local_frame, token_id]),
                        )
                    )
                previous_raw_token = token_id
            for name in state:
                value = np.asarray(by_name[self._state_output_names[name]])
                if (
                    value.shape != state[name].shape
                    or value.dtype != state[name].dtype
                    or not np.all(np.isfinite(value))
                ):
                    raise InvalidProviderOutputError(
                        f"Zipformer state output {name} has invalid shape or values"
                    )
                state[name] = value
            processed_value = np.asarray(by_name[self._processed_output])
            if (
                processed_value.shape != processed_lens.shape
                or processed_value.dtype != processed_lens.dtype
                or not np.all(np.isfinite(processed_value))
            ):
                raise InvalidProviderOutputError("Zipformer processed_lens output is invalid")
            processed_lens = processed_value
            frame_offset += valid_output_frames
        return emissions

    @staticmethod
    def _normalize_logits(logits: np.ndarray) -> np.ndarray:
        if logits.ndim != 3:
            raise InvalidProviderOutputError("Zipformer CTC output must be a rank-3 tensor")
        vocabulary_axes = [
            axis for axis in (1, 2) if logits.shape[axis] == ZIPFORMER_VOCABULARY_SIZE
        ]
        if len(vocabulary_axes) != 1:
            raise InvalidProviderOutputError(
                "Zipformer runtime output must contain one unambiguous 251-token axis"
            )
        if vocabulary_axes[0] == 2:
            return logits
        return np.transpose(logits, (0, 2, 1))

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        values = np.exp(shifted)
        return values / np.sum(values, axis=-1, keepdims=True)

    def close(self) -> None:
        self._session = None


class SherpaZipformerBackend:
    """Run the public Zipformer CTC artifact through sherpa-onnx."""

    def __init__(
        self,
        artifact_dir: str | Path,
        *,
        model_name: str = PUBLIC_ZIPFORMER_MODEL,
        tokens_name: str = PUBLIC_ZIPFORMER_TOKENS,
        recognizer_factory: Callable[..., Any] | None = None,
        num_threads: int = 4,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.model_path = self.artifact_dir / model_name
        self.tokens_path = self.artifact_dir / tokens_name
        if not self.model_path.is_file() or not self.tokens_path.is_file():
            raise ModelUnavailableError("public Zipformer ONNX model or tokens.txt is missing")
        try:
            self._tokens = ZipformerOnnxBackend._parse_token_table(
                self.tokens_path.read_text(encoding="utf-8")
            )
        except Exception as exc:
            raise ModelUnavailableError("unable to parse public Zipformer tokens.txt") from exc
        if len(self._tokens) != ZIPFORMER_VOCABULARY_SIZE:
            raise ModelUnavailableError("public Zipformer tokens.txt must contain 251 tokens")
        if self._tokens[ZIPFORMER_BLANK_ID] != "<blank>":
            raise ModelUnavailableError("public Zipformer blank token metadata is invalid")
        self._token_ids = {token: index for index, token in enumerate(self._tokens)}
        self._num_threads = num_threads
        self._recognizer_factory = recognizer_factory or self._default_recognizer_factory
        self._recognizer: Any | None = None

    @property
    def token_table(self) -> tuple[str, ...]:
        """Return the verified token table used by the public recognizer."""
        return tuple(self._tokens)

    def _default_recognizer_factory(self, *, tokens: str, model: str) -> Any:
        try:
            import sherpa_onnx

            return sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
                tokens=tokens,
                model=model,
                num_threads=self._num_threads,
                sample_rate=16_000,
                feature_dim=80,
                enable_endpoint_detection=False,
                decoding_method="greedy_search",
            )
        except Exception as exc:
            raise ModelUnavailableError(
                "sherpa-onnx is required for the public Zipformer backend"
            ) from exc

    def _get_recognizer(self) -> Any:
        if self._recognizer is None:
            self._recognizer = self._recognizer_factory(
                tokens=str(self.tokens_path), model=str(self.model_path)
            )
        return self._recognizer

    def __call__(self, audio: AudioBuffer, group: BreathGroup) -> list[PhonemeEmission]:
        if audio.sample_rate != 16_000 or group.end > audio.duration:
            raise InvalidProviderOutputError(
                "public Zipformer requires 16 kHz audio inside the breath bounds"
            )
        start = round(group.start * audio.sample_rate)
        end = round(group.end * audio.sample_rate)
        recognizer = self._get_recognizer()
        stream = recognizer.create_stream()
        stream.accept_waveform(
            audio.sample_rate, np.asarray(audio.samples[start:end], dtype=np.float32)
        )
        stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        tokens = list(recognizer.tokens(stream))
        timestamps = list(recognizer.timestamps(stream))
        probabilities = list(recognizer.ys_probs(stream))
        if len(tokens) != len(timestamps):
            raise InvalidProviderOutputError(
                "sherpa-onnx returned mismatched tokens and timestamps"
            )
        emissions: list[PhonemeEmission] = []
        previous_end = 0
        for index, (token, timestamp) in enumerate(zip(tokens, timestamps, strict=True)):
            if token not in self._token_ids:
                raise InvalidProviderOutputError(f"sherpa-onnx returned unknown token {token!r}")
            if not np.isfinite(timestamp) or timestamp < 0:
                raise InvalidProviderOutputError("sherpa-onnx returned invalid token timestamps")
            start_frame = max(previous_end, round(float(timestamp) / ZIPFORMER_FRAME_DURATION_SEC))
            next_timestamp = timestamps[index + 1] if index + 1 < len(timestamps) else None
            end_frame = (
                round(float(next_timestamp) / ZIPFORMER_FRAME_DURATION_SEC)
                if next_timestamp is not None
                else start_frame + 1
            )
            end_frame = max(start_frame + 1, end_frame)
            score = float(probabilities[index]) if index < len(probabilities) else 1.0
            if not np.isfinite(score):
                raise InvalidProviderOutputError("sherpa-onnx returned an invalid token score")
            emissions.append(
                PhonemeEmission(
                    self._token_ids[token], start_frame, end_frame, min(1.0, max(0.0, score))
                )
            )
            previous_end = end_frame
        return emissions

    def close(self) -> None:
        self._recognizer = None
