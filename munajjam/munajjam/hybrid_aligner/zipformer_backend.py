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
            if len(shape or ()) == 3 and shape[-1] == 80 and shape[-2] == 61:
                candidates.append(item.name)
        if candidates != ["x"]:
            raise ModelUnavailableError(
                f"expected one acoustic input named x with shape [N,61,80], got {candidates}"
            )
        return candidates[0]

    def _find_processed_lens(self) -> tuple[str, str]:
        inputs = {item.name for item in self._session.get_inputs()}
        outputs = {item.name for item in self._session.get_outputs()}
        if "processed_lens" not in inputs:
            raise ModelUnavailableError("Zipformer graph is missing processed_lens input")
        output = "new_processed_lens" if "new_processed_lens" in outputs else "processed_lens"
        if output not in outputs:
            raise ModelUnavailableError("Zipformer graph is missing processed_lens output")
        return "processed_lens", output

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
            self._state_output_names[name] = output_name
            result[name] = outputs[output_name]
        return result

    def _logit_candidate_name(self) -> str:
        candidates = []
        for item in self._session.get_outputs():
            shape = self._shape(item)
            if shape and ZIPFORMER_VOCABULARY_SIZE in shape:
                candidates.append(item.name)
        if len(candidates) != 1:
            raise ModelUnavailableError(
                f"could not identify one Zipformer [*,*,251] output: {candidates}"
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
        options.mel_opts.num_bins = 80
        options.frame_opts.dither = 0.0
        options.frame_opts.snip_edges = False
        options.frame_opts.window_type = "povey"
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
        if "float16" in type_name:
            return np.dtype(np.float16)
        if "float" in type_name:
            return np.dtype(np.float32)
        if "int32" in type_name:
            return np.dtype(np.int32)
        if "int64" in type_name:
            return np.dtype(np.int64)
        raise ModelUnavailableError(f"unsupported Zipformer ONNX tensor type: {type_name}")

    def _initial_processed_lens(self) -> np.ndarray:
        item = next(
            item for item in self._session.get_inputs() if item.name == self._processed_input
        )
        shape = self._shape(item)
        if shape is None:
            raise ModelUnavailableError("Zipformer processed_lens has a dynamic shape")
        return np.zeros(shape, dtype=np.int32)

    @staticmethod
    def _chunks(
        features: np.ndarray, *, input_frames: int = 61, decode_chunk_len: int = 48
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

    def __call__(self, audio: AudioBuffer, group: BreathGroup) -> list[PhonemeEmission]:
        if audio.sample_rate != 16_000 or group.end > audio.duration:
            raise InvalidProviderOutputError(
                "Zipformer requires 16 kHz audio inside the breath bounds"
            )
        start = round(group.start * audio.sample_rate)
        end = round(group.end * audio.sample_rate)
        features = self._feature_extractor(np.asarray(audio.samples[start:end]), audio.sample_rate)
        chunks = self._chunks(features, input_frames=61, decode_chunk_len=48)
        state = self._initial_state()
        processed_lens = self._initial_processed_lens()
        emissions: list[PhonemeEmission] = []
        frame_offset = 0
        output_names = [item.name for item in self._session.get_outputs()]
        for chunk in chunks:
            feeds = {self._x_input: chunk, self._processed_input: processed_lens, **state}
            outputs = self._session.run(output_names, feeds)
            by_name = dict(zip(output_names, outputs, strict=True))
            logits = np.asarray(by_name[self._logit_output], dtype=np.float32)
            if logits.ndim != 3 or ZIPFORMER_VOCABULARY_SIZE not in logits.shape:
                raise InvalidProviderOutputError("Zipformer logits do not contain a 251-token axis")
            logits = self._normalize_logits(logits)
            if not np.all(np.isfinite(logits)):
                raise InvalidProviderOutputError("Zipformer logits contain non-finite values")
            ids = np.argmax(logits, axis=-1)
            probs = self._softmax(logits)
            for local_frame, token_id in enumerate(ids[0].tolist()):
                token_id = int(token_id)
                if token_id == ZIPFORMER_BLANK_ID:
                    continue
                if (
                    emissions
                    and emissions[-1].token_id == token_id
                    and emissions[-1].end_frame == frame_offset + local_frame
                ):
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
            for name in state:
                value = np.asarray(by_name[self._state_output_names[name]])
                if value.shape != state[name].shape or not np.all(np.isfinite(value)):
                    raise InvalidProviderOutputError(
                        f"Zipformer state output {name} has invalid shape or values"
                    )
                state[name] = value
            processed_value = np.asarray(by_name[self._processed_output])
            if processed_value.shape != processed_lens.shape or not np.all(
                np.isfinite(processed_value)
            ):
                raise InvalidProviderOutputError("Zipformer processed_lens output is invalid")
            processed_lens = processed_value.astype(np.int32, copy=False)
            frame_offset += logits.shape[1]
        return emissions

    @staticmethod
    def _normalize_logits(logits: np.ndarray) -> np.ndarray:
        if logits.shape[-1] == ZIPFORMER_VOCABULARY_SIZE:
            return logits
        if logits.shape[1] == ZIPFORMER_VOCABULARY_SIZE:
            return np.transpose(logits, (0, 2, 1))
        raise InvalidProviderOutputError("Zipformer output layout is not [N,T,251] or [N,251,T]")

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        values = np.exp(shifted)
        return values / np.sum(values, axis=-1, keepdims=True)

    def close(self) -> None:
        self._session = None
