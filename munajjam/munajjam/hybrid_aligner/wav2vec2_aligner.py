"""Constrained CTC forced alignment with no heavyweight import at module load."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from munajjam.exceptions import AlignmentError, InvalidProviderOutputError, ModelUnavailableError
from munajjam.hybrid_aligner.types import AlignmentSpan, AudioBuffer, BreathGroup

WAV2VEC2_REPOSITORY = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic"
WAV2VEC2_REVISION = "af46c2d8531b8dcbb5e23b952f739b372c2e5d2d"
WAV2VEC2_SAMPLE_RATE = 16_000
WAV2VEC2_VOCABULARY_SIZE = 51


class TransformersWav2Vec2LogitsProvider:
    """Production Wav2Vec2ForCTC adapter with pinned, explicit model loading.

    Heavy dependencies are imported only when this provider is instantiated. No
    synthetic logits or alternate model fallback is permitted.
    """

    def __init__(
        self,
        *,
        model_id: str = WAV2VEC2_REPOSITORY,
        revision: str = WAV2VEC2_REVISION,
        device: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        if model_id != WAV2VEC2_REPOSITORY or revision != WAV2VEC2_REVISION:
            raise ValueError("only the pinned Wav2Vec2 model and revision are supported")
        try:
            import torch
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        except Exception as exc:
            raise ModelUnavailableError(
                "TransformersWav2Vec2LogitsProvider requires torch and transformers"
            ) from exc
        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if self._device == "cuda" and not torch.cuda.is_available():
            raise ModelUnavailableError("CUDA was requested but is unavailable")
        kwargs = {"revision": revision}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        try:
            self.processor = Wav2Vec2Processor.from_pretrained(model_id, **kwargs)
            self.model = Wav2Vec2ForCTC.from_pretrained(model_id, **kwargs).to(self._device)
        except Exception as exc:
            raise ModelUnavailableError("failed to load pinned Wav2Vec2 model") from exc
        self.model.eval()
        vocab_size = int(getattr(self.model.config, "vocab_size", -1))
        if vocab_size != WAV2VEC2_VOCABULARY_SIZE:
            raise ModelUnavailableError(f"unexpected Wav2Vec2 vocabulary size: {vocab_size}")
        if int(getattr(self.model.config, "pad_token_id", -1)) != 0:
            raise ModelUnavailableError("pinned Wav2Vec2 CTC blank/pad ID must be 0")
        sampling_rate = int(getattr(self.processor.feature_extractor, "sampling_rate", -1))
        if sampling_rate != WAV2VEC2_SAMPLE_RATE:
            raise ModelUnavailableError(f"unexpected Wav2Vec2 sample rate: {sampling_rate}")

    @staticmethod
    def _resample(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate == target_rate:
            return np.asarray(samples, dtype=np.float32)
        try:
            import librosa

            return np.asarray(
                librosa.resample(samples, orig_sr=source_rate, target_sr=target_rate),
                dtype=np.float32,
            )
        except Exception as exc:
            raise InvalidProviderOutputError("audio resampling requires librosa") from exc

    def __call__(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        array = np.asarray(samples, dtype=np.float32)
        if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
            raise InvalidProviderOutputError("Wav2Vec2 audio must be finite mono samples")
        array = self._resample(array, sample_rate, WAV2VEC2_SAMPLE_RATE)
        try:
            inputs = self.processor(
                array,
                sampling_rate=WAV2VEC2_SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
            input_values = inputs.input_values.to(self._device)
            attention_mask = getattr(inputs, "attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self._device)
            with self._torch.inference_mode():
                outputs = self.model(input_values=input_values, attention_mask=attention_mask)
            logits = outputs.logits[0].detach().float().cpu().numpy()
        except Exception as exc:
            raise InvalidProviderOutputError("Wav2Vec2 inference failed") from exc
        logits = np.asarray(logits, dtype=np.float32)
        if (
            logits.ndim != 2
            or logits.shape[1] != WAV2VEC2_VOCABULARY_SIZE
            or not np.all(np.isfinite(logits))
        ):
            raise InvalidProviderOutputError("Wav2Vec2 returned malformed logits")
        return logits

    def encode_text(self, text: str) -> tuple[list[int], list[str]]:
        """Encode canonical Arabic with the pinned tokenizer, without ID bridging."""
        if not isinstance(text, str) or not text.strip():
            raise InvalidProviderOutputError("canonical reference text must be non-empty")
        tokenizer = self.processor.tokenizer
        try:
            encoded = tokenizer(text, add_special_tokens=False)
            token_ids = list(encoded["input_ids"])
            unk_id = getattr(tokenizer, "unk_token_id", None)
            if not token_ids or any(token_id == unk_id for token_id in token_ids):
                raise InvalidProviderOutputError(
                    "Wav2Vec2 tokenizer produced an empty or unknown canonical reference"
                )
            if any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id < 0
                or token_id >= WAV2VEC2_VOCABULARY_SIZE
                or token_id == 0
                for token_id in token_ids
            ):
                raise InvalidProviderOutputError(
                    "Wav2Vec2 tokenizer emitted an invalid or blank target ID"
                )
            token_texts = tokenizer.convert_ids_to_tokens(token_ids)
            if isinstance(token_texts, str):
                token_texts = [token_texts]
            if len(token_texts) != len(token_ids) or any(
                not isinstance(token, str) or not token for token in token_texts
            ):
                raise InvalidProviderOutputError("Wav2Vec2 tokenizer returned malformed symbols")
            return token_ids, list(token_texts)
        except InvalidProviderOutputError:
            raise
        except Exception as exc:
            raise InvalidProviderOutputError("Wav2Vec2 canonical text tokenization failed") from exc

    def close(self) -> None:
        self.model = None
        self.processor = None


def load_audio_file(audio: str | Path, sample_rate: int = WAV2VEC2_SAMPLE_RATE) -> AudioBuffer:
    """Load real audio as mono float32 and resample it for the runtime."""
    try:
        import soundfile as sf

        samples, source_rate = sf.read(str(audio), dtype="float32", always_2d=False)
    except Exception as exc:
        raise InvalidProviderOutputError("loading real audio requires soundfile") from exc
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    samples = TransformersWav2Vec2LogitsProvider._resample(samples, int(source_rate), sample_rate)
    return AudioBuffer(samples=samples, sample_rate=sample_rate)


@dataclass(frozen=True, slots=True)
class Wav2Vec2Config:
    blank_id: int = 0
    post_roll_sec: float = 0.20
    max_frames: int = 20_000
    max_tokens: int = 2_000

    def __post_init__(self) -> None:
        if isinstance(self.blank_id, bool) or self.blank_id < 0:
            raise ValueError("blank_id must be non-negative")
        if not math.isfinite(self.post_roll_sec) or self.post_roll_sec <= 0:
            raise ValueError("post_roll_sec must be positive")
        if self.max_frames <= 0 or self.max_tokens <= 0:
            raise ValueError("resource limits must be positive")


class Wav2Vec2ForcedAligner:
    """Provider-injected CTC aligner; processor/model construction is deferred."""

    def __init__(
        self,
        *,
        logits_provider: Callable[[np.ndarray, int], np.ndarray] | None = None,
        logits_provider_factory: Callable[[], Callable[[np.ndarray, int], np.ndarray]]
        | None = None,
        config: Wav2Vec2Config | None = None,
    ) -> None:
        if logits_provider is not None and logits_provider_factory is not None:
            raise ValueError("provide logits_provider or logits_provider_factory, not both")
        self.config = config or Wav2Vec2Config()
        self._provider_factory = logits_provider_factory
        self._provider: Callable[[np.ndarray, int], np.ndarray] | None = logits_provider
        self._lock = threading.Lock()

    def _get_provider(self) -> Callable[[np.ndarray, int], np.ndarray]:
        if self._provider is None:
            with self._lock:
                if self._provider is None:
                    if self._provider_factory is None:
                        raise ModelUnavailableError(
                            "Wav2Vec2 runtime is optional; inject a logits provider"
                        )
                    self._provider = self._provider_factory()
        return self._provider

    def encode_text(self, text: str) -> tuple[list[int], list[str]]:
        """Use the configured provider's tokenizer for a canonical reference."""
        provider = self._get_provider()
        encoder = getattr(provider, "encode_text", None)
        if not callable(encoder):
            raise ModelUnavailableError(
                "configured Wav2Vec2 provider does not expose its reference tokenizer"
            )
        token_ids, token_texts = encoder(text)
        if (
            not token_ids
            or len(token_ids) != len(token_texts)
            or any(
                isinstance(token_id, bool)
                or not isinstance(token_id, int)
                or token_id == self.config.blank_id
                for token_id in token_ids
            )
        ):
            raise InvalidProviderOutputError("canonical Wav2Vec2 target tokens are invalid")
        return list(token_ids), list(token_texts)

    def close(self) -> None:
        """Release any cached provider references."""
        provider = self._provider
        if provider is not None and hasattr(provider, "close"):
            provider.close()  # type: ignore[attr-defined]
        self._provider_factory = None
        self._provider = None

    @staticmethod
    def _log_softmax(logits: np.ndarray) -> np.ndarray:
        if (
            logits.ndim != 2
            or logits.shape[0] == 0
            or logits.shape[1] == 0
            or not np.all(np.isfinite(logits))
        ):
            raise InvalidProviderOutputError(
                "logits must be finite with shape [frames, vocabulary]"
            )
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))

    def _ctc(
        self, log_probs: np.ndarray, tokens: Sequence[int]
    ) -> list[tuple[int, int, int, float]]:
        frames, vocab = log_probs.shape
        if (
            len(tokens) == 0
            or len(tokens) > self.config.max_tokens
            or frames > self.config.max_frames
        ):
            raise AlignmentError("CTC input exceeds valid token/frame limits")
        if self.config.blank_id >= vocab or any(
            isinstance(t, bool) or not isinstance(t, int) or t < 0 or t >= vocab for t in tokens
        ):
            raise InvalidProviderOutputError("CTC token is outside logits vocabulary")
        extended = [self.config.blank_id]
        for token in tokens:
            extended.extend((token, self.config.blank_id))
        states = len(extended)
        if frames * states > self.config.max_frames * self.config.max_tokens:
            raise AlignmentError("CTC trellis cell budget exceeded")
        score = np.full((frames, states), -np.inf, dtype=np.float64)
        back = np.full((frames, states), -1, dtype=np.int32)
        score[0, 0] = log_probs[0, self.config.blank_id]
        if states > 1:
            score[0, 1] = log_probs[0, extended[1]]
        for t in range(1, frames):
            for s, token in enumerate(extended):
                candidates = [(score[t - 1, s], s)]
                if s > 0:
                    candidates.append((score[t - 1, s - 1], s - 1))
                if s > 1 and token != self.config.blank_id and token != extended[s - 2]:
                    candidates.append((score[t - 1, s - 2], s - 2))
                best, prev = max(candidates, key=lambda item: item[0])
                if math.isfinite(float(best)):
                    score[t, s] = best + log_probs[t, token]
                    back[t, s] = prev
        end_state = (
            states - 1 if score[-1, states - 1] >= score[-1, max(0, states - 2)] else states - 2
        )
        if not math.isfinite(float(score[-1, end_state])):
            raise AlignmentError("CTC target cannot be aligned in this breath group")
        path: list[tuple[int, int, float]] = []
        s = end_state
        for t in range(frames - 1, -1, -1):
            if extended[s] != self.config.blank_id:
                path.append((extended[s], t, float(np.exp(score[t, s]))))
            if t:
                s = int(back[t, s])
                if s < 0:
                    raise AlignmentError("invalid CTC backtrace")
        path.reverse()
        grouped: list[tuple[int, int, int, float]] = []
        for token, frame, prob in path:
            if grouped and grouped[-1][0] == token and grouped[-1][2] == frame:
                old_token, start_frame, _, old_prob = grouped[-1]
                grouped[-1] = (old_token, start_frame, frame + 1, max(old_prob, prob))
            else:
                grouped.append((token, frame, frame + 1, prob))
        return grouped

    def align_group(
        self,
        audio: AudioBuffer,
        group: BreathGroup,
        token_ids: Sequence[int],
        token_texts: Sequence[str],
        *,
        next_group_start: float | None = None,
    ) -> tuple[AlignmentSpan, ...]:
        if audio.sample_rate <= 0 or group.end > audio.duration:
            raise InvalidProviderOutputError("invalid audio or breath bounds")
        if len(token_ids) != len(token_texts) or not token_ids:
            raise InvalidProviderOutputError("token IDs and text must be non-empty and parallel")
        upper = min(audio.duration, group.end + self.config.post_roll_sec)
        if next_group_start is not None:
            upper = min(upper, next_group_start)
        if upper <= group.start:
            raise InvalidProviderOutputError("empty local trellis window")
        start_i, end_i = round(group.start * audio.sample_rate), round(upper * audio.sample_rate)
        samples = np.asarray(audio.samples[start_i:end_i], dtype=np.float32)
        logits = np.asarray(self._get_provider()(samples, audio.sample_rate))
        log_probs = self._log_softmax(logits)
        path = self._ctc(log_probs, token_ids)
        frame_count = log_probs.shape[0]
        spans: list[AlignmentSpan] = []
        target_idx = 0
        for token, start_frame, end_frame, probability in path:
            start = group.start + start_frame / frame_count * (upper - group.start)
            end = group.start + end_frame / frame_count * (upper - group.start)
            start, end = max(group.start, start), min(upper, end)
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                raise InvalidProviderOutputError("invalid CTC frame conversion")
            if token == self.config.blank_id:
                continue
            if target_idx >= len(token_ids) or token != token_ids[target_idx]:
                raise InvalidProviderOutputError(
                    "CTC path token does not match expected target sequence"
                )
            spans.append(
                AlignmentSpan(token_texts[target_idx], start, end, probability, "wav2vec2-ctc")
            )
            target_idx += 1
        if target_idx != len(token_ids):
            raise InvalidProviderOutputError("CTC path does not cover all target tokens")
        return tuple(spans)
