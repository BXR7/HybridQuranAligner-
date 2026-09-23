"""Constrained CTC forced alignment with no heavyweight import at module load."""

from __future__ import annotations

import math
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from munajjam.exceptions import AlignmentError, InvalidProviderOutputError, ModelUnavailableError
from munajjam.hybrid_aligner.types import AlignmentSpan, AudioBuffer, BreathGroup


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
        config: Wav2Vec2Config | None = None,
    ) -> None:
        self.config = config or Wav2Vec2Config()
        self._provider_factory = logits_provider
        self._provider: Callable[[np.ndarray, int], np.ndarray] | None = None
        self._lock = threading.Lock()

    def _get_provider(self) -> Callable[[np.ndarray, int], np.ndarray]:
        if self._provider is None:
            with self._lock:
                if self._provider is None:
                    if self._provider_factory is None:
                        raise ModelUnavailableError(
                            "Wav2Vec2 runtime is optional; inject a logits provider"
                        )
                    self._provider = self._provider_factory
        return self._provider

    def close(self) -> None:
        """Release any cached provider references."""
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

    def _ctc(self, log_probs: np.ndarray, tokens: Sequence[int]) -> list[tuple[int, int, float]]:
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
        grouped: list[tuple[int, int, float]] = []
        for token, frame, prob in path:
            if grouped and grouped[-1][0] == token and grouped[-1][1] == frame - 1:
                old = grouped[-1]
                grouped[-1] = (token, frame, max(old[2], prob))
            else:
                grouped.append((token, frame, prob))
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
        for index, (token, frame, probability) in enumerate(path):
            end_frame = path[index + 1][1] if index + 1 < len(path) else frame + 1
            start = group.start + frame / frame_count * (upper - group.start)
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
