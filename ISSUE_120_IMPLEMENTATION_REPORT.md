# Issue #120 Implementation Report

## Scope

This branch adds an opt-in `HybridQuranAligner` under `munajjam.hybrid_aligner` without changing the existing WhisperX, `Aligner`, strategy, CLI, or server contracts.

The implementation provides dependency-free typed contracts, lazy provider boundaries, validated breath groups, a capability-gated Zipformer boundary, and constrained Wav2Vec2 CTC alignment with bounded post-roll. Heavy neural runtimes are intentionally not imported or loaded by ordinary package import or object construction.

## Added components

- `types.py`: `AudioBuffer`, `BreathGroup`, `PhonemeEmission`, `AlignmentSpan`, and `HybridAlignmentResult`.
- `recitation_segmenter.py`: injected/lazy segmenter with CPU-oriented input validation, monotonic interval checks, and resource limits.
- `neural_aligner.py`: Zipformer evidence contract and fail-closed gated provider boundary.
- `wav2vec2_aligner.py`: injected CTC logits provider, local per-breath trellis/backtrace, strict vocabulary checks, and bounded post-roll.
- `hybrid_pipeline.py`: explicit segment → reference evidence → per-breath CTC orchestration.
- `tests/unit/test_hybrid_aligner.py`: deterministic fixture tests for lazy loading, malformed intervals, gated Zipformer behavior, CTC bounds, and impossible paths.

## Validation performed

- Focused tests: **8 passed**.
- Existing unit suite: **261 passed, 1 skipped**.
- Ruff on new/changed implementation and tests: **passed**.
- mypy on `munajjam.hybrid_aligner`: **passed**.
- Python compilation: **passed**.
- Import isolation check: `munajjam` and `munajjam.hybrid_aligner` imported with no `torch`, `transformers`, `onnxruntime`, or `sherpa_onnx` loaded.
- `git diff --check`: **passed**.

## Environment note

The first full-suite attempt was blocked by missing project dependencies. After installing the declared runtime/test dependencies that were available, the suite completed with the result above. The declared `ctc-segmentation` package could not build in this sandbox because its native extension requires a compiler executable that is not installed; the passing unit suite does not require that optional/native path.

## Operational limitations

The real gated Zipformer artifact is not enabled by default and requires verified operator-supplied evidence plus an injected runtime adapter. No model weights, credentials, or fabricated Zipformer API are included. Real model smoke tests remain opt-in and should be run only with approved model access and pinned artifacts.
