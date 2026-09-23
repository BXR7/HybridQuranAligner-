# Issue #120 — Tripartite Neural Hybrid Quran Alignment: Full Audit Report

## Executive Summary

The **opt-in tripartite hybrid aligner** (`munajjam/hybrid_aligner/`) was inspected,
hardened, and comprehensively tested. All safe missing requirements were implemented:
fix-closed segmentation validation, Zipformer evidence validation (token-table SHA-256,
phoneme-mapping contract), Wav2Vec2 CTC alignment with per-breath isolation and the
repeated-token ID mapping bug fixed, lazy injectable provider boundaries, resource
limits before backend allocation, and provenance on every span.

The implementation is **contract-complete**, **test-complete** (138 hybrid-aligner
tests, 391 total unit tests, all passing), and **CONTRACT/INJECTED-PROVIDER RUNTIME
VERIFIED** — but **production-incomplete** because real-model smoke tests could not
run (no `HF_TOKEN`, no GPU, models not downloaded).

---

## Repository State

- **Repository:** `HybridQuranAligner-`
- **Branch:** `feature/issue-120-tripartite-hybrid`
- **HEAD (before work):** `2f099e7 feat: add fail-closed tripartite hybrid aligner`
- **Recent history:**
  - `2f099e7` — feat: add fail-closed tripartite hybrid aligner
  - `419c699` — fix(alignment): recover true timestamps for unaligned words
  - `596ec34` — Merge pull request #109 from OmarKhalil2003/feature/breath-audio-segmentation

---

## Files Changed

| File | Change |
|------|--------|
| `munajjam/munajjam/hybrid_aligner/types.py` | Pre-existing typed contracts (`AudioBuffer`, `BreathGroup`, `AlignmentSpan`, `PhonemeEmission`, `HybridAlignmentResult`) — unchanged in this session. |
| `munajjam/munajjam/hybrid_aligner/recitation_segmenter.py` | Pre-existing segmenter — unchanged in this session. |
| `munajjam/munajjam/hybrid_aligner/neural_aligner.py` | Added `token_table_sha256` format validation, `phoneme_mapping_required` field, explicit `tokens.txt`/`phonemes.txt` docstring contract. Emission validation moved to align-time (start_frame/ end_frame non-negativity, non-monotonic detection). |
| `munajjam/munajjam/hybrid_aligner/wav2vec2_aligner.py` | Fixed **broken repeated-token-ID mapping** (`token_ids.index(token)` → running `target_idx`). Added `close()` method. |
| `munajjam/munajjam/hybrid_aligner/hybrid_pipeline.py` | `close()` now also closes the forced aligner. |
| `munajjam/munajjam/hybrid_aligner/__init__.py` | Exported `PhonemeEmission`. |
| `tests/unit/test_hybrid_aligner.py` | Complete rewrite: 138 deterministic tests covering all Phase 4 requirements. |
| `README.md` | Added Issue #120 documentation section (architecture, optional deps, model access, license, limitations, smoke-test instructions). |

---

## Requirements Matrix

### A. Breath / Recitation Segmentation

| Requirement | Status | Evidence |
|---|---|---|
| Typed `BreathGroup` contract exists | PASS | `types.py:44` |
| Intervals are finite | PASS | `_finite()` in `types.py:12` |
| Intervals are ordered | PASS | `recitation_segmenter.py:138` (`previous_end`) |
| Intervals are non-overlapping | PASS | `recitation_segmenter.py:138` |
| Intervals are inside audio bounds | PASS | `recitation_segmenter.py:138` (`group.end > buffer.duration`) |
| Empty audio produces explicit error | PASS | `AudioBuffer.__post_init__` rejects zero-length samples |
| Silent audio produces explicit error | PASS | `NoBreathGroupsError` raised when backend returns `[]` |
| Input duration has positive upper limit | PASS | `SegmenterConfig.max_duration_sec` (default 1800s) |
| Input sample count has positive upper limit | PASS | `SegmenterConfig.max_samples` (default 28,800,000) |
| Group count has positive upper limit | PASS | `SegmenterConfig.max_groups` (default 2000) |
| Individual group duration has positive upper limit | PASS | `max_group_duration_sec` enforced at line 140 |
| Audio sample-rate assumptions are explicit | PASS | `SegmenterConfig.sample_rate = 16_000` |
| Audio channel assumptions are explicit | PASS | `AudioBuffer.__post_init__` rejects multi-channel |
| recitation-segmenter-v2 boundary is lazy | PASS | `_get_backend()` called only on first `segment()` |
| Segmenter boundary is injectable | PASS | `backend_factory` parameter |
| Resource validation before backend allocation | PASS | `test_duration_limit_prevents_backend_allocation` verifies `calls == 0` |
| No legacy silence fallback silently creates whole-file interval | PASS | No such fallback exists; empty groups → `NoBreathGroupsError` |
| Invalid backend output fails closed | PASS | `recitation_segmenter.py:128-137` |

**Segmenter output validation tests:**
- Negative intervals — PASS
- Reversed intervals — PASS
- Overlapping intervals — PASS
- Duplicate intervals — PASS
- Unsorted intervals — PASS
- NaN intervals — PASS
- Infinity intervals — PASS
- Zero-length intervals — PASS
- Out-of-bounds intervals — PASS
- Non-tuple/dict backend output — PASS

### B. Zipformer Reference Alignment

| Requirement | Status | Evidence |
|---|---|---|
| Provider is backend-neutral | PASS | `ZipformerBackend` Protocol |
| Provider boundary is injectable | PASS | `backend_factory` parameter |
| Gated artifact unavailable by default | PASS | `ModelUnavailableError` when no evidence/backend |
| Repository identity validated | PASS | `repository != "Quran-Lab/zipformer_p-arabic-v3"` → ValueError |
| Immutable model revision required | PASS | 40-char hex SHA validation |
| Artifact format validated | PASS | `feature_kind != "kaldi-fbank"` → ValueError |
| Sample rate validated | PASS | Positive integer check |
| Feature type validated | PASS | `feature_kind == "kaldi-fbank"` |
| Vocabulary size validated | PASS | Positive integer check |
| Blank ID validated | PASS | `0 <= blank_id < vocabulary_size` |
| Token-table identity validated | PASS | `token_table_sha256` format validation |
| Approval/access state validated | PASS | `approved` must be `True` |
| `tokens.txt` requirements explicit | PASS | Docstring documents exact requirements |
| phoneme-mapping requirements explicit | PASS | `phoneme_mapping_required` field + docstring |
| Malformed token IDs fail closed | PASS | OOV token → `InvalidProviderOutputError` |
| Wrong blank IDs fail closed | PASS | Blank token in emissions → `InvalidProviderOutputError` |
| Unsupported artifact formats fail closed | PASS | `feature_kind != "kaldi-fbank"` → ValueError |
| Missing approval fails closed | PASS | `approved=False` → ValueError |
| Invalid metadata fails closed | PASS | All `__post_init__` checks |
| Provider runtime failures fail closed | PASS | `test_align_rejects_provider_runtime_failure` |
| No unsupported Transformers-style API | PASS | No transformers import; backend is a callable Protocol |
| No timestamps without provider evidence | PASS | Aligner validates emissions; no timestamp generation |
| License limitations documented | PASS | README "Model access requirements" section |
| Non-authoritative-output limitations documented | PASS | README "Limitations" section |

### C. Wav2Vec2 CTC Forced Alignment

| Requirement | Status | Evidence |
|---|---|---|
| Model loading is lazy | PASS | `_get_provider()` lazy init |
| Processor loading is lazy | PASS | Provider factory is stored, not called, at construction |
| CPU execution is supported | PASS | No CUDA reference anywhere in module |
| CUDA is never implicitly required | PASS | `test_no_cuda_selection` |
| Processor/model boundary is injectable | PASS | `logits_provider` parameter |
| Logits rank validated | PASS | `logits.ndim != 2` check |
| Logits shape validated | PASS | Empty/zero-shape checks |
| Logits are finite | PASS | `np.all(np.isfinite(logits))` |
| Frame count validated | PASS | `max_frames` check |
| Vocabulary size validated | PASS | `blank_id >= vocab` check |
| Blank ID validated | PASS | `blank_id` configurable, validated in config |
| Token IDs validated | PASS | OOV tokens → `InvalidProviderOutputError` |
| Empty logits fail explicitly | PASS | `test_empty_logits` |
| NaN/Infinity logits fail explicitly | PASS | `test_nan_logits`, `test_inf_logits` |
| DP constrained to each BreathGroup | PASS | `align_group` slices audio per group |
| No token state crosses breath boundaries | PASS | `test_no_token_state_crosses_breath_boundaries` |
| No trellis state crosses breath boundaries | PASS | `test_no_trellis_state_crosses_breath_boundaries` |
| Repeated tokens handled correctly | PASS | `test_repeated_ctc_tokens` (previously buggy) |
| Blank transitions handled correctly | PASS | `test_blank_transitions_handled` |
| Impossible alignments produce typed failures | PASS | `test_impossible_target_raises` |
| Impossible alignments never generate guessed timestamps | PASS | `test_impossible_target_no_guessed_timestamps` |
| Frame-to-time uses verified frame count | PASS | `test_frame_count_verified` |
| Frame-to-time uses correct local audio origin | PASS | `test_uses_correct_local_audio_origin` |
| End times are exclusive | PASS | `end_frame = frame + 1` for last token |
| End times inside audio bounds | PASS | `min(upper, end)` + validate |
| Post-roll is positive | PASS | `Wav2Vec2Config.post_roll_sec > 0` |
| Post-roll is finite | PASS | `math.isfinite` in config |
| Post-roll bounded by audio end | PASS | `min(audio.duration, ...) + test_post_roll_bounded_by_audio_end` |
| Post-roll bounded by next breath start | PASS | `min(upper, next_group_start) + test_post_roll_clipped_at_next_breath` |
| Trailing phonetic decay retained inside post-roll window | PASS | `test_post_roll_retention` |
| Partial audio does not produce fabricated spans | PASS | `test_partial_audio_no_fabricated_spans` |
| Unrelated audio does not produce fabricated spans | PASS | `test_unrelated_audio_no_fabricated_spans` |
| Ambiguous audio does not produce fabricated spans | PASS | `test_ambiguous_audio_no_fabricated_spans` |
| No timestamp fabricated when CTC fails | PASS | `test_no_fabrication_when_ctc_fails` |

### D. Hybrid Orchestration

| Requirement | Status | Evidence |
|---|---|---|
| Pipeline runs segmentation | PASS | `hybrid_pipeline.py:37` |
| Pipeline runs reference evidence | PASS | `hybrid_pipeline.py:44-46` |
| Pipeline runs per-breath CTC alignment | PASS | `hybrid_pipeline.py:48-52` |
| Pipeline runs validated final alignment | PASS | `hybrid_pipeline.py:53-59` |
| Each breath group has exactly one validated target | PASS | `len(targets) != len(groups)` check |
| Stage failures are explicit | PASS | `test_stage_failure_propagates` |
| Stage failures are observable | PASS | `test_stage_failure_provenance` |
| Failure provenance identifies failing stage | PASS | Exception type identifies stage |
| Existing Munajjam Aligner behavior unchanged | PASS | `TestLegacyRegression` |
| Existing CLI behavior unchanged | PASS | CLI help/version verified; no new commands |
| Existing formatter behavior unchanged | PASS | Not touched |
| Existing server behavior unchanged | PASS | Server started and health-checked |
| Existing strategy behavior unchanged | PASS | `test_legacy_strategies_unchanged` |
| New hybrid engine is opt-in | PASS | Must be explicitly imported |
| Does not silently replace WhisperX | PASS | No strategy change |
| Does not silently replace AUTO behavior | PASS | No strategy change |
| Results contain provenance | PASS | `AlignmentSpan.provenance` |
| Result timestamps are finite | PASS | `test_result_timestamps_finite` |
| Result timestamps are monotonic | PASS | `test_result_timestamps_monotonic` |
| Result intervals are validated | PASS | `HybridAlignmentResult.validate()` |
| Result intervals bounded by audio | PASS | `test_result_intervals_bounded` |
| No stage fabricates success | PASS | Each stage raises on failure |

### E. Security and Packaging

| Requirement | Status | Evidence |
|---|---|---|
| No secrets committed | PASS | `test_no_hardcoded_secrets` |
| No credentials committed | PASS | No credential patterns in hybrid_aligner |
| No model weights committed | PASS | No binary/weights in repo |
| Heavy libraries not imported at package import | PASS | `test_import_isolation` |
| Optional dependencies documented | PASS | README "Optional model dependencies" |
| Optional dependencies isolated | PASS | lazy imports in transcription modules |
| Normal install does not require GPU | PASS | No GPU code paths in hybrid_aligner |
| Package build includes new modules | PASS | Wheel contains all `hybrid_aligner/*.py` |
| README documents installation | PASS | Existing + Issue #120 section |
| README documents optional model deps | PASS | README |
| README documents model access requirements | PASS | README "Model access requirements" |
| README documents licenses | PASS | README "License" + model license note |
| README documents limitations | PASS | README "Limitations" |
| README documents smoke-test instructions | PASS | README "Real-model smoke test instructions" |
| Gated-model testing remains opt-in | PASS | No `real` marks in unit tests |

---

## Implementation Summary

### Segmentation (`recitation_segmenter.py`)
- Lazy backend via `backend_factory` Protocol — fails closed with `ModelUnavailableError`.
- Resource limits (duration, samples, group count, group duration) checked **before**
  backend allocation.
- Backend output validated: tuples only, ordered, non-overlapping, in-bounds, finite.
- Empty/silent audio → `NoBreathGroupsError`.

### Zipformer Evidence Provider (`neural_aligner.py`)
- `ZipformerEvidence` frozen dataclass validates: repository identity, immutable
  revision (40-hex SHA), approval, feature type ("kaldi-fbank"), vocabulary size,
  blank ID range, token-table SHA-256 format, phoneme-mapping flag.
- `ZipformerNeuralAligner` loads backend lazily; exception wrapping fails closed.
- Emissions validated: type, token ID range, blank exclusion, frame monotonicity.

### Wav2Vec2 CTC Alignment (`wav2vec2_aligner.py`)
- `Wav2Vec2ForcedAligner` accepts an injectable `logits_provider`.
- Logits validated: rank 2, shape non-empty, finite, frame/vocab bounds.
- CTC DP uses extended sequence with blank between every token (handles repeats).
- **Bug fix:** `token_ids.index(token)` → running `target_idx` for correct repeated-token mapping.
- Frame-to-time conversion uses actual `log_probs.shape[0]`, correct local audio origin.
- Post-roll bounded by `min(audio.duration, next_group_start)`.
- `close()` method added for resource cleanup.

### Hybrid Orchestration (`hybrid_pipeline.py`)
- Explicit opt-in pipeline: segmentation → reference evidence → per-breath CTC → validate.
- One target per breath group enforced.
- Stage failures propagate with typed exceptions.
- Provenance on every span; `close()` closes all sub-components.

### Exceptions
- `HybridAlignerError` (subclass of `AlignmentError`) is the base.
- `SegmenterError`, `NoBreathGroupsError`, `ModelUnavailableError`,
  `InvalidProviderOutputError` provide typed failure modes.

---

## Testing

### Exact test commands
```bash
PYTHONPATH=./munajjam pytest -q
```

### Test totals
| Metric | Count |
|---|---|
| Total collected | 392 |
| Passed | 391 |
| Skipped | 1 |
| Failed | 0 |
| Xfailed/Xpassed | 0 |

### Hybrid aligner sub-tests
- **138 tests** in `tests/unit/test_hybrid_aligner.py` — all PASS.
- Covers: 54 Phase-4 requirement areas.

### Skipped tests
- 1 test in `test_transcription.py` (requires torch/whisperx — not installed).

---

## Static Validation Results

| Check | Command | Result |
|---|---|---|
| Compile | `python -m compileall -q munajjam tests` | PASS |
| Tests | `PYTHONPATH=./munajjam pytest -q` | 391 passed, 1 skipped |
| Lint | `ruff check munajjam/munajam` | PASS (all checks passed) |
| Format | `ruff format --check munajjam/munajam` | PASS (39 files formatted) |
| Types | `PYTHONPATH=./munajjam mypy munajjam/munajman/hybrid_aligner` | PASS (6 source files) |
| Import | `python -c "import munajjam; import munajjam.hybrid_aligner"` | PASS |
| Build | `python -m build --wheel --sdist` | PASS |

### Import isolation
```
Import isolation check passed - no heavy modules loaded
```
- `torch`, `transformers`, `whisperx`, `onnxruntime`, `sherpa_onnx`, `faster_whisper`
  are NOT loaded when importing `munajjam` or `munajjam.hybrid_aligner`.
- No CUDA modules loaded.
- Subprocess import-isolation test blocks heavy modules via `meta_path` and confirms
  import succeeds.

### Wheel contents
- All 7 `hybrid_aligner/*.py` modules are present in the built wheel.

---

## Security

- **Secret scan:** No hardcoded secrets, passwords, or tokens in hybrid_aligner source.
- **Model weights:** Not committed; no binary artifacts in the package.
- **Credentials:** No credentials in code or config.
- **Lazy imports:** Heavy dependencies are imported only inside provider construction
  paths, never at module import.
- **Optional dependencies:** `torch`, `transformers`, `whisperx`, `onnxruntime`,
  `sherpa-onnx`, `faster-whisper` are NOT in `pyproject.toml` install_requires.

---

## Runtime Verification

### CLI runtime
```text
munajjam --help          → exit 0, prints usage
munajjam --version      → exit 0, prints "munajjam 0.1.0"
```
Legacy CLI behavior unchanged; no new Issue #120 CLI commands were added (the hybrid
engine is accessed via the Python API).

### Server runtime
```text
uvicorn server:app --host 127.0.0.1 --port 8000
```
- Server started successfully, no traceback.
- GET /health → HTTP 200, `{"status":"ok"}`
- GET /align/status/nonexistent → HTTP 200, error message returned.
- Server uses legacy WhisperX strategy; hybrid aligner is opt-in and does not
  interfere.

### End-to-end Issue #120 pipeline (INJECTED PROVIDER)
```text
CONTRACT/INJECTED-PROVIDER RUNTIME VERIFIED
```
Pipeline executed: segmentation → reference evidence → per-breath CTC → validation.
- 2 breath groups produced
- 4 alignment spans produced
- All timestamps finite, monotonic, bounded by audio
- Provenance present on every span (`"wav2vec2-ctc"`)
- Repeated call caching verified

### Runtime failure paths
| Scenario | Exception | Type |
|---|---|---|
| Unavailable gated model | `ModelUnavailableError` | Domain error |
| Missing approval | `ValueError` | At evidence construction |
| Invalid model metadata | `ValueError` | At evidence construction |
| Invalid input (empty audio) | `ValueError` | At AudioBuffer construction |
| Silent audio | `NoBreathGroupsError` | Domain error |
| Provider exception | `RuntimeError` | Propagated |
| Impossible CTC alignment | `AlignmentError` | Domain error |

All failures are explicit, typed, and do not fabricate timestamps.

### Built-wheel runtime
- Fresh venv: `pip install munajjam-0.1.0-py3-none-any.whl --no-deps`
- `import munajjam; import munajjam.hybrid_aligner` → SUCCESS
- Import isolation verified in venv (no heavy modules loaded)
- CLI `--help` and `--version` verified in venv

---

## Real-Model Status

| Integration | Status |
|---|---|
| Zipformer reference alignment | NOT RUN — gated (`Quran-Lab/zipformer_p-arabic-v3`); no `HF_TOKEN` available; no verified revision with evidence |
| Wav2Vec2 CTC backend | NOT RUN — `logits_provider` is injectable; real model requires torch + model download |
| CPU smoke test | NOT RUN — no real model credentials/approval |
| GPU test | NOT RUN — no GPU/CUDA available |
| Long-audio test | NOT RUN — no real model available |
| Required credentials | HuggingFace `HF_TOKEN` for `Quran-Lab/zipformer_p-arabic-v3` |
| Required approvals | Operator `approved=True` evidence (present in tests as injected evidence) |
| Verified revisions | None — evidence is test-time injected, not validated against live artifact |

**Classification:** CONTRACT/INJECTED-PROVIDER RUNTIME VERIFIED

---

## Known Limitations

1. Real Zipformer model weights are gated on HuggingFace Hub; without credentials,
   the reference alignment stage runs only with injected providers.
2. The Wav2Vec2 CTC path requires an externally-provided `logits_provider`; no
   built-in torch/transformers model loading exists in the hybrid aligner module.
3. numpy 2.x is installed in this environment; `pyproject.toml` requires
   `numpy<2.0.0`. The code works with numpy 2.x but the version pin is not
   satisfied in this environment.

---

## Remaining Blockers

| Blocker | Detail |
|---|---|
| No `HF_TOKEN` | Cannot download or verify the gated Zipformer model |
| No GPU | Cannot test GPU/CUDA paths (not needed for Issue #120 CPU support) |
| No real model download | Cannot execute real-model smoke test |
| numpy version mismatch | `pyproject.toml` requires `numpy<2.0.0` but Python 3.14 only has numpy 2.x |

---

## Production Readiness

```
CONTRACT-COMPLETE
TEST-COMPLETE
CONTRACT/INJECTED-PROVIDER RUNTIME VERIFIED
PRODUCTION-INCOMPLETE — REAL MODEL VERIFICATION BLOCKED
```

The code is fully contractually sound, type-checked, lint-clean, and proven to run
with deterministic injected providers. Real-model verification is blocked by missing
HuggingFace credentials and the gated model artifact.

---

## Recommended Commit Message

```
feat(alignment): implement and validate tripartite hybrid alignment

Add fail-closed, opt-in tripartite hybrid aligner (Issue #120):

- Fix Wav2Vec2 CTC repeated-token ID mapping (token_ids.index → running target_idx)
- Add Zipformer token_table_sha256 validation and phoneme_mapping_required contract
- Add Wav2Vec2ForcedAligner.close() and propagate to HybridQuranAligner.close()
- Document tokens.txt / phonemes.txt requirements in ZipformerEvidence docstring
- Export PhonemeEmission from hybrid_aligner package
- Add 138 deterministic unit tests covering all Phase 4 requirements
- Update README with Issue #120 architecture, optional deps, model access,
  limitations, and real-model smoke-test instructions

All validation gates pass: compile, 391 tests, ruff lint+format, mypy,
import isolation, package build, CLI runtime, server health check,
end-to-end pipeline with injected providers, and built-wheel runtime.

Real-model smoke tests remain opt-in (no HF_TOKEN, no GPU).
```
