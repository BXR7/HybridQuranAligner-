# Issue #120 implementation status

Updated 2026-10-04. This work is **not a claim that Issue #120 is complete**: the gated Quran-Lab Zipformer phoneme mapping needed for canonical reference alignment is unavailable in this environment.

## Requirement status

| Phase | Status | Evidence and remaining work |
|---|---|---|
| A — Real breath segmentation | **PARTIAL** | Production defaults to the author API for `obadx/recitation-segmenter-v2@5ee90364e7090ea6eb9dffe80353bed06996a196`; its 16 kHz interval outputs are normalized and validated against `BreathGroup` bounds/order/duration rules. Runtime and weights load lazily; there is no production energy-VAD fallback. Adapter contracts are unit-tested, but no new live model smoke test was run here. |
| B — Canonical Quran reference | **PARTIAL** | Exact bundled Hafs/Warsh ayah text is resolved by explicit per-breath `CanonicalReferenceRequest`; source-file and selected-text SHA-256 digests are returned. Basmalah is opt-in and sourced from bundled 1:1 without duplicating Fatiha; Isti'adhah requires an explicitly sourced phrase with a matching digest. Canonical phoneme targets are not available in the bundled text resources. |
| C — Zipformer reference alignment | **BLOCKED** | The real pinned Zipformer runtime/emission path and approximately 40 ms frame grid were verified in prior work. The pinned model card lists `quran_text2phoneme.json` and `ordered_quran_phonemes.json`, but the gated JSON payloads returned HTTP 401 and no local cache/credentials are present. A typed provider/target contract exists; current outputs are marked `unaligned_phoneme_emissions`, with `zipformer_reference_alignment_completed=false`. No raw greedy output is called reference alignment. |
| D — Wav2Vec2 microscopic forced alignment | **PARTIAL** | Uses the pinned Wav2Vec2 provider/tokenizer, target-specific CTC trellis, per-breath audio slices, and token timing/confidence. The window is clipped at audio end and the next breath start, preventing cross-breath trellis leakage; a bounded 0.20 s post-roll permits trailing decay. Token spans retain their full consecutive-frame interval. It cannot yet be constrained by Zipformer-aligned evidence because Phase C is blocked. Existing real Wav2Vec2 inference evidence was not repeated in this task. |
| E — Orchestration | **PARTIAL** | The opt-in path runs segmentation, per-breath Zipformer emissions, canonical reference lookup/tokenization, then per-breath Wav2Vec2 CTC. Heavy runtimes load on demand. The caller explicitly maps one verse range to each detected breath; the code does not guess. Zipformer emissions are not fused with Wav2Vec2 spans or represented as aligned until an authoritative target map is available. `server.py` and the existing server/API contracts were not changed. |
| F — Tests | **DONE for deterministic contracts** | Focused hybrid tests: 152 passed. Full unit suite: 444 passed, 3 skipped. Ruff lint and format checks, import isolation, Python compilation, and `git diff --check` pass. These tests use deterministic fixtures for contract behavior and do not claim live model correctness. The real-model integration module was not rerun. |

## Runtime and reference details

- **Segmenter:** `obadx/recitation-segmenter-v2`, immutable Hub revision `5ee90364e7090ea6eb9dffe80353bed06996a196`; calls the author's `recitations-segmenter==1.0.0` `segment_recitations` and `clean_speech_intervals` API with 16 kHz audio and 30 ms cleanup/padding values. Pins for Torch `2.6.0`, Torchaudio `2.6.0`, and Transformers `4.51.3` are isolated in the optional `segmenter` extra.
- **Canonical text:** the repository's `quran_hafs.json` and `quran_warsh.json`, selected by the request's riwaya. Reference output records the JSON SHA-256, each part's `file#surah:ayah` source, and the combined text SHA-256. Empty Warsh placeholders are skipped by the data loader but remain unavailable by their original ayah number, so requests fail closed.
- **Isti'adhah:** not fabricated from ayah text. A request must include `VerifiedSpecialPhrase(kind="istiadhah", text=..., source=..., sha256=...)`; the text must exactly match the declared phrase after Arabic normalization and its digest must match. It is placed before the requested ayah passage.
- **Basmalah:** only included when explicitly requested. It is sourced from the selected riwaya's bundled ayah 1:1 and omitted as an extra when the requested passage already includes Fatiha 1:1.
- **Zipformer:** its validated per-breath emissions remain acoustic evidence only. The target provider contract cites only the pinned authoritative JSON filenames/revision and requires hashes; it deliberately does not implement an unverified parser or map Zipformer IDs to Wav2Vec2 IDs. Exact source findings are in [`docs/issue_120_external_sources.md`](docs/issue_120_external_sources.md).
- **Wav2Vec2 trellis and decay:** the aligner slices only `[breath.start, min(audio.end, breath.end + post_roll, next_breath.start)]`, builds a fresh CTC trellis for that breath's tokenizer-produced target, and converts the full token frame runs to audio timestamps. The next-group clip prevents the post-roll from crossing physical breath boundaries.

## Files changed

- `munajjam/munajjam/hybrid_aligner/recitation_segmenter.py` — pinned, lazy author-API adapter and production default.
- `munajjam/munajjam/hybrid_aligner/reference.py` — provenance-bearing Hafs/Warsh references, special phrase checks, and gated Zipformer target interface.
- `munajjam/munajjam/hybrid_aligner/hybrid_pipeline.py` — production factory/lazy loading, explicit canonical per-breath input, stage ordering, and honest Zipformer completion metadata.
- `munajjam/munajjam/hybrid_aligner/wav2vec2_aligner.py` — factory-lazy model/tokenizer and full CTC token-run timing.
- `munajjam/munajjam/hybrid_aligner/__init__.py` — public exports.
- `munajjam/munajjam/data/quran.py` — skip empty Warsh placeholders without renumbering.
- `munajjam/pyproject.toml` — exact optional segmenter runtime pins.
- `tests/unit/test_hybrid_aligner.py` — stage, canonical-input, and CTC timing coverage.
- `tests/unit/test_hybrid_reference.py` — segmenter adapter, provenance, special phrase, laziness, and failure-contract coverage.
- `README.md` — production usage, data policy, and limitations.
- `docs/issue_120_external_sources.md` — pinned external sources and gated-file findings.
- `ISSUE_120_IMPLEMENTATION_REPORT.md` — current status and evidence.

## Verification and known limitations

- Focused tests: **152 passed**.
- Full unit suite: **444 passed, 3 skipped**.
- Ruff lint: passed on changed Python files.
- Ruff format check: passed on changed Python files.
- Import isolation: importing `munajjam` and `munajjam.hybrid_aligner` loads none of Torch, Transformers, ONNX Runtime, Sherpa-ONNX, or the segmenter package.
- `py_compile` and `git diff --check`: passed.
- No global NumPy/Torch upgrade was performed. A project `uv sync` attempt could not build the unrelated declared native `ctc-segmentation` dependency because Python development headers are absent; tests used an isolated minimal venv. Real-model tests were not rerun. The production segmenter adapter is based on the model card and author's published API, but its live inference remains to be smoke-tested in an authorized model environment.
- The main functional blocker is authorized access to and schema verification of the pinned Zipformer canonical phoneme files. Until then, canonical Zipformer reference alignment and evidence-constrained fusion are **not implemented**; do not mark the whole issue complete.
