# Issue #120 — Pinned Zipformer Artifact Investigation

**Status: REAL-MODEL PARTIALLY VERIFIED — BLOCKED**
**Branch:** `feature/issue-120-tripartite-hybrid`
**Baseline:** `05b07a5ada39480b3f125d73c387e5d340dc0b32`
**Investigation correction commit:** `5e4c1db14165327e471986fc6a047960c2c6607d`

No PR was opened. `server.py` was not changed.

## Conclusion / root cause

The earlier wording that implied the canonical Quran phoneme map was **missing** was wrong. At the pinned revision, the Hugging Face repository tree **does contain** the canonical maps, tokenizer inventory, CTC symbol table, evaluator scripts, exporter, and model weights. The issue is that the repository is **manually gated**: all no-token reads attempted for the map/tokenizer/source files returned HTTP 401. This environment has no Hugging Face access token and no local copies of the relevant files or weights.

Therefore:

- The blocker is **authorized artifact-content and model-weight access**, not a missing filename.
- The artifacts are expressly intended to provide canonical per-ayah phoneme references. Their exact JSON schemas, normalization, special cases, and token-ID construction cannot be read or verified here.
- No parser or token-ID sequence was guessed, no alternate model was substituted, and no `phonemes.txt` was invented.
- A production reference adapter and real inference cannot responsibly be completed until authorized artifact access is available.

## Pinned artifact inventory

Source: [pinned model card](https://huggingface.co/Quran-Lab/zipformer_p-arabic-v3/blob/506422c82a81c86e7ae74a5a2ab4641724bcd3b3/README.md), [pinned tree API](https://huggingface.co/api/models/Quran-Lab/zipformer_p-arabic-v3/tree/506422c82a81c86e7ae74a5a2ab4641724bcd3b3?recursive=true), and [model API](https://huggingface.co/api/models/Quran-Lab/zipformer_p-arabic-v3). The API reports `gated: manual`. The IDs below are Git blob IDs from tree metadata, **not** SHA-256 content hashes.

| File at pinned revision | Bytes | Git blob ID | Documented purpose |
|---|---:|---|---|
| `ordered_quran_phonemes.json` | 5,106,711 | `8dc0fab9f128cde16ccc2cb32de5eee42e3ce850` | Canonical phonemization of all 6,236 ayat; retrieval/grading |
| `quran_text2phoneme.json` | 3,881,782 | `eed2fa2a86c23e269f9e7f8c4db58c5b7302a5a0` | Evaluation text-to-phoneme lookup |
| `phoneme_units.json` | 3,350 | `9ea207e78a5fbce8dbefe23602697c65440ad7b8` | Phoneme tokenizer-unit inventory |
| `tokens.txt` | 2,346 | `5b0ed53f48819bac097568c596c668aa14d21f50` | CTC symbol-to-ID table; CTC blank is ID 250 |
| `decode_with_confidence.py` | 6,313 | `3b8ea5f037d81beff1a3d8d7ce1dfb95c0fd69d4` | CTC decoding with per-symbol confidence |
| `quran_per_eval.py` | 6,668 | `5b587f1592b8048fde94fa68c3ba687e9234f930` | PER evaluation and the full streaming loop |
| `quran_wer_retrieval.py` | 4,758 | `657fbb16478a0f224396d31d15a6a9b10f5cbae3` | Quran-lexicon phoneme retrieval/WER evaluation |
| `export_quran_streaming_onnx.py` | 4,376 | `b905ac3ce60b452c9069f201809c098f0c3c7f47` | Exact ONNX exporter |
| `packing_front.json` | 2,782 | `ed3f09eda71767a08657054e866ec7328ac4f3fd` | CoreML front-function packed state layout |
| `packing_back.json` | 2,896 | `f558cdf439b4fd956e5b49a4b2541ecec0b00845` | CoreML back-function packed state layout |
| `zipformer_p_arabic_v3.1.onnx` | 262,977,606 | `2ee39273e5033e9aebb040c442163cc768926a3d` | Streaming CTC ONNX model |
| `zipformer_p_arabic_v3.1.pt` | 263,577,418 | `24ead65c357476d8693462a7a881b609066cf784` | PyTorch checkpoint |

The pinned tree contains **no** file named `phonemes.txt`; the previous evidence docstring claim requiring one has been corrected. The exact schema-bearing scripts (`quran_per_eval.py`, `quran_wer_retrieval.py`, `decode_with_confidence.py`, and the exporter) are themselves gated. Direct no-token resolution returned HTTP 401 for each of the four map/tokenizer files and those scripts. No matching file or `.onnx`/`.pt` model was present in the inspected Hugging Face or Munajjam caches.

## What the existing Quran files mean

1. **`quran_text2phoneme.json`:** the model card identifies this as the text-to-phoneme lookup used in evaluation. The card does not reveal its keys, value shape, normalization, or special phrase encoding because the JSON is gated.
2. **`ordered_quran_phonemes.json`:** the model card describes canonical phonemization for every one of the 6,236 ayat, used for retrieval/grading. This is the strongest named artifact for a per-ayah reference sequence.
3. **`phoneme_units.json`:** the phoneme-unit inventory for the model tokenizer. It is not the CTC output-ID table.
4. **`tokens.txt`:** the model card explicitly says to use this CTC symbol table because raw `phoneme_units.json` IDs are offset by one from CTC output IDs; the CTC blank is 250. A target encoder must respect that offset/table rather than treating IDs as interchangeable.
5. **`packing_front.json` and `packing_back.json`:** these describe how cache state is packed for the **CoreML split front/back functions**. They are not a phoneme map and not a bridge to Wav2Vec2 IDs.
6. **Decoder/evaluator/export scripts:** `decode_with_confidence.py` handles CTC decoding/confidence; `quran_per_eval.py` documents the streaming evaluation loop; `quran_wer_retrieval.py` performs phoneme-to-Quran retrieval; `export_quran_streaming_onnx.py` is the exporter. Their exact implementation is unavailable without gate access.

### Can the pinned artifacts construct the canonical target?

**Yes, in design and by the model card’s stated purpose; no filename is missing.** Given the caller’s explicit per-breath verse reference, the `ordered_quran_phonemes.json`/`quran_text2phoneme.json` content can supply the canonical phoneme sequence, and `phoneme_units.json` plus `tokens.txt` can encode it in the Zipformer CTC space. This is the right production adapter path.

**It has not been implemented or verified here:** the payload schemas, ayah ordering/index rules, canonical-text normalization, special cases, and exact unit-to-CTC token mapping are not visible. Parsing or hard-coding those details from filenames alone would be guesswork. The project still requires one explicit verse-range request per detected breath; no verse-to-breath mapping is inferred. The card’s 6,236-ayah statement also does not, by itself, specify how standalone Isti'adhah or Basmalah should be phonemized.

A corroborating [Muno459 mirror card](https://huggingface.co/Muno459/zipformer_p-quran) describes similar use and provides a human-readable example of the streaming/PT calls. Its API tree has identical Git blob IDs to the pinned Quran-Lab tree for the three canonical JSON files and the PER/retrieval/export scripts. However, that mirror is also `gated: manual`; its `tokens.txt` is a **different** file (2,597 bytes, blob `7463d224e23a1e7f4a231f932ce3f87129ba1753`), and its weights are not the pinned weights. Its README is secondary corroboration only; it was not used as a substitute.

## Zipformer input/output contract

### Publicly documented acoustic/streaming contract

- **Audio/features:** 16 kHz mono, 80-dimensional **Kaldi fbank** with the stated Povey window; the model card warns against substituting a generic mel spectrogram.
- **Chunking:** streaming Zipformer2/CTC, 48 input frames of decode advance per 0.48 s chunk. The documented 61-frame input window carries context around that advance.
- **Output:** 12 CTC emission frames per full advance, with 251 output values per frame. Each emission frame corresponds to 40 ms. The decoder uses `tokens.txt`; blank ID is 250.
- **Ordinary ONNX shape/flow:** one fbank chunk `x` (documented mirror form `[1,T,80]`; pinned model card supplies the 61-frame window/48-frame advance), plus cache-state tensors and `processed_lens`; it returns `log_probs` plus updated `new_*` cache states. The same-script mirror README enumerates cache groups as `cached_key`, `nonlin_attn`, `val1`, `val2`, `conv1`, `conv2` per layer, `embed_states`, and `processed_lens`. Complete pinned ONNX state names, dimensions, and dtypes cannot be checked without the gated graph/evaluator.
- **Explicit CoreML split interface (do not confuse this with raw ONNX tensors):** `front` takes `x (1,61,80)`, `processed_lens (1,) int32`, and packed state blobs `sg_0..sg_12`; it returns an intermediate `(24,1,384)` tensor, `new_processed_lens`, and updated state. `back` takes that intermediate, the **pre-update** processed length, and `sg_0..sg_11`; it returns `log_probs (1,12,251)` and updated state. The card says both CoreML functions perform the same cache-aware streaming computation as the ONNX exports. `packing_front/back.json` apply to this split CoreML layout.

### PyTorch checkpoint

The pinned card lists the `.pt` file as averaged PyTorch weights but does not provide its exact Python call signature in accessible pinned content. The same-script mirror README shows an inference pattern of building the model from the phoneme-unit count, loading checkpoint key `model`, calling `model.encode(feats, feat_lens)`, and applying `model.ctc_head(enc)` to obtain frame IDs. That is useful corroboration, not direct verification of the pinned checkpoint’s state-dict keys or call parameters; the pinned PT file and its source instantiator remain gated. No PyTorch inference was run.

## 251 Zipformer tokens vs. 51 Wav2Vec2 tokens

They are **independent model-specific CTC vocabularies**. The pinned Zipformer has 251 output classes, phoneme units, and blank ID 250. The pinned Wav2Vec2 provider has 51 output classes, pad/blank ID 0, and its own Transformers tokenizer. Canonical Arabic text must be encoded separately with each model’s own authoritative resources. No upstream architecture requirement for a numeric 251↔51 mapping was found; no such mapping was created. The Zipformer token table’s offset warning concerns its **own** `phoneme_units.json` versus `tokens.txt`, not Wav2Vec2.

## Code changes, tests, and live verification

Correction commit `5e4c1db14165327e471986fc6a047960c2c6607d` was pushed to the same branch. It:

- changes runtime metadata from `blocked_missing_gated_phoneme_map` to the accurate `blocked_gated_artifact_access`;
- documents that the canonical files exist, but their contents are gated;
- removes the false `phonemes.txt` requirement from the Zipformer evidence docstring and marks that earlier audit claim superseded;
- records exact pinned tree IDs/sizes and the publicly available interface details in the repository evidence docs.

`server.py` is untouched; no PR was opened.

| Verification | Result |
|---|---|
| Focused hybrid/Zipformer unit tests | **193 passed** |
| Full `tests/unit` suite | **445 passed, 2 skipped** |
| Ruff check on modified implementation modules | **Passed** |
| Ruff check on hybrid test module | Passes with its existing `I001` import-order finding ignored; no unrelated import/style churn was made |
| Ruff format check on implementation modules | **Passed** |
| `git diff --check` | **Passed** |
| Real Zipformer inference smoke | **Not run** — gated map/model files returned 401; no local model weights/cache or authorized HF token |
| Real breath-segmentation smoke | **Not run in this investigation** |
| Real Wav2Vec2 smoke | **Not run in this investigation** |
| Real tripartite E2E on Quran audio | **Not run** — Zipformer access/schema remains blocked, and no real audio plus explicit per-breath verse assignments were supplied |
| `REAL_TRIPARTITE_E2E_EVIDENCE.json` | **Not created** |

**No successful real-model or E2E classification is claimed.** To finish, an operator authorized for the pinned Quran-Lab repository must obtain gate access and provide an authorized secret-backed Hugging Face token/cache. They should review and personally accept the model’s NPL-1.2 gate terms if applicable; no terms were accepted on the user’s behalf. Then schema-check the named JSON/scripts and model files, implement the adapter, and run real inference using real Quran audio with explicit verse ranges per breath. Until then, the correct classification remains **REAL-MODEL PARTIALLY VERIFIED — BLOCKED**.

## Source links

- [Pinned Quran-Lab model card and license](https://huggingface.co/Quran-Lab/zipformer_p-arabic-v3/blob/506422c82a81c86e7ae74a5a2ab4641724bcd3b3/README.md)
- [Pinned Quran-Lab repository tree API](https://huggingface.co/api/models/Quran-Lab/zipformer_p-arabic-v3/tree/506422c82a81c86e7ae74a5a2ab4641724bcd3b3?recursive=true)
- [Pinned Quran-Lab model access metadata](https://huggingface.co/api/models/Quran-Lab/zipformer_p-arabic-v3)
- [Muno459 mirror README, used only as corroboration](https://huggingface.co/Muno459/zipformer_p-quran)
- [Muno459 mirror metadata](https://huggingface.co/api/models/Muno459/zipformer_p-quran)
- [Upstream quran-transcript label source](https://github.com/obadx/quran-transcript)
