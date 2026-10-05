# Issue #120 external sources and pinned evidence

Research captured on 2026-10-04 and re-verified on 2026-10-05 for implementation of the hybrid Quran aligner.

## Upstream acceptance context

- [Itqan-community/Munajjam Issue #120](https://github.com/Itqan-community/Munajjam/issues/120) describes the pipeline roles: recitation/breath segmentation, Zipformer phoneme-level reference stage, then Wav2Vec2 CTC forced alignment at finer granularity. It calls out Isti'adhah and Basmalah handling. It does **not** define an automatic verse-to-breath assignment rule, a fusion algorithm, or an ID mapping between the Zipformer and Wav2Vec2 vocabularies.

## Production recitation segmenter

- Model: [obadx/recitation-segmenter-v2](https://huggingface.co/obadx/recitation-segmenter-v2)
- Immutable Hub revision resolved from the repository API: `5ee90364e7090ea6eb9dffe80353bed06996a196`.
- Model metadata identifies `Wav2Vec2BertForAudioFrameClassification`, `AutoFeatureExtractor`/`AutoModelForAudioFrameClassification`, and 16 kHz audio.
- Author API/library: [recitations-segmenter on PyPI](https://pypi.org/project/recitations-segmenter/) and [upstream repository](https://github.com/obadx/recitations-segmenter). Version `1.0.0` documents `segment_recitations(...)` followed by `clean_speech_intervals(..., min_silence_duration_ms=30, min_speech_duration_ms=30, pad_duration_ms=30, return_seconds=True)`.
- The package declares Python `>=3.10`, Torch `>=2.6.0`, Torchaudio, and Transformers `>=4.50.3`. The project isolates exact runtime pins in the optional `segmenter` extra rather than upgrading base dependencies.

## Zipformer reference artifact

- Model: [Quran-Lab/zipformer_p-arabic-v3](https://huggingface.co/Quran-Lab/zipformer_p-arabic-v3), pinned project revision `506422c82a81c86e7ae74a5a2ab4641724bcd3b3`.
- The revision-pinned model card documents a 251-symbol Hafs phoneme CTC output and explicitly states that `ordered_quran_phonemes.json` contains canonical phonemization for all 6,236 ayat, while `quran_text2phoneme.json` is a text-to-phoneme lookup used in evaluation. `phoneme_units.json` is the tokenizer-unit inventory. `tokens.txt` is the CTC output symbol table: the card warns that raw phoneme-unit IDs are offset from CTC output IDs and that the Zipformer blank is ID 250. These findings conclusively refute the earlier implication that no canonical mapping artifact exists.
- The pinned Hugging Face tree API confirms the following files exist at the exact revision. The `oid` values below are Git blob IDs from repository metadata, not SHA-256 content digests:

| Pinned file | Bytes | Git blob ID | Documented role |
|---|---:|---|---|
| `ordered_quran_phonemes.json` | 5,106,711 | `8dc0fab9f128cde16ccc2cb32de5eee42e3ce850` | Canonical phonemization of all 6,236 ayat; retrieval/grading |
| `quran_text2phoneme.json` | 3,881,782 | `eed2fa2a86c23e269f9e7f8c4db58c5b7302a5a0` | Evaluation text-to-phoneme lookup |
| `phoneme_units.json` | 3,350 | `9ea207e78a5fbce8dbefe23602697c65440ad7b8` | Phoneme tokenizer-unit inventory |
| `tokens.txt` | 2,346 | `5b0ed53f48819bac097568c596c668aa14d21f50` | CTC symbol-to-ID table; blank is 250 |
| `decode_with_confidence.py` | 6,313 | `3b8ea5f037d81beff1a3d8d7ce1dfb95c0fd69d4` | CTC decoding with per-symbol confidence |
| `quran_per_eval.py` | 6,668 | `5b587f1592b8048fde94fa68c3ba687e9234f930` | PER evaluation and documented full streaming loop |
| `quran_wer_retrieval.py` | 4,758 | `657fbb16478a0f224396d31d15a6a9b10f5cbae3` | Quran-lexicon retrieval WER evaluation |
| `export_quran_streaming_onnx.py` | 4,376 | `b905ac3ce60b452c9069f201809c098f0c3c7f47` | Exact ONNX exporter |
| `packing_front.json` / `packing_back.json` | 2,782 / 2,896 | `ed3f09eda71767a08657054e866ec7328ac4f3fd` / `f558cdf439b4fd956e5b49a4b2541ecec0b00845` | CoreML split-function state packing layouts; not a Zipformer/Wav2Vec vocabulary bridge |
| `zipformer_p_arabic_v3.1.onnx` | 262,977,606 | `2ee39273e5033e9aebb040c442163cc768926a3d` | Streaming CTC ONNX model |
| `zipformer_p_arabic_v3.1.pt` | 263,577,418 | `24ead65c357476d8693462a7a881b609066cf784` | PyTorch checkpoint |

- The model API marks the pinned repo `gated: manual`. Direct no-token resolution of every listed canonical map, tokenizer, evaluation and export script returned HTTP 401. The environment has neither an `HF_TOKEN` nor an `HF_HUB_TOKEN`, and no local cache contains these files/weights. The code payloads and JSON schemas therefore remain inaccessible; the exact source of those schemas is the named upstream `quran_per_eval.py`, `quran_wer_retrieval.py`, `decode_with_confidence.py`, and exporter, all themselves gated.
- An apparent `Muno459/zipformer_p-quran` mirror exposes the same Git blob IDs for `ordered_quran_phonemes.json`, `quran_text2phoneme.json`, `phoneme_units.json`, and the evaluation/export scripts. The HF API marks that model `gated: manual`; direct no-token reads of its four data/token files returned HTTP 401. Its `tokens.txt` is a **different** file (2,597 bytes, Git blob `7463d224e23a1e7f4a231f932ce3f87129ba1753`, versus the pinned model's 2,346-byte table), so it cannot supply the pinned model's CTC ID mapping. Its public README is corroborating documentation only, not a weight/token substitute. A separate ungated `digisolutionsapps/zikrhub-recitation-zipformer` is a different model lineage, lists only a single INT8 ONNX file and its own `tokens.txt`, and does not provide the needed canonical JSON or scripts. It is not an authoritative substitute and was not used.
- This means the blocker is **authorized content/runtime access**, not a missing filename: the existing artifacts are expressly intended to provide the canonical targets. In principle `ordered_quran_phonemes.json` can supply per-ayah phoneme sequences; `phoneme_units.json` and the offset-aware `tokens.txt` are needed to align those units with CTC IDs. But this runtime cannot verify JSON key/array structure, special-case representation, normalization, sequence indexing, hashes, or target IDs, nor run real inference without access to the pinned model. No guessed parser, `phonemes.txt`, or token-ID bridge is implemented. The strict `ZipformerReferenceTargetProvider` remains the correct seam until authorized files can be schema-checked.

### Verified acoustic/model interface

- The pinned card specifies 16 kHz mono audio converted to 80-dimensional Kaldi fbank features with the stated Povey window; it explicitly warns against substituting mel-spectrograms. The streaming ONNX path advances 48 input frames (0.48 s) and returns 12 CTC frames per full chunk, with 251 values per frame. Thus emitted frames are spaced at 40 ms. The card calls this a standard sherpa-onnx cache-aware streaming Zipformer2 CTC model and points to the gated `quran_per_eval.py` for complete state input names/shapes and feature options; without the graph/script we cannot independently verify all ONNX state dtypes/names.
- The card's separate CoreML interface table gives split `front` / `back` calls: `front` consumes `x (1,61,80)`, `processed_lens (1,) int32`, and `sg_0..sg_12`; it returns a `(24,1,384)` crossing tensor plus updated length/state. `back` consumes that tensor, the *pre-update* processed length, and `sg_0..sg_11`; it returns `log_probs (1,12,251)` plus new state. `packing_front/back.json` describe this CoreML blob layout, not the ordinary ONNX input tensors.
- The card identifies the `.pt` artifact as averaged PyTorch weights. It does not publish an exact PT call signature in the accessible content. The exporter/evaluator source that instantiates the network and loads the checkpoint is present but gated, so exact PT checkpoint keys and streaming state signature are unverified here.
- Vocabulary conclusion: Zipformer and Wav2Vec2 are **independent token spaces**. The pinned Zipformer CTC has 251 logits with blank 250 and a phoneme tokenizer; the pinned Wav2Vec2 CTC has 51 logits, pad/blank 0, and its own Transformers tokenizer. Canonical Arabic text must be tokenized separately by each model's authoritative resources. There is no upstream requirement to create a 251↔51 integer-ID mapping, and no such bridge was made.

## Bundled Quran text

- The repository contains Hafs and Warsh text JSON resources. The Warsh resource has six empty placeholders; the data loader now skips those entries while preserving ayah numbers/global IDs. Requests for such absent references fail closed rather than receiving blank text.
- Reference results compute a SHA-256 for the source JSON and for each resolved text, so the selected riwaya and actual reference content are auditable.
