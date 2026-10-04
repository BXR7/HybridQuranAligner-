# Issue #120 external sources and pinned evidence

Research captured on 2026-10-04 for implementation of the hybrid Quran aligner.

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
- Its [revision-pinned model card](https://huggingface.co/Quran-Lab/zipformer_p-arabic-v3/blob/506422c82a81c86e7ae74a5a2ab4641724bcd3b3/README.md) documents a 251-symbol Hafs phoneme CTC output and lists `tokens.txt`, `phoneme_units.json`, `ordered_quran_phonemes.json` (canonical phonemisation of 6,236 ayat), and `quran_text2phoneme.json` (text-to-phoneme lookup used in evaluation).
- The card declares the Quran-Lab No-Profit License 1.2 and a gated download. Artifact metadata was readable, but attempting to resolve the reference JSON payload returned HTTP 401 in this sandbox. There is no `HF_TOKEN`/`HF_HUB_TOKEN` in the current environment and no local model cache. Consequently, this implementation does not invent a parser, phoneme spelling, or token-ID bridge for those gated files. The typed `ZipformerReferenceTargetProvider` contract records the expected pinned source boundary; Zipformer emissions are explicitly marked unaligned until an authorized, schema-verified provider is available.

## Bundled Quran text

- The repository contains Hafs and Warsh text JSON resources. The Warsh resource has six empty placeholders; the data loader now skips those entries while preserving ayah numbers/global IDs. Requests for such absent references fail closed rather than receiving blank text.
- Reference results compute a SHA-256 for the source JSON and for each resolved text, so the selected riwaya and actual reference content are auditable.
