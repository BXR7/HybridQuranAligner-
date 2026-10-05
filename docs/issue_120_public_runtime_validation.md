# Issue #120 — Public Zipformer Runtime Validation

## Public production path

The production Zipformer path uses the ungated Hugging Face repository
[`Alimalas/munajjam-onnx-models`](https://huggingface.co/Alimalas/munajjam-onnx-models) at
revision `5dbab4db48a88f5a2a76ead282b2bc3d4b958ee0`:

- `model_zipformer/zipformer_p_arabic_v3.onnx` — 262,977,606 bytes
- `model_zipformer/tokens.txt` — SHA-256 `252c10687e442aa9291973065fae19fa39bcd681c4f5612ec496a647e20b43a1`

The backend uses `sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc` with 4 threads,
16 kHz audio, 80-dimensional features, endpoint detection disabled, and greedy decoding.
The model binary is never committed to the repository.

## Canonical character-DP fusion

The public 251-token stream is decoded from the pinned `tokens.txt`, normalized
with the production `normalize_arabic()` implementation, expanded into character
units, and aligned to the bundled Hafs/Warsh canonical reference using a
four-operation dynamic program: `MATCH`, `SUBSTITUTE`, `INSERT`, and `DELETE`.
Each evidence cell retains token ID, raw and normalized symbols, frame range,
score, DP operation/cost, and canonical source provenance. Accepted evidence is
fused into final spans by adding the
`wav2vec2-ctc+zipformer-character-dp` provenance marker.

The fail-closed defaults are normalized edit cost `<= 0.45` and canonical match
coverage `>= 0.70`; below either threshold the production path raises an
alignment error rather than setting success metadata.

## Verification performed in this sandbox

- Public model and token URLs: HTTP 200, no authentication token used.
- Real audio: Mishari Alafasy, Al-Fatiha 1:1, converted to 16 kHz mono WAV.
- Public sherpa smoke: **passed**; 24 timestamped Zipformer emissions were returned
  from the first 8 seconds of the public Al-Fatiha recording.
- Real full-surah character-DP: **accepted** on 142 emissions against the
  canonical Hafs reference for Al-Fatiha with normalized cost `0.40` and match
  coverage `0.7301`.
- Opt-in public-model integration test: **1 passed**.
- Unit suite after fusion implementation: **454 passed, 3 skipped**.
- Changed-file Ruff check, format check, and `git diff --check`: **passed**.

## Remaining limitation

The full `HybridQuranAligner.from_pretrained()` E2E was attempted on the real audio and
fails at the independent recitation-segmenter stage because this environment does not
have the pinned `torch`/`recitations-segmenter` runtime installed. Therefore the full
segmenter → Wav2Vec2 → Zipformer tripartite pipeline has not been claimed as verified
in this sandbox. The public Zipformer model and its canonical character-DP fusion are
verified independently; the remaining limitation is only the unavailable full
recitation-segmenter/Wav2Vec2 runtime combination.
