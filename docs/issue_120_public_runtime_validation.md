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

## Verification performed in this sandbox

- Public model and token URLs: HTTP 200, no authentication token used.
- Real audio: Mishari Alafasy, Al-Fatiha 1:1, converted to 16 kHz mono WAV.
- Public sherpa smoke: **passed**; 14 timestamped Zipformer emissions were returned.
- Opt-in integration test: **1 passed**.
- Unit suite: **448 passed, 2 skipped**.
- Ruff check, format check, and `git diff --check`: **passed**.

## Remaining limitation

The full `HybridQuranAligner.from_pretrained()` E2E was attempted on the real audio and
fails at the independent recitation-segmenter stage because this environment does not
have the pinned `torch`/`recitations-segmenter` runtime installed. Therefore this work
must not be classified as complete tripartite E2E verification. Wav2Vec2 forced alignment
and canonical Zipformer target DP fusion remain unverified here as part of the complete
pipeline. The public Zipformer model itself is **not** blocked by gated artifact access.
