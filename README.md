# Munajjam

**A Python library and API Server to synchronize Quran ayat with audio recitations.**

Munajjam uses AI-powered speech recognition to automatically generate precise timestamps for each ayah in a Quran audio recording.

## API Server & Docker (Recommended)

You can run Munajjam as a standalone API server with asynchronous processing and GPU support.

### Running with Docker

The easiest way to run the API server is using Docker Compose:

```bash
git clone https://github.com/Itqan-community/munajjam.git
cd munajjam

# To run with GPU support (default)
docker compose up --build

# To run with CPU only
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up --build
```

### API Endpoints

Once the server is running (default: http://localhost:8000), you can use the following endpoints:

- **`POST /align/{surah_number}`**: Upload an audio file for a specific surah. Returns a `job_id`.
  - Form Data: `file` (audio file), `riwaya` (e.g., "hafs")
- **`GET /align/status/{job_id}`**: Check the status of the alignment job and get the results when ready.
- **`GET /health`**: Health check.

## Library Installation

If you want to use Munajjam as a Python library:

Clone the repository:

```bash
git clone https://github.com/Itqan-community/munajjam.git
cd munajjam/munajjam
```

Install the package:

```bash
pip install .
```

For faster transcription with [faster-whisper](https://github.com/SYSTRAN/faster-whisper):

```bash
pip install ".[faster-whisper]"
```

For development (editable install):

```bash
pip install -e ".[dev]"
```

## Quick Start (Library)

### 1. Download a sample recitation

Download a sample audio file (Surah Al-Fatiha):

```bash
curl -L -o 001.mp3 "https://pub-9ee413c8af4041c6bd5223d08f5d0f0f.r2.dev/media/uploads/assets/11/recitations/001.mp3"
```

> **Note:** Audio files should be named by surah number (e.g., `001.mp3`, `002.mp3`).
> Browse more recitations at [cms.itqan.dev](https://cms.itqan.dev)

### 2. Run the alignment

```python
from munajjam.core import align
from munajjam.data import load_surah_ayahs
from munajjam.transcription import WhisperTranscriber

# Transcribe audio
with WhisperTranscriber() as transcriber:
    segments = transcriber.transcribe("001.mp3")

# Align to ayahs (uses auto strategy by default; override with "greedy", "dp", or "hybrid")
ayahs = load_surah_ayahs(1)
results = align("001.mp3", segments, ayahs)

# Get timestamps
for result in results:
    print(
        f"Ayah {result.ayah.ayah_number}: "
        f"{result.start_time:.2f}s - {result.end_time:.2f}s"
    )
```

### 3. Output

```
Ayah 1: 5.62s - 9.57s
Ayah 2: 10.51s - 14.72s
Ayah 3: 15.45s - 18.53s
Ayah 4: 19.21s - 22.54s
Ayah 5: 23.27s - 28.19s
Ayah 6: 29.00s - 33.07s
Ayah 7: 33.98s - 46.44s
```

## Features

- **API Server** - Async FastAPI server for handling concurrent alignment jobs
- **Whisper Transcription** - Uses faster-whisper as default backend with Quran-tuned models
- **Four Alignment Strategies** - Auto, Hybrid, DP, and Greedy
- **Arabic Text Normalization** - Handles diacritics, hamzas, and character variations
- **Automatic Drift Correction** - Multi-pass zone realignment for long recordings
- **Quality Metrics** - Confidence scores for each aligned ayah
- **Phonetic Similarity** - Arabic ASR confusion-aware similarity scoring
- **Word-level Precision** - Uses per-word timestamps (when available) to improve drift recovery

## Alignment Strategies

The default `auto` strategy works best for most cases. You can override it:

```python
from munajjam.core import Aligner

# Auto (recommended) - picks the best strategy, full pipeline by default
aligner = Aligner("001.mp3")

# Hybrid - DP with greedy fallback (legacy)
aligner = Aligner(
    "001.mp3",  # Audio file path (required)
    strategy="auto",  # "greedy", "dp", "hybrid", or "auto" (default)
    quality_threshold=0.85,  # Similarity threshold for high-quality alignment
    fix_drift=True,  # Run zone realignment for long surahs
    fix_overlaps=True,  # Fix overlapping ayah timings
    min_gap=0.3,  # Minimum gap between consecutive ayahs (seconds)
    energy_snap=True,  # Snap boundaries to energy minima (default True)
)

results = aligner.align(segments, ayahs)
```

## Examples

See the [examples](./examples) directory for more usage patterns:

- `01_basic_usage.py` - Simple transcription and alignment
- `02_comparing_strategies.py` - Compare alignment strategies
- `03_advanced_configuration.py` - Custom settings and options
- `04_batch_processing.py` - Process multiple files

## Requirements

- Python 3.10+
- PyTorch 2.0+
- FFmpeg (for audio processing)
- Docker & Docker Compose (Optional, for running the API server)

## Community

- [Website](https://munajjam.itqan.dev)
- [ITQAN Community](https://community.itqan.dev)

## Acknowledgments

- [Tarteel AI](https://tarteel.ai) for the Quran-specialized Whisper model

## License

MIT License - see [LICENSE](./LICENSE) for details.

## Opt-in Tripartite Hybrid Aligner (Issue #120)

Munajjam includes an **experimental, opt-in** alignment engine that combines breath-group
segmentation, Zipformer reference evidence, and Wav2Vec2 CTC forced alignment.

This engine is **NOT** used by the default `auto` strategy or the server. It must be
explicitly imported and invoked:

```python
from munajjam.hybrid_aligner import HybridQuranAligner

aligner = HybridQuranAligner(
    # inject a logits provider that returns a [frames, vocab] array
    forced_aligner=Wav2Vec2ForcedAligner(logits_provider=my_ctc_provider),
)
result = aligner.align(audio_buffer, targets)
```

`targets` is caller-supplied and must contain exactly one `(token_ids, token_texts)` pair
for each detected breath group. The pipeline fails closed on a count mismatch; it does not
guess Quran phrase boundaries or automatically assign ayat to breaths.

### Architecture

```
audio
  ↓
breath segmentation  (signal-processing segmenter; group count is not fixed)
  ↓
Zipformer reference evidence (gated, per-breath)
  ↓
per-breath CTC       (Wav2Vec2ForcedAligner — provider-injected)
  ↓
validated final alignment
```

The upstream [Issue #120 acceptance criteria](https://github.com/Itqan-community/Munajjam/issues/120)
describe Zipformer as the **reference phoneme-alignment stage** (including canonical
reference text and Isti'adhah/Basmalah identification), followed by Wav2Vec2 microscopic
forced alignment. The issue does not prescribe an evidence-fusion algorithm or a mapping
between the 251-symbol Zipformer vocabulary and Wav2Vec2's 51-symbol vocabulary. Therefore
this implementation retains Zipformer's validated per-breath emissions as explicit
reference-phoneme evidence in result metadata, but does not claim that those emissions
are fused into final spans. The two token spaces are never mapped or interchanged. This
preserves the supported stage boundary without inventing fusion semantics; canonical
text matching and Isti'adhah/Basmalah identification remain unimplemented until their
authoritative mapping/decision contract is available.

For the pinned Zipformer model card at revision
`506422c82a81c86e7ae74a5a2ab4641724bcd3b3`, the documented streaming grid is a 61-frame
fbank input, a 48-frame (0.48 s) advance, and 12 CTC output frames per full advance. Thus
reference emission frame `i` maps to `i * 0.04` seconds relative to its breath-group start.
Final padded-window trimming and the complete per-setting fbank equivalence still require
verification against the gated pinned evaluator/exporter sources before claiming artifact-
level equivalence.

### Optional model dependencies

The hybrid aligner does **not** import any heavy ML framework at import time. Heavy
libraries are lazily imported only when a matching provider/backend is explicitly
constructed:

- **torch** — optional, required only by neural model backends.
- **transformers** — optional, required only by Transformer-based CTC backends.
- **whisperx** — optional, used by the legacy server transcription path.
- **onnxruntime** — optional, for ONNX-based model backends.
- **kaldi-native-fbank** — optional, for Zipformer feature extraction.
- **sherpa-onnx** — optional, for sherpa-onnx CTC backends.
- **faster-whisper** — optional, for faster-whisper transcription backends.

`pip install .` does **not** install these. Install them only when you need the
corresponding neural backend:

```bash
pip install torch transformers  # for CTC providers
pip install sherpa-onnx         # for sherpa-onnx backend
cd munajjam && pip install '.[zipformer]'  # ONNX Runtime + Kaldi native fbank
```

### Model access requirements

The `ZipformerNeuralAligner` validates operator-supplied evidence at construction
time and fails **closed** when evidence is missing or invalid:

- **Repository:** must be `Quran-Lab/zipformer_p-arabic-v3`.
- **Revision:** must be an immutable 40-character git SHA.
- **Approval:** must be explicitly set to `True`.
- **`tokens.txt`:** the pinned file is parsed by its explicit `<piece> <id>`
  entries, not by line position; it must define all 251 IDs and `<blank>` at ID 250.
  The pinned vocabulary SHA-256 is
  `252c10687e442aa9291973065fae19fa39bcd681c4f5612ec496a647e20b43a1`.
- **Feature extractor:** uses 16 kHz mono audio and 80-bin Kaldi fbank features;
  see the pinned model-card revision and the implementation notes above for the
  currently explicit settings and outstanding reference-script parity check.
- **License:** the Zipformer model is subject to its upstream license; verify
  compliance before use.

### Limitations

- `ZipformerNeuralAligner.from_pretrained()` loads the pinned model into the
  Hugging Face cache; an explicit `backend_factory` can instead be injected for tests
  or a separately managed artifact directory.
- Real-model smoke tests are **opt-in** and require credentials for HuggingFace Hub
  access to `Quran-Lab/zipformer_p-arabic-v3`.
- Wav2Vec2 defaults to CUDA when available and otherwise uses CPU; an explicit
  device can be supplied. Zipformer's ONNX Runtime provider list prefers CUDA and
  falls back to CPU.
- Post-roll retention is bounded by the next breath group boundary and the audio end.
- No timestamps are ever synthesized without validated acoustic evidence from the CTC
  trellis.

### Real-model smoke test instructions

Set `HF_TOKEN` in your environment and run:

```bash
export HF_TOKEN=your-huggingface-token
RUN_REAL_MODEL=1 PYTHONPATH=./munajjam pytest -m real_model \
  tests/integration/test_zipformer_real_model.py
```

The integration test also requires a real audio file (`QURAN_AUDIO_PATH`) or
`ZIPFORMER_ARTIFACT_DIR`. If those, model access credentials, or a usable audio
fixture are unavailable, pytest reports a skip rather than treating the model check
as passed.
