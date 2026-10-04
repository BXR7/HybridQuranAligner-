from __future__ import annotations

import os
from pathlib import Path

import pytest
from munajjam.hybrid_aligner import (
    AudioBuffer,
    BreathGroup,
    ModelManager,
    ModelSpec,
    ZipformerOnnxBackend,
    load_audio_file,
)
from munajjam.hybrid_aligner.zipformer_backend import (
    ZIPFORMER_MODEL,
    ZIPFORMER_REPOSITORY,
    ZIPFORMER_REVISION,
    ZIPFORMER_TOKEN_SHA256,
)


@pytest.mark.real_model
@pytest.mark.integration
@pytest.mark.slow
def test_zipformer_real_backend_on_real_audio():
    if os.environ.get("RUN_REAL_MODEL") != "1":
        pytest.skip("set RUN_REAL_MODEL=1 to enable real-model tests")
    artifact_dir = os.environ.get("ZIPFORMER_ARTIFACT_DIR")
    audio_path = os.environ.get(
        "QURAN_AUDIO_PATH", "/kaggle/working/001001_alafasy.mp3"
    )
    if not os.environ.get("HF_TOKEN") and not artifact_dir:
        pytest.skip("HF_TOKEN or ZIPFORMER_ARTIFACT_DIR is required")
    if not Path(audio_path).is_file():
        pytest.skip(f"real Quran audio not found: {audio_path}")
    audio: AudioBuffer = load_audio_file(audio_path)
    groups = [(0.0, min(audio.duration, 6.0))]
    if artifact_dir is None:
        artifact_dir = str(
            ModelManager().resolve(
                ModelSpec(
                    repository=ZIPFORMER_REPOSITORY,
                    revision=ZIPFORMER_REVISION,
                    files=("config.json", "tokens.txt", ZIPFORMER_MODEL),
                    hashes={"tokens.txt": ZIPFORMER_TOKEN_SHA256},
                )
            )
        )
    backend = ZipformerOnnxBackend(artifact_dir)
    emissions = backend(audio, BreathGroup(*groups[0]))
    assert emissions
    assert all(0 <= item.token_id < 251 for item in emissions)
    assert all(item.end_frame > item.start_frame for item in emissions)
    backend.close()
