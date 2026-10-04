"""Cache-first management of immutable Hugging Face model artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from munajjam.exceptions import ModelUnavailableError


@dataclass(frozen=True, slots=True)
class ModelSpec:
    repository: str
    revision: str
    files: tuple[str, ...]
    hashes: dict[str, str] | None = None


class ModelManager:
    """Resolve pinned artifacts without ever logging credentials or committing weights."""

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        self.cache_dir = Path(
            cache_dir or os.environ.get("MUNAJJAM_MODEL_CACHE", "~/.cache/munajjam/models")
        ).expanduser()

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def verify_directory(self, directory: str | Path, spec: ModelSpec) -> dict[str, Any]:
        root = Path(directory)
        missing = [name for name in spec.files if not (root / name).is_file()]
        if missing:
            raise ModelUnavailableError(f"pinned model is missing required files: {missing}")
        hashes = {name: self._sha256(root / name) for name in spec.files}
        for name, expected in (spec.hashes or {}).items():
            if hashes.get(name) != expected.lower():
                raise ModelUnavailableError(f"hash mismatch for pinned artifact {name}")
        manifest = {
            "repository": spec.repository,
            "revision": spec.revision,
            "files": list(spec.files),
            "sha256": hashes,
        }
        return manifest

    def resolve(
        self, spec: ModelSpec, *, local_dir: str | Path | None = None, allow_download: bool = True
    ) -> Path:
        target = (
            Path(local_dir)
            if local_dir
            else self.cache_dir / spec.repository.replace("/", "--") / spec.revision
        )
        try:
            manifest = self.verify_directory(target, spec)
        except ModelUnavailableError:
            if not allow_download:
                raise
            try:
                from huggingface_hub import snapshot_download

                downloaded = Path(
                    snapshot_download(
                        repo_id=spec.repository,
                        revision=spec.revision,
                        allow_patterns=list(spec.files),
                        local_dir=str(target),
                        local_dir_use_symlinks=False,
                        token=os.environ.get("HF_TOKEN"),
                    )
                )
                manifest = self.verify_directory(downloaded, spec)
                target = downloaded
            except Exception as exc:
                raise ModelUnavailableError(
                    f"unable to download or verify pinned model {spec.repository}@{spec.revision}"
                ) from exc
        manifest_path = target / "manifest.json"
        fd, tmp = tempfile.mkstemp(prefix="manifest.", suffix=".json", dir=target)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, sort_keys=True, indent=2)
                handle.write("\n")
            os.replace(tmp, manifest_path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        return target
