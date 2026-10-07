"""On-disk embedding cache so 40k-passage indexes are not rebuilt every run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def cache_file(cache_dir: str | Path, model_name: str, passage_ids: list[Any]) -> Path:
    digest = hashlib.sha1()
    digest.update(model_name.encode("utf-8"))
    digest.update(str(len(passage_ids)).encode("utf-8"))
    if passage_ids:
        digest.update(str(passage_ids[0]).encode("utf-8"))
        digest.update(str(passage_ids[-1]).encode("utf-8"))
    return Path(cache_dir) / f"{digest.hexdigest()}.npz"


def load_cached_embeddings(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    payload = np.load(path, allow_pickle=False)
    return np.asarray(payload["embeddings"], dtype=float)


def save_cached_embeddings(path: Path, embeddings: np.ndarray, *, model_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, embeddings=np.asarray(embeddings, dtype=float))
    meta = path.with_suffix(".json")
    meta.write_text(json.dumps({"model_name": model_name, "shape": list(embeddings.shape)}), encoding="utf-8")
