"""Dense embedding model selection. MiniLM stays the default baseline."""

from __future__ import annotations

from pathlib import Path

from src.config import RETRIEVAL, UPGRADE, UpgradeConfig

MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_DIMENSION = 384
DEFAULT_BIOMEDICAL_MODEL = "pritamdeka/S-PubMedBert-MS-MARCO"

# Practical sentence-transformer biomedical/scientific alternatives if the
# primary candidate is unavailable. MiniLM is never silently replaced.
BIOMEDICAL_CANDIDATES = (
    "pritamdeka/S-PubMedBert-MS-MARCO",
    "NeuML/pubmedbert-base-embeddings",
    "sentence-transformers/allenai-specter",
)

MINILM_RATIONALE = (
    "Lightweight local baseline (~80MB, 384-d) that runs on CPU without an API key. "
    "It is the comparison baseline, not a claimed biomedical SOTA encoder."
)


def resolve_embedding_model(config: UpgradeConfig | None = None) -> str:
    """Return MiniLM unless biomedical embeddings are explicitly enabled."""

    cfg = config or UPGRADE
    if cfg.biomedical_embeddings_enabled:
        return cfg.biomedical_embedding_model or DEFAULT_BIOMEDICAL_MODEL
    return RETRIEVAL.dense_model_name or MINILM_MODEL


def embedding_cache_dir(config: UpgradeConfig | None = None) -> str | None:
    cfg = config or UPGRADE
    if not cfg.embedding_cache_dir:
        return None
    return cfg.embedding_cache_dir


def huggingface_model_cached(model_name: str) -> bool:
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    slug = "models--" + model_name.replace("/", "--")
    return (cache_root / slug).exists()


def embedding_memory_bytes(n_passages: int, dimension: int, *, dtype_bytes: int = 4) -> int:
    return int(n_passages) * int(dimension) * int(dtype_bytes)
