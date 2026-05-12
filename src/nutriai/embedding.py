"""Sentence-transformer embeddings (same model for indexing and retrieval)."""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.nutriai.config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def encode_texts(
    model: SentenceTransformer,
    texts: list[str],
    *,
    batch_size: int = 64,
    show_progress_bar: bool = False,
) -> list[list[float]]:
    if not texts:
        return []
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return emb.tolist()
