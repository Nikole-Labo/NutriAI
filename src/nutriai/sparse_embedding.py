"""FastEmbed sparse BM25 vectors (Qdrant ``Qdrant/bm25`` + IDF modifier on collection)."""

from __future__ import annotations

from functools import lru_cache

from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector

from nutriai.config import SPARSE_EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_sparse_embedding_model() -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=SPARSE_EMBEDDING_MODEL_NAME)


def texts_to_sparse_vectors(
    texts: list[str],
    *,
    batch_size: int = 32,
) -> list[SparseVector]:
    """Encode texts to Qdrant ``SparseVector`` instances (BM25-style)."""
    if not texts:
        return []
    model = get_sparse_embedding_model()
    out: list[SparseVector] = []
    for emb in model.embed(texts, batch_size=batch_size):
        idx = emb.indices
        val = emb.values
        if hasattr(idx, "tolist"):
            idx = idx.tolist()
        if hasattr(val, "tolist"):
            val = val.tolist()
        idx_i = [int(i) for i in idx]
        val_f = [float(v) for v in val]
        if not idx_i:
            # Qdrant rejects empty sparse vectors; should not happen for real recipe text.
            idx_i, val_f = [0], [0.0]
        out.append(SparseVector(indices=idx_i, values=val_f))
    return out
