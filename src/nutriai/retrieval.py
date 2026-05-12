"""Qdrant retrieval: hybrid dense (MiniLM) + sparse (BM25) with RRF fusion."""

from __future__ import annotations

import os
from collections import Counter
from typing import Any, Literal, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
)

from nutriai.config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_VECTOR_SIZE,
    HYBRID_PREFETCH_LIMIT,
    QDRANT_COLLECTION_NAME,
    QDRANT_SPARSE_VECTOR_NAME,
    QDRANT_VECTOR_NAME,
    SPARSE_EMBEDDING_MODEL_NAME,
)
from nutriai.embedding import encode_texts, get_embedding_model
from nutriai.qdrant_indexes import ensure_filter_payload_indexes
from nutriai.sparse_embedding import texts_to_sparse_vectors


def _client_from_env(
    url: Optional[str] = None,
    api_key: Optional[str] = None,
    host: str = "localhost",
    port: int = 6333,
) -> QdrantClient:
    url = url or os.getenv("QDRANT_URL")
    key = api_key if api_key is not None else os.getenv("QDRANT_API_KEY")
    if url:
        return QdrantClient(url=url, api_key=key or None)
    return QdrantClient(host=host, port=port, api_key=key or None)


class CulinaryTools:
    """Hybrid search (RRF) over child chunks; dense/sparse-only modes available."""

    def __init__(
        self,
        *,
        client: Optional[QdrantClient] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = QDRANT_COLLECTION_NAME,
        dense_vector_name: str = QDRANT_VECTOR_NAME,
        sparse_vector_name: str = QDRANT_SPARSE_VECTOR_NAME,
        prefetch_limit: int = HYBRID_PREFETCH_LIMIT,
    ) -> None:
        self.client = client or _client_from_env(url=url, api_key=api_key, host=host, port=port)
        self.collection_name = collection_name
        self.dense_vector_name = dense_vector_name
        self.sparse_vector_name = sparse_vector_name
        self.prefetch_limit = prefetch_limit
        ensure_filter_payload_indexes(self.client, self.collection_name)

    def _encode_dense_query(self, text: str) -> list[float]:
        return encode_texts(get_embedding_model(), [text], batch_size=32)[0]

    def _encode_sparse_query(self, text: str):
        return texts_to_sparse_vectors([text], batch_size=1)[0]

    def _optional_kind_filter(self, kind: Optional[str]) -> Optional[Filter]:
        if not kind:
            return None
        return Filter(must=[FieldCondition(key="kind", match=MatchValue(value=kind))])

    def _query_points_hybrid(
        self,
        text: str,
        *,
        limit: int,
        kind: Optional[str] = None,
    ) -> list[Any]:
        dense = self._encode_dense_query(text)
        sparse = self._encode_sparse_query(text)
        flt = self._optional_kind_filter(kind)
        resp = self.client.query_points(
            collection_name=self.collection_name,
            prefetch=[
                Prefetch(
                    query=sparse,
                    using=self.sparse_vector_name,
                    limit=self.prefetch_limit,
                ),
                Prefetch(
                    query=dense,
                    using=self.dense_vector_name,
                    limit=self.prefetch_limit,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            query_filter=flt,
            limit=limit,
            with_payload=True,
        )
        return list(resp.points)

    def _query_points_dense_only(
        self,
        text: str,
        *,
        limit: int,
        kind: Optional[str] = None,
    ) -> list[Any]:
        dense = self._encode_dense_query(text)
        flt = self._optional_kind_filter(kind)
        resp = self.client.query_points(
            collection_name=self.collection_name,
            query=dense,
            using=self.dense_vector_name,
            query_filter=flt,
            limit=limit,
            with_payload=True,
        )
        return list(resp.points)

    def _query_points_sparse_only(
        self,
        text: str,
        *,
        limit: int,
        kind: Optional[str] = None,
    ) -> list[Any]:
        sparse = self._encode_sparse_query(text)
        flt = self._optional_kind_filter(kind)
        resp = self.client.query_points(
            collection_name=self.collection_name,
            query=sparse,
            using=self.sparse_vector_name,
            query_filter=flt,
            limit=limit,
            with_payload=True,
        )
        return list(resp.points)

    def search_by_ingredients(
        self,
        ingredients: list[str],
        *,
        limit: int = 3,
        hits_per_ingredient: int = 8,
        mode: Literal["hybrid", "dense", "sparse"] = "hybrid",
    ):
        """
        For each ingredient, search ``kind=ingredient`` children, aggregate by
        ``parent_id``. Default **hybrid** (BM25 + dense, RRF) balances exact
        tokens (e.g. ``steak``) and phrasing.
        """
        parent_hits: Counter[str] = Counter()
        for ing in ingredients:
            if not (ing or "").strip():
                continue
            q = ing.strip()
            if mode == "hybrid":
                hits = self._query_points_hybrid(q, limit=hits_per_ingredient, kind="ingredient")
            elif mode == "dense":
                hits = self._query_points_dense_only(q, limit=hits_per_ingredient, kind="ingredient")
            else:
                hits = self._query_points_sparse_only(q, limit=hits_per_ingredient, kind="ingredient")
            for hit in hits:
                pid = hit.payload.get("parent_id") if hit.payload else None
                if pid:
                    parent_hits[pid] += 1
        return parent_hits.most_common(limit)

    def search_children(
        self,
        query: str,
        *,
        limit: int = 10,
        kind: Optional[str] = None,
        mode: Literal["hybrid", "dense", "sparse"] = "hybrid",
    ) -> list[Any]:
        """Search child (or filtered) chunks; default is hybrid RRF."""
        q = query.strip()
        if mode == "hybrid":
            return self._query_points_hybrid(q, limit=limit, kind=kind)
        if mode == "dense":
            return self._query_points_dense_only(q, limit=limit, kind=kind)
        return self._query_points_sparse_only(q, limit=limit, kind=kind)

    def get_full_recipe(self, parent_id: str) -> str:
        """All parent shards for this recipe id, ordered by ``chunk_id``."""
        flt = Filter(
            must=[
                FieldCondition(key="parent_id", match=MatchValue(value=parent_id)),
                FieldCondition(key="kind", match=MatchValue(value="parent")),
            ]
        )
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=flt,
            limit=256,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return "Recipe not found."
        ordered = sorted(
            points,
            key=lambda p: (p.payload or {}).get("chunk_id", ""),
        )
        texts = [(p.payload or {}).get("text") for p in ordered]
        texts = [t for t in texts if t]
        return "\n\n---\n\n".join(texts) if texts else "Recipe not found."


def describe_embedding_setup() -> dict[str, str | int]:
    return {
        "dense_model": EMBEDDING_MODEL_NAME,
        "sparse_model": SPARSE_EMBEDDING_MODEL_NAME,
        "dense_vector": QDRANT_VECTOR_NAME,
        "sparse_vector": QDRANT_SPARSE_VECTOR_NAME,
        "collection": QDRANT_COLLECTION_NAME,
        "dense_dim": EMBEDDING_VECTOR_SIZE,
        "retrieval": "hybrid_rrf",
    }
