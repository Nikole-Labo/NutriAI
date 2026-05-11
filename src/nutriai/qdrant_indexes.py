"""Payload indexes required for filtered queries (Qdrant Cloud strict mode)."""

from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import PayloadSchemaType

from nutriai.config import QDRANT_COLLECTION_NAME


def ensure_filter_payload_indexes(
    client: QdrantClient,
    collection_name: str = QDRANT_COLLECTION_NAME,
) -> None:
    """Create keyword indexes on ``kind`` and ``parent_id`` if missing.

    Qdrant Cloud often enables **strict mode**, which rejects ``query_filter``
    and ``scroll_filter`` on unindexed payload keys. Our retrieval uses:

    * ``kind`` — e.g. ``ingredient`` / ``parent`` / ``nutrient``
    * ``parent_id`` — parent recipe id
    """
    info = client.get_collection(collection_name)
    existing: set[str] = set()
    if info.payload_schema:
        existing = set(info.payload_schema.keys())

    for field_name in ("kind", "parent_id"):
        if field_name in existing:
            continue
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=PayloadSchemaType.KEYWORD,
            )
        except UnexpectedResponse as e:
            blob = (e.content or b"").decode("utf-8", errors="ignore").lower()
            if "already" in blob or "exists" in blob or "duplicate" in blob:
                continue
            raise
