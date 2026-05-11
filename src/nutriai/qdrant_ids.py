"""Stable UUID point ids for Qdrant (ids must be int or UUID, not arbitrary strings)."""

from __future__ import annotations

import uuid

# Fixed namespace so re-indexing upserts the same point ids.
_NUTRIAI_NAMESPACE = uuid.UUID("0193b000-7000-7000-7000-000000000001")


def child_point_uuid(chunk_id: str) -> uuid.UUID:
    return uuid.uuid5(_NUTRIAI_NAMESPACE, f"child:{chunk_id}")


def parent_point_uuid(parent_chunk_id: str) -> uuid.UUID:
    return uuid.uuid5(_NUTRIAI_NAMESPACE, f"parent:{parent_chunk_id}")
