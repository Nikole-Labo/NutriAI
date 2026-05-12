"""Create the Qdrant collection and upsert parents + children from JSONL.

Hybrid vectors per point:

* **dense** — ``sentence-transformers/all-MiniLM-L6-v2`` (384-d, cosine).
* **sparse** — FastEmbed ``Qdrant/bm25`` (lexical / ingredient-style matches).

Requires:
  - ``data/processed/parents.jsonl`` and ``children.jsonl`` (run ``run_chunking.py`` first)
  - Qdrant reachable at ``QDRANT_URL`` or localhost:6333
  - ``pip install -r requirements.txt``

**Upgrading from an older dense-only collection:** run with ``--recreate`` so the
collection is rebuilt with a sparse vector index (``Modifier.IDF``).

Examples (PowerShell)::

    cd C:\\Users\\lazab\\NutriAI
    .\\.venv\\Scripts\\Activate.ps1
    python scripts/index_qdrant.py --recreate

    # Only refresh children
    python scripts/index_qdrant.py --children-only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Modifier,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")

from src.nutriai.config import (  # noqa: E402
    EMBEDDING_VECTOR_SIZE,
    PROCESSED_DIR,
    QDRANT_COLLECTION_NAME,
    QDRANT_SPARSE_VECTOR_NAME,
    QDRANT_VECTOR_NAME,
)
from src.nutriai.embedding import encode_texts, get_embedding_model  # noqa: E402
from src.nutriai.qdrant_ids import child_point_uuid, parent_point_uuid  # noqa: E402
from src.nutriai.qdrant_indexes import ensure_filter_payload_indexes  # noqa: E402
from src.nutriai.sparse_embedding import texts_to_sparse_vectors  # noqa: E402


def _client() -> QdrantClient:
    import os

    url = os.getenv("QDRANT_URL")
    key = os.getenv("QDRANT_API_KEY")
    if url:
        return QdrantClient(url=url, api_key=key or None)
    return QdrantClient(host=os.getenv("QDRANT_HOST", "localhost"), port=int(os.getenv("QDRANT_PORT", "6333")), api_key=key or None)


def _ensure_collection(client: QdrantClient, *, recreate: bool) -> None:
    names = {c.name for c in client.get_collections().collections}
    if QDRANT_COLLECTION_NAME in names and recreate:
        client.delete_collection(QDRANT_COLLECTION_NAME)
        names.discard(QDRANT_COLLECTION_NAME)
    if QDRANT_COLLECTION_NAME not in names:
        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config={
                QDRANT_VECTOR_NAME: VectorParams(
                    size=EMBEDDING_VECTOR_SIZE,
                    distance=Distance.COSINE,
                )
            },
            sparse_vectors_config={
                QDRANT_SPARSE_VECTOR_NAME: SparseVectorParams(modifier=Modifier.IDF),
            },
        )


def _payload_drop_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _upsert_batch(
    client: QdrantClient,
    dense_model,
    ids: list,
    texts: list[str],
    payloads: list[dict],
    *,
    batch_size_encode: int,
) -> None:
    if not ids:
        return
    dense_vectors = encode_texts(
        dense_model, texts, batch_size=batch_size_encode, show_progress_bar=False
    )
    sparse_vectors = texts_to_sparse_vectors(texts, batch_size=batch_size_encode)
    points = [
        PointStruct(
            id=pid,
            vector={
                QDRANT_VECTOR_NAME: dvec,
                QDRANT_SPARSE_VECTOR_NAME: svec,
            },
            payload=_payload_drop_none(pl),
        )
        for pid, dvec, svec, pl in zip(ids, dense_vectors, sparse_vectors, payloads)
    ]
    client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)


def index_parents(
    client: QdrantClient,
    dense_model,
    path: Path,
    *,
    batch_lines: int,
    batch_size_encode: int,
) -> int:
    if not path.is_file():
        raise SystemExit(f"Missing {path} — run scripts/run_chunking.py first.")
    buf_ids: list = []
    buf_texts: list[str] = []
    buf_payloads: list[dict] = []
    total = 0
    for row in _iter_jsonl(path):
        cid = row["chunk_id"]
        buf_ids.append(parent_point_uuid(cid))
        buf_texts.append(row["text"])
        buf_payloads.append(
            {
                "chunk_id": cid,
                "parent_id": row["parent_id"],
                "kind": row["kind"],
                "title": row.get("title"),
                "text": row["text"],
                "ingredients": row.get("ingredients"),
                "source": row.get("source"),
                "url": row.get("url"),
                "n_tokens": row.get("n_tokens"),
            }
        )
        if len(buf_ids) >= batch_lines:
            _upsert_batch(
                client, dense_model, buf_ids, buf_texts, buf_payloads, batch_size_encode=batch_size_encode
            )
            total += len(buf_ids)
            buf_ids.clear()
            buf_texts.clear()
            buf_payloads.clear()
    _upsert_batch(client, dense_model, buf_ids, buf_texts, buf_payloads, batch_size_encode=batch_size_encode)
    total += len(buf_ids)
    return total


def index_children(
    client: QdrantClient,
    dense_model,
    path: Path,
    *,
    batch_lines: int,
    batch_size_encode: int,
) -> int:
    if not path.is_file():
        raise SystemExit(f"Missing {path} — run scripts/run_chunking.py first.")
    buf_ids: list = []
    buf_texts: list[str] = []
    buf_payloads: list[dict] = []
    total = 0
    for row in _iter_jsonl(path):
        cid = row["chunk_id"]
        buf_ids.append(child_point_uuid(cid))
        buf_texts.append(row["text"])
        buf_payloads.append(
            {
                "chunk_id": cid,
                "parent_id": row["parent_id"],
                "kind": row["kind"],
                "text": row["text"],
                "ingredient_name": row.get("ingredient_name"),
                "nutrient_name": row.get("nutrient_name"),
                "nutrient_amount": row.get("nutrient_amount"),
                "nutrient_unit": row.get("nutrient_unit"),
                "n_tokens": row.get("n_tokens"),
            }
        )
        if len(buf_ids) >= batch_lines:
            _upsert_batch(
                client, dense_model, buf_ids, buf_texts, buf_payloads, batch_size_encode=batch_size_encode
            )
            total += len(buf_ids)
            buf_ids.clear()
            buf_texts.clear()
            buf_payloads.clear()
    _upsert_batch(client, dense_model, buf_ids, buf_texts, buf_payloads, batch_size_encode=batch_size_encode)
    total += len(buf_ids)
    return total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Index NutriAI JSONL into Qdrant (hybrid dense + BM25 sparse).")
    p.add_argument("--out", type=Path, default=PROCESSED_DIR, help="Directory with parents.jsonl / children.jsonl.")
    p.add_argument(
        "--recreate",
        action="store_true",
        help="Delete existing collection if present, then create fresh (required when upgrading schema).",
    )
    p.add_argument("--parents-only", action="store_true", help="Only upsert parent points.")
    p.add_argument("--children-only", action="store_true", help="Only upsert child points.")
    p.add_argument("--batch-lines", type=int, default=128, help="JSONL lines per upsert batch.")
    p.add_argument("--encode-batch", type=int, default=64, help="Batch size for dense + sparse encoders.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.parents_only and args.children_only:
        raise SystemExit("Use at most one of --parents-only / --children-only.")

    parents_path = args.out / "parents.jsonl"
    children_path = args.out / "children.jsonl"

    client = _client()
    _ensure_collection(client, recreate=args.recreate)
    ensure_filter_payload_indexes(client)

    print("Loading dense embedding model (first run may download weights)…")
    dense_model = get_embedding_model()
    print("Warming sparse BM25 model (FastEmbed Qdrant/bm25)…")
    texts_to_sparse_vectors(["warmup"], batch_size=1)

    n_p = n_c = 0
    if not args.children_only:
        print(f"Indexing parents from {parents_path} …")
        n_p = index_parents(
            client, dense_model, parents_path, batch_lines=args.batch_lines, batch_size_encode=args.encode_batch
        )
        print(f"  upserted {n_p} parent points")
    if not args.parents_only:
        print(f"Indexing children from {children_path} …")
        n_c = index_children(
            client, dense_model, children_path, batch_lines=args.batch_lines, batch_size_encode=args.encode_batch
        )
        print(f"  upserted {n_c} child points")

    info = client.get_collection(QDRANT_COLLECTION_NAME)
    n_pts = getattr(info, "points_count", None)
    print(f"Collection {QDRANT_COLLECTION_NAME!r} points_count={n_pts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
