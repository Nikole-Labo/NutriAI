"""Centralised configuration for the chunking pipeline.

Every magic number / token budget lives here so the rest of the codebase
stays clean and we have one place to tune the pipeline.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# --- Chunking knobs --------------------------------------------------------
# SmolLM-1.7B-Instruct has a 2048-token context. We reserve ~1024 tokens for
# the system prompt + user question + generation, leaving ~1024 tokens for
# retrieved context. A typical retrieval pulls top-k=4 parents, so the
# per-parent ceiling is ~256 tokens. That is the value the recursive
# fallback splitter enforces when a parent recipe is too long.
PARENT_MAX_TOKENS: int = 256
PARENT_CHUNK_OVERLAP: int = 32

# Children are short by design (one ingredient line, one nutrient), so the
# fallback should rarely fire. We keep them tight to avoid noise in the
# child index.
CHILD_MAX_TOKENS: int = 64
CHILD_CHUNK_OVERLAP: int = 0

# Tokenizer used to *count* tokens during fallback splitting. We use the
# tiktoken cl100k_base encoder as a cheap, language-agnostic proxy that
# correlates well with SmolLM's tokenizer for English recipe text.
TOKENIZER_ENCODING: str = "cl100k_base"

# Markdown headers we split parent recipes on. Order matters: we split on
# the deepest level first (## Ingredients) so a recipe becomes:
#   - Title section (level 1)
#   - Ingredients section (level 2)
#   - Instructions section (level 2)
#   - Nutrition section (level 2)
MARKDOWN_HEADERS_TO_SPLIT_ON: list[tuple[str, str]] = [
    ("#", "title"),
    ("##", "section"),
]

# --- Qdrant / hybrid embeddings (dense + sparse BM25) ----------------------
EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_VECTOR_SIZE: int = 384
SPARSE_EMBEDDING_MODEL_NAME: str = "Qdrant/bm25"
QDRANT_COLLECTION_NAME: str = "recipe_chunks"
QDRANT_VECTOR_NAME: str = "dense"
QDRANT_SPARSE_VECTOR_NAME: str = "sparse"
# Candidate pool per branch before RRF fusion (retrieval).
HYBRID_PREFETCH_LIMIT: int = 48

# --- Cross-encoder rerank + macro fit ------------------------------------
RERANK_CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_PASSAGE_MAX_CHARS: int = 1400
# Macro term dominates so recipes near remaining kcal / macro targets rank first;
# cross-encoder breaks ties using the natural-language query + recipe text.
RERANK_MACRO_WEIGHT: float = 0.72
RERANK_CE_WEIGHT: float = 0.28
