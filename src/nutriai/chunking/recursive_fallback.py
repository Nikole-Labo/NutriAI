"""Recursive Character Splitting fallback.

Markdown-Header splitting handles 99% of recipes cleanly, but the long-tail
of Recipe1M+ contains documents with 50+ instruction steps that still
overflow our per-parent token budget (see ``config.PARENT_MAX_TOKENS``).

This fallback is invoked **only** for those oversized parent chunks. We use
``RecursiveCharacterTextSplitter`` (LangChain) configured to:

1. Count tokens with ``tiktoken`` (cl100k_base) instead of characters, so
   the limit corresponds to real LLM context cost.
2. Split on a hierarchy of separators that preserves semantic boundaries:
   paragraphs -> lines -> sentences -> spaces -> chars. The splitter only
   moves to a finer-grained separator when the previous one cannot satisfy
   the size limit.
3. Keep a small overlap so a chunk that lands mid-paragraph still has the
   lead-in context.

The output preserves the parent's ``parent_id`` so downstream retrieval can
still hop child -> parent.
"""

from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter

from nutriai.config import (
    PARENT_CHUNK_OVERLAP,
    PARENT_MAX_TOKENS,
    TOKENIZER_ENCODING,
)


def _build_splitter(
    chunk_size: int,
    chunk_overlap: int,
    encoding: str,
) -> RecursiveCharacterTextSplitter:
    """Build a token-aware recursive splitter.

    We use ``from_tiktoken_encoder`` so ``chunk_size`` is interpreted as a
    number of tokens, not characters. The default separator hierarchy
    (``["\\n\\n", "\\n", " ", ""]``) already prioritises paragraph -> line
    -> word -> char, which is exactly what we want for cooking instructions.
    """
    return RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=encoding,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def enforce_token_limit(
    text: str,
    max_tokens: int = PARENT_MAX_TOKENS,
    overlap: int = PARENT_CHUNK_OVERLAP,
    encoding: str = TOKENIZER_ENCODING,
) -> list[str]:
    """Split ``text`` into pieces of at most ``max_tokens`` tokens.

    Returns a single-element list when ``text`` already fits, so callers can
    treat the result uniformly.
    """
    splitter = _build_splitter(max_tokens, overlap, encoding)
    pieces = splitter.split_text(text)
    return pieces or [text]


def count_tokens(text: str, encoding: str = TOKENIZER_ENCODING) -> int:
    """Count tokens with the same encoder used by the splitter."""
    import tiktoken

    enc = tiktoken.get_encoding(encoding)
    return len(enc.encode(text))
