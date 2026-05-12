"""Parent-Child chunking for NutriAI.

Strategy (matches the design doc):

* **Parent chunk**: one full recipe Markdown document. Kept whole so the
  LLM receives a coherent, end-to-end recipe at generation time.
* **Child chunks**: atomic units extracted from the parent. Today we emit
  one child per ingredient line and one child per nutrient fact. Children
  carry ``parent_id`` so a child match in Qdrant can hop up to the parent.

Why this layout works:

* **Hybrid retrieval**: the children are short and lexical-friendly, so
  BM25 (sparse) finds exact ingredient hits like "chickpeas" with high
  precision. Dense embeddings on the same children capture the culinary
  meaning ("plant protein", "legume") for soft matches.
* **Multi-ingredient queries**: when the user lists 5 things from the
  fridge, we retrieve children matching each, group by ``parent_id`` and
  rank parents by hit-count -- exactly the "cover-the-most-ingredients"
  behaviour described in the design.
* **Macro reranking**: the per-nutrient children let the cross-encoder
  reranker score parents on macro-fit without re-parsing the parent
  Markdown.

For the structural step we use LangChain's ``MarkdownHeaderTextSplitter``
to locate the ``## Ingredients`` and ``## Nutrition`` sections cleanly,
even when ingredient lines contain dashes or numbers that would confuse a
naive regex.
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterable

from langchain_text_splitters import MarkdownHeaderTextSplitter

from nutriai.chunking.markdown_converter import recipe_to_markdown
from nutriai.chunking.recursive_fallback import count_tokens, enforce_token_limit
from nutriai.config import (
    CHILD_MAX_TOKENS,
    MARKDOWN_HEADERS_TO_SPLIT_ON,
    PARENT_MAX_TOKENS,
)
from nutriai.schemas import ChildChunk, ParentChunk, Recipe


_MARKDOWN_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=MARKDOWN_HEADERS_TO_SPLIT_ON,
    strip_headers=False,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def split_recipes(
    recipes: Iterable[Recipe],
) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """Run the full parent + child chunking pipeline on a recipe iterable."""
    parents: list[ParentChunk] = []
    children: list[ChildChunk] = []
    for recipe in recipes:
        p, c = build_chunks(recipe)
        parents.extend(p)
        children.extend(c)
    return parents, children


def build_chunks(recipe: Recipe) -> tuple[list[ParentChunk], list[ChildChunk]]:
    """Produce the parent + child chunks for a single recipe."""
    markdown = recipe_to_markdown(recipe)

    # Use the Markdown-Header splitter just to *parse* sections cleanly; we
    # still want a single parent per recipe, so we re-stitch the parts.
    sections = _MARKDOWN_SPLITTER.split_text(markdown)

    # Group ingredients for the parent payload (used later for hybrid filtering).
    ingredient_lines = _extract_section_lines(sections, section="Ingredients")
    nutrient_lines = _extract_section_lines(sections, section="Nutrition")

    parents = _build_parent_chunks(recipe, markdown, ingredient_lines)
    children = _build_child_chunks(recipe, ingredient_lines, nutrient_lines)
    return parents, children


# ---------------------------------------------------------------------------
# Parent chunks
# ---------------------------------------------------------------------------
def _build_parent_chunks(
    recipe: Recipe,
    markdown: str,
    ingredient_lines: list[str],
) -> list[ParentChunk]:
    """Return one parent per recipe, falling back to recursive split if too long."""
    n_tokens = count_tokens(markdown)

    if n_tokens <= PARENT_MAX_TOKENS:
        return [
            ParentChunk(
                chunk_id=f"{recipe.id}::parent",
                parent_id=recipe.id,
                title=recipe.title,
                text=markdown,
                ingredients=ingredient_lines,
                source=recipe.source,
                url=recipe.url,
                n_tokens=n_tokens,
            )
        ]

    # Oversized -> recursive fallback. Keep parent_id stable, suffix the chunk_id.
    pieces = enforce_token_limit(markdown)
    return [
        ParentChunk(
            chunk_id=f"{recipe.id}::parent::{i}",
            parent_id=recipe.id,
            title=recipe.title,
            text=piece,
            ingredients=ingredient_lines,
            source=recipe.source,
            url=recipe.url,
            n_tokens=count_tokens(piece),
        )
        for i, piece in enumerate(pieces)
    ]


# ---------------------------------------------------------------------------
# Child chunks
# ---------------------------------------------------------------------------
_NUTRIENT_LINE_RE = re.compile(
    r"^\s*-\s*(?P<name>[^:]+?)\s*:\s*(?P<amount>[-+]?\d*\.?\d+)\s*(?P<unit>[A-Za-z%]+)?\s*$"
)


def _build_child_chunks(
    recipe: Recipe,
    ingredient_lines: list[str],
    nutrient_lines: list[str],
) -> list[ChildChunk]:
    children: list[ChildChunk] = []

    for ing_line in ingredient_lines:
        text = _normalise_ingredient(ing_line)
        if not text:
            continue
        chunk_id = _stable_id(recipe.id, "ingredient", text)
        n_tokens = count_tokens(text)
        # Children should be short by construction. Hard-trim any rare outlier
        # so the child index stays uniform.
        if n_tokens > CHILD_MAX_TOKENS:
            text = enforce_token_limit(text, max_tokens=CHILD_MAX_TOKENS, overlap=0)[0]
            n_tokens = count_tokens(text)
        children.append(
            ChildChunk(
                chunk_id=chunk_id,
                parent_id=recipe.id,
                kind="ingredient",
                text=text,
                ingredient_name=_extract_ingredient_name(text),
                n_tokens=n_tokens,
            )
        )

    for nut_line in nutrient_lines:
        parsed = _parse_nutrient_line(nut_line)
        if parsed is None:
            continue
        name, amount, unit = parsed
        text = f"{name}: {amount} {unit}".strip()
        chunk_id = _stable_id(recipe.id, "nutrient", name)
        children.append(
            ChildChunk(
                chunk_id=chunk_id,
                parent_id=recipe.id,
                kind="nutrient",
                text=text,
                nutrient_name=name,
                nutrient_amount=amount,
                nutrient_unit=unit,
                n_tokens=count_tokens(text),
            )
        )

    return children


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _extract_section_lines(sections, section: str) -> list[str]:
    """Pull bullet-list lines from a named ``## <section>`` document."""
    target = section.strip().lower()
    for doc in sections:
        meta_section = (doc.metadata.get("section") or "").strip().lower()
        if meta_section != target:
            continue
        return _bullets(doc.page_content)
    return []


def _bullets(block: str) -> list[str]:
    """Return the ``- ...`` bullet lines from a Markdown block."""
    out: list[str] = []
    for raw in block.splitlines():
        line = raw.strip()
        if line.startswith("- "):
            out.append(line[2:].strip())
    return out


def _normalise_ingredient(line: str) -> str:
    """Collapse whitespace and strip trailing punctuation noise."""
    return re.sub(r"\s+", " ", line).strip().rstrip(".")


def _extract_ingredient_name(line: str) -> str:
    """Best-effort extraction of the ingredient name (drops quantity/unit).

    We strip a leading "<number><unit?>" token if present. The downstream
    BM25 index will use the full text anyway; this field is for filtering
    and aggregation only, so heuristic-quality is fine.
    """
    tokens = line.split()
    out: list[str] = []
    skipped_quantity = False
    for tok in tokens:
        if not skipped_quantity and re.match(r"^[-+]?\d+([./]\d+)?$", tok):
            skipped_quantity = True
            continue
        if not skipped_quantity and re.match(r"^[-+]?\d*\.?\d+[a-zA-Z]+$", tok):
            skipped_quantity = True
            continue
        out.append(tok)
    name = " ".join(out).strip(",. ")
    # Drop a leading unit word like "tbsp", "g", "cup" etc.
    units = {
        "g", "kg", "mg", "ml", "l", "tsp", "tbsp", "cup", "cups",
        "oz", "lb", "pinch", "dash", "clove", "cloves",
    }
    parts = name.split()
    if parts and parts[0].lower() in units:
        parts = parts[1:]
    return " ".join(parts).strip(",. ").lower()


def _parse_nutrient_line(line: str) -> tuple[str, float, str] | None:
    """Parse ``Calories: 520.0 kcal`` -> ``("Calories", 520.0, "kcal")``."""
    m = _NUTRIENT_LINE_RE.match(f"- {line}" if not line.startswith("-") else line)
    if not m:
        return None
    name = m.group("name").strip()
    amount = float(m.group("amount"))
    unit = (m.group("unit") or "").strip()
    return name, amount, unit


def _stable_id(parent_id: str, kind: str, payload: str) -> str:
    """Deterministic, collision-resistant child id."""
    h = hashlib.sha1(payload.lower().encode("utf-8")).hexdigest()[:10]
    return f"{parent_id}::{kind}::{h}"
