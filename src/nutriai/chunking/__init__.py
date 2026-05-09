"""Chunking pipeline for NutriAI.

The flow is:

    Recipe (pydantic)
        |  recipe_to_markdown
        v
    Markdown text  -- structural Header splitter -> parent sections
        |
        |  RecursiveCharacterTextSplitter (fallback if too long)
        v
    ParentChunk(s)         <-- stored, returned to the LLM
        |
        |  per-ingredient and per-nutrient extraction
        v
    ChildChunk(s)          <-- indexed in Qdrant, link to parent_id
"""

from nutriai.chunking.markdown_converter import recipe_to_markdown
from nutriai.chunking.parent_child import build_chunks, split_recipes
from nutriai.chunking.recursive_fallback import enforce_token_limit

__all__ = [
    "recipe_to_markdown",
    "build_chunks",
    "split_recipes",
    "enforce_token_limit",
]
