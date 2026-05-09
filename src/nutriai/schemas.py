"""Pydantic schemas for the chunking pipeline.

The Parent-Child design is encoded directly in these models:

- ``Recipe``        : raw upstream record (after light normalisation).
- ``ParentChunk``   : one full recipe in Markdown form. Stored once. This is
                      what we return to the LLM after a child match hops up.
- ``ChildChunk``    : an atomic unit (one ingredient OR one nutrient fact).
                      Carries ``parent_id`` so retrieval can look up the
                      parent recipe in O(1).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Nutrient(BaseModel):
    """A single nutrient fact for a recipe (per serving)."""

    name: str
    amount: float
    unit: str


class Recipe(BaseModel):
    """Normalised upstream recipe, before chunking.

    All loaders (Recipe1M+, RecipeNLG, sample, ...) produce this shape so the
    rest of the pipeline does not care where the data came from.
    """

    id: str
    title: str
    ingredients: list[str] = Field(default_factory=list)
    instructions: list[str] = Field(default_factory=list)
    nutrients: list[Nutrient] = Field(default_factory=list)
    source: str = "unknown"
    url: Optional[str] = None


class ParentChunk(BaseModel):
    """A whole recipe, kept together so the LLM can read it end-to-end."""

    chunk_id: str
    parent_id: str
    kind: Literal["parent"] = "parent"
    title: str
    text: str
    ingredients: list[str]
    source: str
    url: Optional[str] = None
    n_tokens: int


class ChildChunk(BaseModel):
    """An atomic chunk (ingredient line or nutrient fact).

    We index these in Qdrant. A retrieval hit returns ``parent_id`` which we
    use to fetch the corresponding ``ParentChunk`` and feed it to the LLM.
    """

    chunk_id: str
    parent_id: str
    kind: Literal["ingredient", "nutrient"]
    text: str
    # Light-weight payload used for filtering / aggregation.
    ingredient_name: Optional[str] = None
    nutrient_name: Optional[str] = None
    nutrient_amount: Optional[float] = None
    nutrient_unit: Optional[str] = None
    n_tokens: int
