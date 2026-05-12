"""Minimal smoke tests for the chunking pipeline.

Run with: pytest -q
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.nutriai.chunking import build_chunks, recipe_to_markdown, split_recipes
from src.nutriai.data import load_sample
from src.nutriai.schemas import ChildChunk, ParentChunk


def test_markdown_has_required_headers() -> None:
    recipe = load_sample()[0]
    md = recipe_to_markdown(recipe)
    assert md.startswith(f"# {recipe.title}")
    assert "## Ingredients" in md
    assert "## Instructions" in md
    assert "## Nutrition" in md


def test_each_recipe_has_one_parent() -> None:
    recipes = load_sample()
    parents, _ = split_recipes(recipes)
    parent_ids = {p.parent_id for p in parents}
    assert parent_ids == {r.id for r in recipes}, "every recipe must produce a parent"


def test_children_link_back_to_parent() -> None:
    recipes = load_sample()
    parents, children = split_recipes(recipes)
    parent_ids = {p.parent_id for p in parents}
    for child in children:
        assert child.parent_id in parent_ids, f"orphan child {child.chunk_id}"


def test_one_child_per_ingredient_and_nutrient() -> None:
    recipe = load_sample()[0]  # Shepherd's Pie
    parents, children = build_chunks(recipe)
    n_ing = sum(1 for c in children if c.kind == "ingredient")
    n_nut = sum(1 for c in children if c.kind == "nutrient")
    assert n_ing == len(recipe.ingredients)
    assert n_nut == len(recipe.nutrients)


def test_parent_token_count_is_set() -> None:
    parents, _ = split_recipes(load_sample())
    for p in parents:
        assert isinstance(p, ParentChunk)
        assert p.n_tokens > 0


def test_nutrient_child_payload_is_parsed() -> None:
    recipe = load_sample()[0]
    _, children = build_chunks(recipe)
    cals = [c for c in children if c.kind == "nutrient" and c.nutrient_name == "Calories"]
    assert len(cals) == 1
    nutrient: ChildChunk = cals[0]
    assert nutrient.nutrient_amount == 520.0
    assert nutrient.nutrient_unit == "kcal"
