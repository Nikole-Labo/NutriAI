"""Loaders that normalise upstream datasets into our ``Recipe`` schema.

We keep loaders thin and side-effect free; they take a path / dataset name
and return a ``list[Recipe]``. The chunking pipeline does not know which
loader produced its input.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, Optional

from nutriai.data.sample import SAMPLE_RECIPES
from nutriai.schemas import Nutrient, Recipe


def load_sample() -> list[Recipe]:
    """Return the built-in 5-recipe sample. Useful for tests and demos."""
    return list(SAMPLE_RECIPES)


# ---------------------------------------------------------------------------
# Recipe1M+
# ---------------------------------------------------------------------------
def load_recipe1m(path: str | Path, limit: Optional[int] = None) -> list[Recipe]:
    """Load Recipe1M+ ``layer1.json`` (the recipe text layer).

    Each entry in layer1.json looks like::

        {
            "id": "000018c8a5",
            "title": "Worlds Best Mac and Cheese",
            "ingredients": [{"text": "1 cup elbow macaroni"}, ...],
            "instructions": [{"text": "Boil the pasta..."}, ...],
            "url": "http://...",
            "partition": "train"
        }

    Recipe1M+ does not ship per-recipe nutrition; we leave ``nutrients``
    empty here and (later) join nutrition from USDA FoodData Central in a
    separate enrichment step.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Recipe1M+ file not found at {path}. "
            "Request access at http://pic2recipe.csail.mit.edu/ and put "
            "layer1.json under data/raw/."
        )

    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    recipes: list[Recipe] = []
    for i, rec in enumerate(records):
        if limit is not None and i >= limit:
            break
        recipes.append(
            Recipe(
                id=str(rec.get("id", f"recipe1m-{i}")),
                title=rec.get("title", "").strip(),
                ingredients=[ing["text"].strip() for ing in rec.get("ingredients", [])],
                instructions=[ins["text"].strip() for ins in rec.get("instructions", [])],
                nutrients=[],
                source="recipe1m",
                url=rec.get("url"),
            )
        )
    return recipes


# ---------------------------------------------------------------------------
# Epicurious (Kaggle: Epicurious - Recipes with Rating and Nutrition)
# ---------------------------------------------------------------------------
def load_epicurious(path: str | Path, limit: Optional[int] = None) -> list[Recipe]:
    """Load ``full_format_recipes.json`` from the Epicurious CSV/JSON bundle.

    If ``path`` is a directory, reads ``<path>/full_format_recipes.json``.
    If ``path`` is a file, reads that JSON directly.

    Each record carries ``ingredients``, ``directions``, and optional per-serving
    nutrition fields (``calories``, ``protein``, ``fat``, ``sodium``) which we map
    into ``Recipe.nutrients`` for the Nutrition section in chunking.
    """
    path = Path(path)
    json_path = path / "full_format_recipes.json" if path.is_dir() else path
    if not json_path.exists():
        raise FileNotFoundError(
            f"Epicurious data not found at {json_path}. "
            "Place full_format_recipes.json in that folder (Kaggle bundle)."
        )

    with json_path.open("r", encoding="utf-8") as f:
        records = json.load(f)

    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {json_path}")

    recipes: list[Recipe] = []
    for i, rec in enumerate(records):
        if limit is not None and i >= limit:
            break

        title = str(rec.get("title", "")).strip()
        ingredients_raw = rec.get("ingredients") or []
        if isinstance(ingredients_raw, list):
            ingredients = [str(x).strip() for x in ingredients_raw if str(x).strip()]
        else:
            ingredients = []

        directions_raw = rec.get("directions") or []
        if isinstance(directions_raw, list):
            instructions = [str(x).strip() for x in directions_raw if str(x).strip()]
        else:
            instructions = []

        if not instructions:
            desc = rec.get("desc")
            if isinstance(desc, str) and desc.strip():
                instructions = [desc.strip()]

        nutrients: list[Nutrient] = []
        if _is_number(rec.get("calories")):
            nutrients.append(Nutrient(name="Calories", amount=float(rec["calories"]), unit="kcal"))
        if _is_number(rec.get("protein")):
            nutrients.append(Nutrient(name="Protein", amount=float(rec["protein"]), unit="g"))
        if _is_number(rec.get("fat")):
            nutrients.append(Nutrient(name="Fat", amount=float(rec["fat"]), unit="g"))
        if _is_number(rec.get("sodium")):
            nutrients.append(Nutrient(name="Sodium", amount=float(rec["sodium"]), unit="mg"))

        recipes.append(
            Recipe(
                id=f"epicurious-{i}",
                title=title,
                ingredients=ingredients,
                instructions=instructions,
                nutrients=nutrients,
                source="epicurious",
                url=None,
            )
        )
    return recipes


# ---------------------------------------------------------------------------
# HuggingFace recipe datasets (e.g. mbien/recipe_nlg)
# ---------------------------------------------------------------------------
def load_huggingface_recipes(
    dataset_name: str,
    split: str = "train",
    limit: Optional[int] = None,
) -> list[Recipe]:
    """Load a HuggingFace recipe dataset and normalise it.

    Supports the common ``recipe_nlg``-style schema where each row has::

        {
            "title": str,
            "ingredients": list[str],   # or JSON-encoded string
            "directions":  list[str],   # or JSON-encoded string
            "link":        str,
            "source":      str,
        }
    """
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "datasets is required for HuggingFace loading. "
            "Install with: pip install datasets"
        ) from e

    ds = load_dataset(dataset_name, split=split, streaming=True)

    recipes: list[Recipe] = []
    for i, row in enumerate(_iter_rows(ds)):
        if limit is not None and i >= limit:
            break
        recipes.append(
            Recipe(
                id=str(row.get("id", f"{dataset_name.replace('/', '_')}-{i}")),
                title=str(row.get("title", "")).strip(),
                ingredients=_coerce_string_list(row.get("ingredients")),
                instructions=_coerce_string_list(row.get("directions") or row.get("instructions")),
                nutrients=[],
                source=dataset_name,
                url=row.get("link") or row.get("url"),
            )
        )
    return recipes


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _iter_rows(ds) -> Iterator[dict]:
    for row in ds:
        yield row


def _is_number(val) -> bool:
    if val is None:
        return False
    if isinstance(val, bool):
        return False
    if isinstance(val, (int, float)):
        return True
    try:
        float(str(val).strip())
        return str(val).strip() != ""
    except ValueError:
        return False


def _coerce_string_list(value) -> list[str]:
    """Some HF datasets store the list as a JSON-encoded string."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if v]
    if isinstance(value, str):
        s = value.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    return [str(v).strip() for v in parsed if v]
            except json.JSONDecodeError:
                pass
        return [s] if s else []
    return []


# Placeholder so future Nutrient enrichment (USDA FDC) plugs in cleanly.
def attach_nutrients(recipe: Recipe, nutrients: list[Nutrient]) -> Recipe:
    return recipe.model_copy(update={"nutrients": nutrients})
