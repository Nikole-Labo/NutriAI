"""Dataset loaders. All loaders return ``list[Recipe]``."""

from src.nutriai.data.loaders import (
    load_epicurious,
    load_huggingface_recipes,
    load_recipe1m,
    load_sample,
)

__all__ = [
    "load_sample",
    "load_recipe1m",
    "load_huggingface_recipes",
    "load_epicurious",
]
