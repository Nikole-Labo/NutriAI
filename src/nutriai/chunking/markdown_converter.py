"""Convert a normalised ``Recipe`` into a Markdown document.

The Markdown layout is intentional: it produces predictable ``##`` boundaries
that ``MarkdownHeaderTextSplitter`` can split on without ever cutting an
instruction in half.

Layout::

    # <Title>

    ## Ingredients
    - <ingredient line 1>
    - <ingredient line 2>
    ...

    ## Instructions
    1. <step 1>
    2. <step 2>
    ...

    ## Nutrition          (only if nutrients are present)
    - <Nutrient>: <amount> <unit>
    ...
"""

from __future__ import annotations

from nutriai.schemas import Recipe


def recipe_to_markdown(recipe: Recipe) -> str:
    """Render a Recipe as a Markdown document with ``##`` section headers."""
    parts: list[str] = []

    title = recipe.title.strip() or "Untitled Recipe"
    parts.append(f"# {title}\n")

    if recipe.ingredients:
        parts.append("## Ingredients")
        for ing in recipe.ingredients:
            cleaned = ing.strip()
            if cleaned:
                parts.append(f"- {cleaned}")
        parts.append("")  # blank line between sections

    if recipe.instructions:
        parts.append("## Instructions")
        for i, step in enumerate(recipe.instructions, start=1):
            cleaned = step.strip()
            if cleaned:
                parts.append(f"{i}. {cleaned}")
        parts.append("")

    if recipe.nutrients:
        parts.append("## Nutrition")
        for nut in recipe.nutrients:
            parts.append(f"- {nut.name}: {nut.amount} {nut.unit}".rstrip())
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"
