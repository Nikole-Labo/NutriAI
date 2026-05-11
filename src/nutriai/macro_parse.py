"""Parse per-serving nutrients from parent Markdown (``## Nutrition`` section)."""

from __future__ import annotations

import re

_NUT_BULLET = re.compile(
    r"-\s*(?P<name>[^:]+?)\s*:\s*(?P<amount>[-+]?\d*\.?\d+)\s*(?P<unit>[A-Za-z%]+)?"
)


def _nutrition_block(markdown: str) -> str:
    if "## Nutrition" not in markdown:
        return ""
    start = markdown.index("## Nutrition") + len("## Nutrition")
    rest = markdown[start:]
    m = re.search(r"\n##\s+\S", rest)
    if m:
        rest = rest[: m.start()]
    return rest


def parse_nutrients_from_markdown(markdown: str) -> dict[str, float]:
    """Canonical keys: ``calories`` (kcal), ``protein_g``, ``fat_g``, ``carbs_g``."""
    block = _nutrition_block(markdown)
    out: dict[str, float] = {}
    for line in block.splitlines():
        m = _NUT_BULLET.search(line)
        if not m:
            continue
        name = m.group("name").strip().lower()
        amt = float(m.group("amount"))
        if "calorie" in name or name in ("energy", "kcal"):
            out["calories"] = amt
        elif "protein" in name:
            out["protein_g"] = amt
        elif "carb" in name:
            out["carbs_g"] = amt
        elif "fat" in name and "saturat" not in name and "trans" not in name:
            out["fat_g"] = amt
    return out
