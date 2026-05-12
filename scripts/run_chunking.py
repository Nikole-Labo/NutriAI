"""Run the chunking pipeline on a chosen data source.

Examples (PowerShell)::

    # Built-in 5-recipe sample (no downloads required)
    python scripts/run_chunking.py --source sample

    # HuggingFace recipe dataset (RecipeNLG mirror)
    python scripts/run_chunking.py --source hf --hf-name mbien/recipe_nlg --limit 1000

    # Recipe1M+ layer1.json after you obtain access
    python scripts/run_chunking.py --source recipe1m --path data/raw/layer1.json --limit 1000

    # Epicurious (Kaggle bundle: full_format_recipes.json under recipe_dataset/)
    python scripts/run_chunking.py --source epicurious --path recipe_dataset

Output (under ``--out``):
    parents.jsonl   one full recipe per line
    children.jsonl  one ingredient or nutrient per line, with parent_id
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make ``src`` importable when running the script directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.nutriai.chunking import split_recipes  # noqa: E402
from src.nutriai.config import PROCESSED_DIR  # noqa: E402
from src.nutriai.data import (  # noqa: E402
    load_epicurious,
    load_huggingface_recipes,
    load_recipe1m,
    load_sample,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the NutriAI chunking pipeline.")
    p.add_argument(
        "--source",
        choices=["sample", "hf", "recipe1m", "epicurious"],
        default="sample",
        help="Where to read recipes from.",
    )
    p.add_argument(
        "--path",
        type=Path,
        help="Data path: Recipe1M+ JSON, or Epicurious folder / full_format_recipes.json.",
    )
    p.add_argument("--hf-name", type=str, help="HuggingFace dataset name (for --source hf).")
    p.add_argument("--hf-split", type=str, default="train", help="HuggingFace split.")
    p.add_argument("--limit", type=int, default=None, help="Cap the number of recipes ingested.")
    p.add_argument("--out", type=Path, default=PROCESSED_DIR, help="Output directory for JSONL files.")
    return p.parse_args()


def write_jsonl(path: Path, items) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")


def main() -> int:
    args = parse_args()

    if args.source == "sample":
        recipes = load_sample()
    elif args.source == "hf":
        if not args.hf_name:
            raise SystemExit("--hf-name is required when --source hf")
        recipes = load_huggingface_recipes(args.hf_name, split=args.hf_split, limit=args.limit)
    elif args.source == "recipe1m":
        if not args.path:
            raise SystemExit("--path is required when --source recipe1m")
        recipes = load_recipe1m(args.path, limit=args.limit)
    elif args.source == "epicurious":
        epicurious_path = args.path or (PROJECT_ROOT / "recipe_dataset")
        recipes = load_epicurious(epicurious_path, limit=args.limit)
    else:  # defensive
        raise SystemExit(f"Unknown source: {args.source}")

    if args.limit is not None:
        recipes = recipes[: args.limit]

    print(f"Loaded {len(recipes)} recipes from source={args.source}.")

    parents, children = split_recipes(recipes)
    print(f"Produced {len(parents)} parent chunks, {len(children)} child chunks.")

    parents_path = args.out / "parents.jsonl"
    children_path = args.out / "children.jsonl"
    write_jsonl(parents_path, parents)
    write_jsonl(children_path, children)

    print(f"Wrote {parents_path}")
    print(f"Wrote {children_path}")

    # Tiny sanity preview so the user sees what came out.
    if parents:
        sample = parents[0]
        print("\n--- Sample parent ---")
        print(f"id={sample.chunk_id}  tokens={sample.n_tokens}")
        print(sample.text[:400] + ("..." if len(sample.text) > 400 else ""))
    if children:
        print("\n--- First 5 children ---")
        for c in children[:5]:
            print(f"[{c.kind:10s}] {c.chunk_id} -> {c.text}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
