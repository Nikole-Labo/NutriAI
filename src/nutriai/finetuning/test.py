"""Quick manual test: hybrid + macro rerank (reads Qdrant from project root ``.env``)."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_ROOT = _SRC_ROOT.parent
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from dotenv import load_dotenv

from nutriai.reranking import MacroTargets, search_ingredients_with_macro_rerank
from nutriai.retrieval import CulinaryTools


def fast_test() -> None:
    load_dotenv(_PROJECT_ROOT / ".env")

    print("--- Initializing Culinary hybrid pipeline ---")
    tools = CulinaryTools()

    my_goals = MacroTargets(
        target_calories=500,
        target_protein_g=40,
        target_carbs_g=50,
        target_fat_g=15,
    )

    ingredients = ["chicken", "spinach", "lemon"]

    print(f"Testing hybrid search + macro rerank for: {ingredients}")

    try:
        results, pids = search_ingredients_with_macro_rerank(
            tools,
            ingredients=ingredients,
            macro=my_goals,
            top_k=3,
        )

        print(f"\nSuccessfully retrieved and reranked {len(results)} recipes (pool={len(pids)}):")
        for i, res in enumerate(results):
            print(f"{i + 1}. {res.title_hint} (score: {res.final_score:.2f})")
            print(f"   Macros: {res.calories} kcal | macro_fit: {res.macro_fit:.2f}")
            print(f"   ID: {res.parent_id}\n")

    except Exception as e:
        print(f"Pipeline failed: {e}")


if __name__ == "__main__":
    fast_test()
