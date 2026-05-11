"""Quick retrieval smoke test against Qdrant (hybrid / dense / sparse).

Requires:
  - Indexed collection (run ``scripts/index_qdrant.py`` first)
  - ``.env`` with ``QDRANT_URL`` + ``QDRANT_API_KEY`` (Cloud) or local Qdrant

Examples::

    cd C:\\Users\\lazab\\NutriAI
    .\\.venv\\Scripts\\Activate.ps1
    python scripts/smoke_retrieval.py
    python scripts/smoke_retrieval.py --query "comfort food dinner" --limit 3
    python scripts/smoke_retrieval.py --ingredients "chicken,rice,tomato"
    python scripts/smoke_retrieval.py --compare --query "steak"
    python scripts/smoke_retrieval.py --parent-id sample-001

Macro + cross-encoder rerank::

    python scripts/smoke_retrieval.py --query "protein meal" --rerank --cal 450 --protein 35
    python scripts/smoke_retrieval.py --ingredients "chicken,rice" --rerank --cal 500 --protein 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

load_dotenv(PROJECT_ROOT / ".env")

from nutriai.config import QDRANT_COLLECTION_NAME  # noqa: E402
from nutriai.reranking import (  # noqa: E402
    MacroTargets,
    search_ingredients_with_macro_rerank,
    search_recipes_with_macro_rerank,
)
from nutriai.retrieval import CulinaryTools  # noqa: E402


def _kind_arg(s: str) -> str | None:
    if s == "any":
        return None
    return s


def print_hits(label: str, hits: list) -> None:
    print(f"\n=== {label} ({len(hits)} hits) ===")
    for i, hit in enumerate(hits, 1):
        pl = hit.payload or {}
        score = getattr(hit, "score", None)
        line = (
            f"{i}. score={score!r} kind={pl.get('kind')} parent_id={pl.get('parent_id')!r} "
            f"text={json.dumps(pl.get('text', '')[:120])}"
        )
        print(line)


def _macro_from_args(args: argparse.Namespace) -> MacroTargets:
    return MacroTargets(
        target_calories=args.cal,
        target_protein_g=args.protein,
        target_carbs_g=args.carbs,
        target_fat_g=args.fat,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Smoke-test NutriAI Qdrant retrieval.")
    p.add_argument("--query", default="steak", help="Free-text query for search_children.")
    p.add_argument("--mode", choices=["hybrid", "dense", "sparse"], default="hybrid")
    p.add_argument("--kind", choices=["ingredient", "nutrient", "any"], default="any")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument(
        "--ingredients",
        metavar="LIST",
        help="Comma-separated ingredients (runs search_by_ingredients instead of search_children).",
    )
    p.add_argument("--parent-id", metavar="ID", help="Print get_full_recipe(parent_id) and exit.")
    p.add_argument(
        "--compare",
        action="store_true",
        help="Run hybrid, dense, and sparse for --query and print each (ignores --mode).",
    )
    p.add_argument(
        "--rerank",
        action="store_true",
        help="After retrieval, macro-fit + cross-encoder rerank of parent recipes.",
    )
    p.add_argument("--cal", type=float, default=None, help="Target recipe calories (kcal), e.g. remaining daily budget.")
    p.add_argument("--protein", type=float, default=None, help="Target protein (g).")
    p.add_argument("--carbs", type=float, default=None, help="Target carbohydrates (g).")
    p.add_argument("--fat", type=float, default=None, help="Target fat (g).")
    p.add_argument(
        "--candidate-hits",
        type=int,
        default=48,
        help="Child hits to pool before reranking (``--rerank`` only).",
    )
    args = p.parse_args()

    tools = CulinaryTools()

    if args.parent_id:
        text = tools.get_full_recipe(args.parent_id)
        print(f"--- get_full_recipe({args.parent_id!r}) ---\n{text[:2000]}")
        if len(text) > 2000:
            print("\n... [truncated] ...")
        return 0

    macro = _macro_from_args(args)

    if args.ingredients:
        parts = [x.strip() for x in args.ingredients.split(",") if x.strip()]
        if args.rerank:
            ranked, pool = search_ingredients_with_macro_rerank(
                tools,
                ingredients=parts,
                macro=macro,
                retrieval_mode=args.mode,  # type: ignore[arg-type]
                top_k=args.limit,
            )
            print(f"\n=== reranked parents (ingredients {parts!r}, pool={len(pool)}) ===")
            for i, r in enumerate(ranked, 1):
                print(
                    f"{i}. parent_id={r.parent_id!r} final={r.final_score:.4f} "
                    f"macro_fit={r.macro_fit:.4f} ce={r.ce_score_raw:.4f} "
                    f"kcal={r.calories} title={json.dumps(r.title_hint[:60])}"
                )
            return 0
        for mode in (["hybrid", "dense", "sparse"] if args.compare else [args.mode]):
            ranked = tools.search_by_ingredients(parts, limit=args.limit, mode=mode)  # type: ignore[arg-type]
            print(f"\n=== search_by_ingredients mode={mode} ({parts!r}) ===")
            for pid, cnt in ranked:
                print(f"  parent_id={pid!r}  hit_count={cnt}")
        return 0

    kind = _kind_arg(args.kind)

    if args.rerank:
        ranked, raw = search_recipes_with_macro_rerank(
            tools,
            query=args.query,
            macro=macro,
            kind=kind,
            retrieval_mode=args.mode,  # type: ignore[arg-type]
            candidate_child_hits=args.candidate_hits,
            top_k=args.limit,
        )
        print_hits(f"child retrieval query={args.query!r} mode={args.mode}", raw)
        print(f"\n=== reranked parents (macro + cross-encoder, top {args.limit}) ===")
        for i, r in enumerate(ranked, 1):
            print(
                f"{i}. parent_id={r.parent_id!r} final={r.final_score:.4f} "
                f"macro_fit={r.macro_fit:.4f} ce_raw={r.ce_score_raw:.4f} "
                f"macro_loss={r.macro_loss:.4f} kcal={r.calories} title={json.dumps(r.title_hint[:60])}"
            )
        info = tools.client.get_collection(QDRANT_COLLECTION_NAME)
        n = getattr(info, "points_count", "?")
        print(f"\n(collection {QDRANT_COLLECTION_NAME!r} points_count={n})")
        return 0

    if args.compare:
        for mode in ("hybrid", "dense", "sparse"):
            hits = tools.search_children(args.query, limit=args.limit, kind=kind, mode=mode)  # type: ignore[arg-type]
            print_hits(f"search_children query={args.query!r} mode={mode}", hits)
        return 0

    hits = tools.search_children(args.query, limit=args.limit, kind=kind, mode=args.mode)  # type: ignore[arg-type]
    print_hits(f"search_children query={args.query!r} mode={args.mode}", hits)

    info = tools.client.get_collection(QDRANT_COLLECTION_NAME)
    n = getattr(info, "points_count", "?")
    print(f"\n(collection {QDRANT_COLLECTION_NAME!r} points_count={n})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
