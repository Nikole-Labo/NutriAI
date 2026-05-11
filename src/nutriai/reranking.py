"""Macro-aware scoring + cross-encoder reranking over retrieved parent recipes."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Literal, Optional

import numpy as np
from pydantic import BaseModel, Field
from sentence_transformers import CrossEncoder

from nutriai.config import (
    RERANK_CE_WEIGHT,
    RERANK_CROSS_ENCODER_MODEL,
    RERANK_MACRO_WEIGHT,
    RERANK_PASSAGE_MAX_CHARS,
)
from nutriai.macro_parse import parse_nutrients_from_markdown
from nutriai.retrieval import CulinaryTools


class MacroTargets(BaseModel):
    """Targets are usually *remaining* daily allowance or a single-meal budget."""

    target_calories: Optional[float] = Field(
        default=None,
        description="Target recipe calories (e.g. remaining kcal for the day or meal slot).",
    )
    target_protein_g: Optional[float] = None
    target_carbs_g: Optional[float] = None
    target_fat_g: Optional[float] = None
    weight_calories: float = 1.0
    weight_protein: float = 0.45
    weight_carbs: float = 0.25
    weight_fat: float = 0.25


class RerankResult(BaseModel):
    parent_id: str
    final_score: float
    macro_fit: float
    ce_score_raw: float
    ce_score_norm: float
    macro_loss: float
    calories: Optional[float] = None
    title_hint: str = ""


@lru_cache(maxsize=1)
def get_cross_encoder() -> CrossEncoder:
    return CrossEncoder(RERANK_CROSS_ENCODER_MODEL)


def _macro_loss(parsed: dict[str, float], macro: MacroTargets) -> float:
    """Lower is better. Missing targets / missing nutrients incur a penalty."""
    loss = 0.0

    def miss() -> float:
        return 3.0

    if macro.target_calories is not None:
        if "calories" in parsed:
            scale = max(200.0, abs(macro.target_calories))
            d = (parsed["calories"] - macro.target_calories) / scale
            loss += macro.weight_calories * (d * d)
        else:
            loss += macro.weight_calories * miss()

    if macro.target_protein_g is not None:
        if "protein_g" in parsed:
            scale = max(10.0, abs(macro.target_protein_g))
            d = (parsed["protein_g"] - macro.target_protein_g) / scale
            loss += macro.weight_protein * (d * d)
        else:
            loss += macro.weight_protein * miss()

    if macro.target_carbs_g is not None:
        if "carbs_g" in parsed:
            scale = max(10.0, abs(macro.target_carbs_g))
            d = (parsed["carbs_g"] - macro.target_carbs_g) / scale
            loss += macro.weight_carbs * (d * d)
        else:
            loss += macro.weight_carbs * miss()

    if macro.target_fat_g is not None:
        if "fat_g" in parsed:
            scale = max(8.0, abs(macro.target_fat_g))
            d = (parsed["fat_g"] - macro.target_fat_g) / scale
            loss += macro.weight_fat * (d * d)
        else:
            loss += macro.weight_fat * miss()

    return float(loss)


def _min_max(values: list[float], *, higher_is_better: bool) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return [0.5 for _ in values]
    if higher_is_better:
        return [(v - lo) / (hi - lo) for v in values]
    hi, lo = lo, hi  # invert: lower raw -> higher norm
    return [(hi - v) / (hi - lo) for v in values]


def _intent_text(query: str, macro: MacroTargets) -> str:
    bits: list[str] = []
    q = (query or "").strip()
    if q:
        bits.append(q)
    macro_bits: list[str] = []
    if macro.target_calories is not None:
        macro_bits.append(f"Target about {macro.target_calories:g} kcal per recipe.")
    if macro.target_protein_g is not None:
        macro_bits.append(f"Protein near {macro.target_protein_g:g} g.")
    if macro.target_carbs_g is not None:
        macro_bits.append(f"Carbohydrates near {macro.target_carbs_g:g} g.")
    if macro.target_fat_g is not None:
        macro_bits.append(f"Fat near {macro.target_fat_g:g} g.")
    if macro_bits:
        bits.append(" ".join(macro_bits))
    if not bits:
        bits.append("Choose the recipe that best matches typical healthy meal goals.")
    return " ".join(bits)


def _title_from_markdown(md: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            return line[2:].strip()[:200]
    return ""


def collect_parent_ids_from_hits(hits: list[Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for h in hits:
        pl = h.payload or {}
        pid = pl.get("parent_id")
        if isinstance(pid, str) and pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out


def rerank_parents(
    tools: CulinaryTools,
    parent_ids: list[str],
    *,
    query: str,
    macro: MacroTargets,
    macro_weight: float = RERANK_MACRO_WEIGHT,
    ce_weight: float = RERANK_CE_WEIGHT,
) -> list[RerankResult]:
    """Score parents by macro fit + cross-encoder relevance to ``query`` + macro intent."""
    if not parent_ids:
        return []

    w_sum = macro_weight + ce_weight
    if w_sum <= 0:
        macro_weight, ce_weight = 0.72, 0.28
        w_sum = 1.0
    macro_w = macro_weight / w_sum
    ce_w = ce_weight / w_sum

    intent = _intent_text(query, macro)
    ce_model = get_cross_encoder()

    bodies: list[str] = []
    parsed_list: list[dict[str, float]] = []
    titles: list[str] = []
    for pid in parent_ids:
        md = tools.get_full_recipe(pid)
        bodies.append(md[:RERANK_PASSAGE_MAX_CHARS])
        parsed_list.append(parse_nutrients_from_markdown(md))
        titles.append(_title_from_markdown(md))

    losses = [_macro_loss(p, macro) for p in parsed_list]
    macro_fits = [1.0 / (1.0 + ell) for ell in losses]
    macro_norm = _min_max(macro_fits, higher_is_better=True)

    pairs = [(intent, passage) for passage in bodies]
    ce_raw_arr = np.asarray(ce_model.predict(pairs, show_progress_bar=False), dtype=np.float64).reshape(-1)
    ce_raw = [float(x) for x in ce_raw_arr.tolist()]
    ce_norm = _min_max(ce_raw, higher_is_better=True)

    results: list[RerankResult] = []
    for i, pid in enumerate(parent_ids):
        fin = macro_w * macro_norm[i] + ce_w * ce_norm[i]
        results.append(
            RerankResult(
                parent_id=pid,
                final_score=float(fin),
                macro_fit=float(macro_fits[i]),
                ce_score_raw=float(ce_raw[i]),
                ce_score_norm=float(ce_norm[i]),
                macro_loss=float(losses[i]),
                calories=parsed_list[i].get("calories"),
                title_hint=titles[i],
            )
        )
    results.sort(key=lambda r: r.final_score, reverse=True)
    return results


def search_ingredients_with_macro_rerank(
    tools: CulinaryTools,
    *,
    ingredients: list[str],
    macro: MacroTargets,
    retrieval_mode: Literal["hybrid", "dense", "sparse"] = "hybrid",
    hits_per_ingredient: int = 12,
    top_k: int = 5,
    macro_weight: Optional[float] = None,
    ce_weight: Optional[float] = None,
) -> tuple[list[RerankResult], list[str]]:
    """Union parent ids from per-ingredient child searches, then macro + CE rerank."""
    seen: set[str] = set()
    pids_ordered: list[str] = []
    for ing in ingredients:
        q = ing.strip()
        if not q:
            continue
        hits = tools.search_children(q, limit=hits_per_ingredient, kind="ingredient", mode=retrieval_mode)
        for h in hits:
            pid = (h.payload or {}).get("parent_id")
            if isinstance(pid, str) and pid not in seen:
                seen.add(pid)
                pids_ordered.append(pid)
    query = ", ".join(x.strip() for x in ingredients if x.strip())
    mw = macro_weight if macro_weight is not None else RERANK_MACRO_WEIGHT
    cw = ce_weight if ce_weight is not None else RERANK_CE_WEIGHT
    ranked = rerank_parents(
        tools, pids_ordered, query=query, macro=macro, macro_weight=mw, ce_weight=cw
    )
    return ranked[:top_k], pids_ordered


def search_recipes_with_macro_rerank(
    tools: CulinaryTools,
    *,
    query: str,
    macro: MacroTargets,
    kind: Optional[str] = None,
    retrieval_mode: Literal["hybrid", "dense", "sparse"] = "hybrid",
    candidate_child_hits: int = 48,
    top_k: int = 5,
    macro_weight: Optional[float] = None,
    ce_weight: Optional[float] = None,
) -> tuple[list[RerankResult], list[Any]]:
    """
    Hybrid (etc.) child search → unique ``parent_id`` list → macro + CE rerank.

    Returns ``(reranked, raw_child_hits)``.
    """
    hits = tools.search_children(
        query.strip(),
        limit=candidate_child_hits,
        kind=kind,
        mode=retrieval_mode,
    )
    pids = collect_parent_ids_from_hits(hits)
    mw = macro_weight if macro_weight is not None else RERANK_MACRO_WEIGHT
    cw = ce_weight if ce_weight is not None else RERANK_CE_WEIGHT
    ranked = rerank_parents(tools, pids, query=query, macro=macro, macro_weight=mw, ce_weight=cw)
    return ranked[:top_k], hits


def describe_rerank_setup() -> dict[str, float | str]:
    return {
        "cross_encoder": RERANK_CROSS_ENCODER_MODEL,
        "macro_weight_default": RERANK_MACRO_WEIGHT,
        "ce_weight_default": RERANK_CE_WEIGHT,
        "passage_chars": RERANK_PASSAGE_MAX_CHARS,
    }
