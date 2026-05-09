# NutriAI

A RAG-based assistant that helps users eat healthier and cook meals with the
ingredients they already have at home.

**Team:** Malos Alexandru, Laza Bogdan, Labo Nikole

---

## Architecture (high level)

| Stage          | Choice                                                                 |
| -------------- | ---------------------------------------------------------------------- |
| Datasets       | Recipe1M+ (recipes) + USDA FoodData Central (nutrition)                |
| Chunking       | Markdown-Header structural + Parent-Child + Recursive fallback         |
| Vector DB      | Qdrant (hybrid dense + sparse, nested payload filtering)               |
| Retrieval      | Hybrid (BM25 sparse + dense embeddings) + Cross-Encoder reranker       |
| Generator LLM  | SmolLM-1.7B-Instruct                                                   |

This repo currently implements **Stage 1: Data Chunking**.

---

## Project layout

```
NutriAI/
├── data/
│   ├── raw/            # Raw Recipe1M+ / RecipeNLG / USDA dumps (gitignored)
│   ├── interim/        # Recipes converted to Markdown (gitignored)
│   └── processed/      # Final parent + child chunks (JSONL, gitignored)
├── src/nutriai/
│   ├── config.py
│   ├── schemas.py      # Pydantic models: Recipe, ParentChunk, ChildChunk
│   ├── data/           # Sample data + Recipe1M+ / RecipeNLG loaders
│   └── chunking/       # Markdown converter + Parent-Child + Recursive fallback
├── scripts/
│   └── run_chunking.py # CLI entry point
└── tests/
```

---

## Setup

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy env template and fill in keys (only needed for USDA / HuggingFace)
copy .env.example .env
```

---

## Stage 1 — Run the chunking pipeline

The pipeline is dataset-agnostic. Out of the box it ships with a small
hand-crafted sample so you can run it without downloading anything.

```powershell
# Run on the built-in sample (5 recipes)
python scripts/run_chunking.py --source sample --out data/processed

# Run on a HuggingFace recipe dataset (e.g. RecipeNLG mirror)
python scripts/run_chunking.py --source hf --hf-name mbien/recipe_nlg --limit 1000 --out data/processed

# Run on a local Recipe1M+ JSON file (after you obtain access)
python scripts/run_chunking.py --source recipe1m --path data/raw/layer1.json --limit 1000 --out data/processed
```

Output:

- `data/processed/parents.jsonl` — one full recipe per line (the **parent** docs)
- `data/processed/children.jsonl` — per-ingredient and per-nutrient atomic chunks,
  each carrying `parent_id` so retrieval can hop child → parent

---

## Datasets — how to obtain them

### Recipe1M+
Recipe1M+ is hosted by MIT CSAIL and requires registration:
http://pic2recipe.csail.mit.edu/ . After approval, download `layer1.json`
(recipes + ingredients + instructions) into `data/raw/`.

### RecipeNLG (recommended for prototyping)
Drop-in alternative available on HuggingFace without registration:
https://huggingface.co/datasets/mbien/recipe_nlg

### USDA FoodData Central (nutrition metadata)
Free API key (instant): https://fdc.nal.usda.gov/api-key-signup.html
Put the key in `.env` as `USDA_FDC_API_KEY=...`.

---

## Next stages (not in this repo yet)

2. **Embeddings & Indexing** — encode chunks (dense + sparse) and load into Qdrant.
3. **Retrieval** — hybrid search + cross-encoder reranker scoring on macro-fit.
4. **Generation** — SmolLM-1.7B-Instruct trims retrieved recipe to the user's macros.
