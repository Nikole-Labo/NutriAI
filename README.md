# NutriAI: Nutrition Assistant

**Team name:** AIChefs  

**Team members:**

- Laza Bogdan  
- Labo Nikole  
- Maloș Alexandru  

NutriAI is an intelligent culinary assistant designed to seamlessly integrate recipe discovery, macronutrient tracking, and personalized meal planning into a single conversation. It moves beyond reactive tracking to proactively suggest macro-friendly meals based on a user's real-time ingredient inventory, previous consumption, and dynamic preferences (e.g., using perishable items, specific cravings).

### Goals

- Craft personalized recipes aligned with daily calorie and macronutrient targets based on biometric data (gender, height, weight, age) and fitness goals.
- Build persistent user profiles through a continuous feedback loop (rating recipes and ingredients) to refine flavor preferences and dietary restrictions.
- Address limitations of existing solutions (traditional recipe discovery, reactive tracking, rigid automated planning) by offering a seamless, conversation-driven integration that adapts in real-time.

---

## Architecture overview

Higher-level diagrams and narrative live in **`Architecture.md`**.

The system uses an agentic culinary assistant architecture built on:

- **Custom RAG pipeline:** Recipe data, nutritional metadata (USDA FoodData Central), and (later) user inventory in a vector database for retrieval and recipe adaptation.
- **Model:** fine-tuned **SmolLM-1.7B (Instruct)** as the reasoning agent, with a **ReAct** loop for orchestration.

#### Detailed technical choices

- **Chunking:** Markdown-header structural chunking (Ingredients / Instructions); **parent–child** indexing (ingredients and nutrient lines as children linked to full-recipe parents); **recursive token-limited splitting** as a fallback for long parents (see codebase under `src/nutriai/chunking/`).
- **Vector database:** **Qdrant** — hybrid search, payloads (e.g. `parent_id` on children), nested filtering where needed.
- **RAG:** hybrid **dense + sparse (BM25)** retrieval; **cross-encoder reranking** for macro-fit-style scoring.

| Stage           | Choice |
| ---------------- | ------ |
| Datasets (data) | Recipe sources + USDA FoodData Central; prototyping uses RecipeNLG / Epicurious loaders |
| Chunking       | Implemented in this repo (**Stage 1**) |
| Vector DB       | Qdrant (planned **Stage 2**) |
| Generator       | SmolLM-1.7B-Instruct (later stages) |

---

## Current status

This repository implements **Stage 1: data chunking** — normalized recipes → Markdown parents + ingredient/nutrient child chunks (`parents.jsonl` / `children.jsonl`). Embeddings, Qdrant, retrieval API, and the chat UI are **not** wired up here yet.

---

## Project layout

```
NutriAI/
├── Architecture.md      # Architecture write-up from the team repo
├── data/
│   ├── raw/             # Raw dumps (gitignored)
│   ├── interim/        # Intermediate artifacts (gitignored)
│   └── processed/       # Chunk output JSONL (gitignored)
├── src/nutriai/
│   ├── config.py
│   ├── schemas.py
│   ├── data/            # Loaders (sample, Recipe1M+, HF, Epicurious, …)
│   └── chunking/
├── scripts/
│   └── run_chunking.py
└── tests/
```

---

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

---

## Stage 1 — Run the chunking pipeline

```powershell
# Built-in sample (no downloads)
python scripts/run_chunking.py --source sample --out data/processed

# HuggingFace RecipeNLG mirror
python scripts/run_chunking.py --source hf --hf-name mbien/recipe_nlg --limit 1000 --out data/processed

# Recipe1M+ layer1.json (after you obtain access)
python scripts/run_chunking.py --source recipe1m --path data/raw/layer1.json --limit 1000 --out data/processed

# Epicurious (full_format_recipes.json in recipe_dataset/, not committed — download separately)
python scripts/run_chunking.py --source epicurious --path recipe_dataset --out data/processed
```

Output:

- `data/processed/parents.jsonl` — parent recipe chunks  
- `data/processed/children.jsonl` — child chunks with `parent_id`

---

## Datasets

- **Recipe1M+:** http://pic2recipe.csail.mit.edu/ — place `layer1.json` under `data/raw/` if you use that loader.  
- **RecipeNLG:** https://huggingface.co/datasets/mbien/recipe_nlg  
- **Epicurious (bundle used for Stage 1):** place extracted `full_format_recipes.json` in `recipe_dataset/` (folder is `.gitignore`d).  
- **USDA FDC:** https://fdc.nal.usda.gov/api-key-signup.html → set `USDA_FDC_API_KEY` in `.env` for future enrichment steps.

---

## Next stages (roadmap)

1. **Embeddings & indexing** — dense + sparse encoders → Qdrant.  
2. **Retrieval** — hybrid query + reranking.  
3. **Generation** — SmolLM orchestration over retrieved parents.
