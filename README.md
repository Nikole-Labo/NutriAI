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
| Vector DB       | **Qdrant Cloud** — hybrid index + retrieval (`scripts/index_qdrant.py`, `nutriai.retrieval`) |
| Generator       | SmolLM-1.7B-Instruct (later stages) |

---

## Current status

This repository implements **Stage 1: data chunking**, **Stage 2: hybrid Qdrant indexing** (dense MiniLM + sparse BM25 + RRF), and **Stage 2b: macro + cross-encoder reranking** over parent recipes. The chat UI and full SmolLM agent loop are still optional follow-ups.

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
│   ├── embedding.py         # Dense MiniLM encode
│   ├── sparse_embedding.py  # FastEmbed Qdrant/bm25 sparse vectors
│   ├── qdrant_ids.py        # Stable UUIDs for Qdrant point ids
│   ├── qdrant_indexes.py    # Payload keyword indexes (Cloud strict mode)
│   ├── retrieval.py         # CulinaryTools — hybrid RRF search
│   ├── macro_parse.py       # Parse ## Nutrition from parent Markdown
│   ├── reranking.py         # Macro fit + cross-encoder rerank
│   ├── data/                # Loaders (sample, Recipe1M+, HF, Epicurious, …)
│   └── chunking/
├── scripts/
│   ├── run_chunking.py
│   ├── index_qdrant.py
│   └── smoke_retrieval.py
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

For **Stage 2 (Qdrant Cloud)**, edit `.env` and set **`QDRANT_URL`** and **`QDRANT_API_KEY`** (see Stage 2 **Step 2** below).

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

## Stage 2 — Qdrant Cloud (hybrid index, retrieval, rerank)

Stage 2 is documented for **Qdrant Cloud** (managed cluster). You run scripts from your PC; vectors and payloads live in the cloud. A **local Docker** Qdrant option is described at the **end** of this section only if you need it for development.

Each indexed point has:

- **`dense`** — `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, cosine distance).  
- **`sparse`** — FastEmbed `Qdrant/bm25` with Qdrant’s **IDF** modifier (lexical matches, e.g. ingredient tokens).  

Hybrid queries combine both branches with **RRF** (reciprocal rank fusion). Payloads carry `parent_id`, `kind`, `text`, and optional nutrient fields. **Macro + cross-encoder reranking** (Stage 2b) runs in Python after retrieval; see **Step 8**.

---

### Prerequisites (before Step 1)

1. **Stage 1 outputs exist** in `data/processed/`:

   - `parents.jsonl`  
   - `children.jsonl`  

   Example if you still need to generate them:

   ```powershell
   python scripts/run_chunking.py --source sample --out data/processed
   ```

2. **Environment ready** — [Setup](#setup): virtualenv created, `pip install -r requirements.txt` done. First runs download model weights (MiniLM, FastEmbed; cross-encoder when you rerank).

---

### Step 1 — Qdrant Cloud cluster

1. Go to [https://cloud.qdrant.io/](https://cloud.qdrant.io/) and sign in.  
2. **Create** a cluster and wait until its status is **green**.  
3. In the cluster UI, copy:

   - **Cluster / REST endpoint** (HTTPS URL). Use it **exactly** as shown (include **`:6333`** only if the console includes it in the URL).  
   - An **API key** that can manage collections and points.

---

### Step 2 — Configure environment variables (.env)

1. If you do not have a local env file yet:

   ```powershell
   copy .env.example .env
   ```

2. Edit **`NutriAI/.env`** (same directory as this `README.md`) and set:

   ```env
   QDRANT_URL=https://your-cluster-endpoint-from-console
   QDRANT_API_KEY=your_api_key_here
   ```

3. Do **not** commit `.env` to git.

`scripts/index_qdrant.py` and `scripts/smoke_retrieval.py` load `.env` from the project root automatically.

---

### Step 3 — (Optional) Quick connectivity check

From the repo root, venv activated:

```powershell
cd C:\Users\lazab\NutriAI
.\.venv\Scripts\Activate.ps1
python -c "from dotenv import load_dotenv; load_dotenv(); import os; from qdrant_client import QdrantClient; c=QdrantClient(url=os.environ['QDRANT_URL'], api_key=os.environ.get('QDRANT_API_KEY')); print([x.name for x in c.get_collections().collections])"
```

You should see a list of collection names (may be `[]` before first index). Fix URL/key if this fails.

---

### Step 4 — Index parents + children into Cloud

Creates collection **`recipe_chunks`**, ensures **keyword payload indexes** on `kind` and `parent_id` (needed for **Qdrant Cloud strict mode** when filtering), then upserts **dense + sparse** vectors for every parent and child line in your JSONL.

```powershell
python scripts/index_qdrant.py --recreate
```

| Flag | When to use |
| ----- | ------------- |
| **`--recreate`** | First index, or after any **schema** change (e.g. old dense-only collection). **Deletes** `recipe_chunks` if it exists, then rebuilds. |
| *(no flag)* | Re-upsert only: same deterministic point UUIDs get **overwritten** (e.g. after editing JSONL). |
| **`--parents-only`** / **`--children-only`** | Refresh just parents or just children. |

Wait until the script prints `Collection 'recipe_chunks' points_count=…`.

---

### Step 5 — Verify in Qdrant Cloud dashboard

1. **Clusters** → your cluster → **Collections** → **`recipe_chunks`**.  
2. **Points count** should equal parents + children indexed (sample pipeline: **69**).  
3. Open a **point**: payload fields present; **`dense`** length **384**; **`sparse`** present (short chunks have few non-zero terms — normal).

---

### Step 6 — Smoke-test hybrid retrieval (Cloud)

All commands assume repo root, venv on, `.env` pointing at Cloud.

```powershell
python scripts/smoke_retrieval.py
python scripts/smoke_retrieval.py --query "quick chicken dinner" --limit 3 --kind ingredient
python scripts/smoke_retrieval.py --ingredients "chicken,rice,tomato"
python scripts/smoke_retrieval.py --compare --query "steak"
python scripts/smoke_retrieval.py --parent-id sample-001
```

| Flag | Purpose |
| ----- | -------- |
| **`--kind ingredient`** | Search only **ingredient** child lines (stops full **parent** shards from dominating broad queries). |
| **`--ingredients`** | One search per ingredient (`kind=ingredient`), then aggregate hits by `parent_id`. |
| **`--compare`** | Print **hybrid**, **dense-only**, and **sparse-only** side by side (sparse may show **0** hits on tiny data if there is no BM25 overlap). |
| **`--parent-id`** | Print stitched **parent** Markdown for that recipe id. |

---

### Step 7 — Call retrieval from your own code

Scripts add `src` to `PYTHONPATH` automatically. For ad-hoc commands:

```powershell
$env:PYTHONPATH = "src"
python -c "from nutriai.retrieval import CulinaryTools; t=CulinaryTools(); print(t.search_children('chicken', limit=3, kind='ingredient'))"
```

API: **`nutriai.retrieval.CulinaryTools`** — `search_children`, `search_by_ingredients`, `get_full_recipe` (default **`mode="hybrid"`**; `mode="dense"` / `mode="sparse"` optional).

---

### Step 8 — Macro + cross-encoder rerank (Stage 2b)

Uses parsed **`## Nutrition`** from each parent Markdown plus **`cross-encoder/ms-marco-MiniLM-L-6-v2`**. Defaults **72%** macro / **28%** CE — edit `RERANK_MACRO_WEIGHT` and `RERANK_CE_WEIGHT` in `src/nutriai/config.py`.

```powershell
python scripts/smoke_retrieval.py --query "high protein dinner" --rerank --cal 450 --protein 35 --limit 5
python scripts/smoke_retrieval.py --ingredients "chicken,rice" --rerank --cal 500 --protein 30
```

Also: `--carbs`, `--fat`, `--candidate-hits` (child hit pool before rerank).

**Code:** `nutriai.reranking` — `MacroTargets`, `search_recipes_with_macro_rerank`, `search_ingredients_with_macro_rerank`, `rerank_parents`.

---

### Troubleshooting (Cloud)

| Symptom | Action |
| ------- | ------ |
| Auth / TLS / timeout | Recheck `QDRANT_URL` and `QDRANT_API_KEY`; cluster must be **green**. |
| `400` “Index required … `kind` / `parent_id`” | Run `python scripts/index_qdrant.py` once (indexes are created automatically), or run any command that constructs **`CulinaryTools()`** (it ensures keyword indexes on init). |
| Wrong or empty collection | `python scripts/index_qdrant.py --recreate`. |
| Sparse-only shows 0 hits | Expected on **very small** corpora without lexical overlap; use **hybrid** (default). |

---

### Optional: local Qdrant (Docker) instead of Cloud

For offline dev only: run Qdrant locally (e.g. Docker on `localhost:6333`), then in `.env` **remove or comment out `QDRANT_URL`** and set `QDRANT_HOST=localhost` and `QDRANT_PORT=6333` as in `.env.example`. Re-run **Step 4** and the smoke tests the same way.

---

## Datasets

- **Recipe1M+:** http://pic2recipe.csail.mit.edu/ — place `layer1.json` under `data/raw/` if you use that loader.  
- **RecipeNLG:** https://huggingface.co/datasets/mbien/recipe_nlg  
- **Epicurious (bundle used for Stage 1):** place extracted `full_format_recipes.json` in `recipe_dataset/` (folder is `.gitignore`d).  
- **USDA FDC:** https://fdc.nal.usda.gov/api-key-signup.html → set `USDA_FDC_API_KEY` in `.env` for future enrichment steps.

---

## Next stages (roadmap)

1. **Generation** — SmolLM orchestration over retrieved + reranked parents.  
2. **User inventory / feedback** — personalize retrieval and rerank weights over time.
