# Why the Agent Uses Sample Data (And How to Fix It)

## Current Architecture

The NutriAI agent **is correctly designed to use Qdrant**, not sample.py locally. However, there's an important setup requirement that needs to be completed first.

## The Data Pipeline

There are **three required steps** to get real data into the agent:

### Step 1: Chunking Pipeline (MUST RUN FIRST)
**File:** `scripts/run_chunking.py`

This script reads recipes from a source and splits them into parent/child chunks, outputting JSONL files.

```powershell
# Default: uses built-in 5-recipe sample (found in src/nutriai/data/sample.py)
python scripts/run_chunking.py --source sample

# To use real recipe data:
python scripts/run_chunking.py --source hf --hf-name mbien/recipe_nlg --limit 1000
python scripts/run_chunking.py --source epicurious --path recipe_dataset/ --limit 2000
python scripts/run_chunking.py --source recipe1m --path data/raw/layer1.json --limit 1000
```

**Output:** Creates `data/processed/parents.jsonl` and `data/processed/children.jsonl`

### Step 2: Qdrant Indexing (MUST RUN SECOND)
**File:** `scripts/index_qdrant.py`

This script reads the JSONL files from Step 1 and indexes them into Qdrant (either cloud or local).

```powershell
# Index the processed data into Qdrant
python scripts/index_qdrant.py

# If upgrading schema, recreate the collection:
python scripts/index_qdrant.py --recreate
```

**Important:** If `data/processed/` is empty, this script will fail with: `Missing data/processed/parents.jsonl`

### Step 3: Agent Uses Qdrant
**File:** `src/nutriai/finetuning/agent_core.py`

The agent connects to Qdrant and searches for recipes. It does NOT use sample.py directly.

```python
self.tools = CulinaryTools(url=URL, api_key=KEY)
# Searches happen through self.tools.search_children()
# Data retrieval uses self.tools.get_full_recipe()
```

## Current Status

⚠️ **Your `data/processed/` directory is empty**, meaning:
- ❌ No JSONL files have been generated
- ❌ Qdrant cannot have been indexed
- ❌ The agent might be getting empty results or old data

### Why You Might Think It's Using sample.py

If the agent IS returning recipe data, one of these is true:

1. **Qdrant was indexed with sample data in the past**
   - Run Step 1 with `--source sample` and Step 2
   - Result: Qdrant contains the 5 sample recipes
   - Agent queries Qdrant and gets those 5 recipes back

2. **There's cached data in Qdrant from a previous session**
   - Qdrant is persistent (all cloud)
   - Even though local JSONL files are gone, Qdrant still has the data

## The Fix: Use Real Data

To use real recipe data instead of the sample recipes:

```powershell
# 1. Run chunking with your real data source
python scripts/run_chunking.py --source epicurious --path recipe_dataset/ --limit 5000

# 2. Recreate Qdrant collection and index new data
python scripts/index_qdrant.py --recreate

# 3. Run the agent - it will now query Qdrant for real recipes
python scripts/run_chunking.py --source sample  # Back to sample if needed
```

## Architecture Diagram

```
Real Recipe Data (Epicurious/Recipe1M+/HF)
    ↓
[run_chunking.py] → parents.jsonl, children.jsonl
    ↓
[index_qdrant.py] → Upload to Qdrant (Cloud)
    ↓
agent_core.py ← Uses CulinaryTools to query Qdrant
    ↓
SmolLM-1.7B generates response based on Qdrant results
```

**Note:** `sample.py` is only used:
- In tests (see `tests/test_chunking.py`)
- As the default source for `run_chunking.py` when `--source sample`
- NOT directly by the agent in production

