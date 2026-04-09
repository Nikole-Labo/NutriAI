# NutriAI: Nutrition Assistant

**Team Name:** AIChefs

**Team Members:**
- Laza Bogdan
- Labo Nikole
- Maloș Alexandru


## Description

NutriAI is an intelligent culinary assistant designed to seamlessly integrate recipe discovery, macronutrient tracking, and personalized meal planning into a single conversation. It moves beyond reactive tracking to proactively suggest macro-friendly meals based on a user's real-time ingredient inventory, previous consumption, and dynamic preferences (e.g., using perishable items, specific cravings).

### Goals

- Craft personalized recipes aligned with daily calorie and macronutrient targets based on biometric data (gender, height, weight, age) and fitness goals.
- Build persistent user profiles through a continuous feedback loop (rating recipes and ingredients) to refine flavor preferences and dietary restrictions.
- Address limitations of existing solutions (traditional recipe discovery, reactive tracking, rigid automated planning) by offering a seamless, conversation-driven integration that adapts in real-time.

### Architecture

The system utilizes an agentic culinary assistant architecture built on:

- **Custom RAG Pipeline:** Ingests the Recipe1M+ dataset, nutritional metadata (USDA FoodData Central), and user inventory into a vector database for dynamic retrieval and modification of recipes.
- **Model:** fine-tuned SmolLM-1.7B (Instruct), acting as the reasoning agent, utilizes a ReAct (Reason + Act) loop to orchestrate decision-making and personalize recipe suggestions.

#### Detailed Technical Details

- **Chunking Strategy:**
    - **Markdown-Header Structural Chunking for Recipes:** Splitting documents by headers (Ingredients, Instructions) ensures coherent context and faster ingredient searches.
    - **Parent-Child Retrieval:** Individual ingredients/nutritional facts are indexed as child chunks, linked to parent recipes, enabling fast retrieval and parent recipe matching based on multiple ingredient counts.
- **Vector Database:** Qdrant is selected for its support of nested filtering and payloads (storing parent recipe IDs for ingredients), as well as hybrid search (combining traditional keyword and semantic vector search) to understand culinary context.
- **RAG Strategies:**
    - **Hybrid Search (Dense + Sparse):** Combines exact ingredient matches with semantic searches for culinary styles/contexts.
    - **Reranking:** Utilizes a Cross-Encoder reranker to score and prioritize retrieved recipes based on remaining daily macro-nutrient fit.
    - **Recursive Character Splitting (Fallback):** Enforces strict token limits for the SmolLM orchestrator, preserving semantic boundaries for long recipes.
