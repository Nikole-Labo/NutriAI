# LLM Architecture
**NutriAI. Malos Alexandru, Laza Bogdan, Labo Nikole**

## Dataset
We will use the Recipe1M+ Dataset for the recipes and the nutritional metadata from USDA FoodData Central

## Chunking Strategy
* **Markdown-Header Structural Chunking for Recipes**:
    * The system will split documents based on markdown headers (`## Ingredients`, `## Instructions`)
    * This prevents the “Instruction” from being cut mid sentences, and makes searches based on ingredients faster and with less noise
* **Parent-Child Retrieval**:
    * The system will split individual ingredients and nutritional facts in child chunks, these will be indexed for high speed search
    * When a child is matched, for example potatoes from Shepherd's Pie, it retrieves the parent recipe. This will work very well when the user gives multiple ingredients, getting all the children for all ingredients and seeing parent recipes that hit the most ingredient counts

### 3. Vector Database Choice
Because we are using a Parent-Child strategy, we believe Qdrant is the best fit because it allows for nested filtering and payloads (in order to store the parent recipe ID for an ingredient). It also handles Hybrid Search which combines traditional keyword matching for specific ingredients with Semantic Vector Search. This allows the system to understand the culinary context of a query, such as 'comfort food' or 'quick late-night snack'. This wil be done by measuring the mathematical proximity between the user’s intent and the recipe embeddings.

### 4. Model Choices: SmolLM-1.7B (Instruct)
We xan use this small model thanks to the precisness of the RAG pipeline (Parent-Child). SmolLM is incredibly fast and specialized in following instructions. It can also take the retrieved recipe and trim it to fit the user's specific calorie/macro count for that day.

### 5. RAG Strategies
* **Hybrid Search (Dense + Sparse)**: Use Sparse vectors (BM25) for exact ingredient matches (like Steak) and Dense vectors for "style" (like Quick Dinner or comfort food).
* **Reranking**: After the system pulls a big number of potential recipes, we will use a Cross-Encoder reranker to score them based specifically on the Macro-Nutrient fit. The recipe that closest hits the user's remaining calories for the day gets pushed to the #1 spot.
* **Recursive Character splitting (Fallback)**: We implement Recursive Character Splitting as a fallback to enforce strict token limits required by our SmolLM orchestrator. This hierarchical approach preserves semantic boundaries by prioritizing paragraph and line breaks, ensuring that even exceptionally long recipes are digestible without data loss or context-window overflow.
