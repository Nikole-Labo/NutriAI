import os
from nutriai.retrieval import CulinaryTools
from nutriai.reranking import (
    search_ingredients_with_macro_rerank,
    MacroTargets
)

# Use the credentials you have
URL = "https://a75b8deb-f8d8-4e71-92de-038169e741b9.eu-central-1-0.aws.cloud.qdrant.io"
KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6M2VkMDY3OTktYzkyYS00MDA1LThlN2EtNDE2YTFhZTU2NzZjIn0.L-dJM96g0DUTyF1WwnTaHKzxqPE4eJCG4vc3NbF1OOQ"


def fast_test():
    print("--- Initializing Culinary Hybrid Pipeline ---")
    # Initialize the tools with keywords as required by your colleague's __init__
    tools = CulinaryTools(url=URL, api_key=KEY)

    # Define a Macro Target (e.g., a 500 kcal lunch with high protein)
    # This is what triggers the 'Grade 10' reranking logic
    my_goals = MacroTargets(
        target_calories=500,
        target_protein_g=40,
        target_carbs_g=50,
        target_fat_g=15
    )

    ingredients = ["chicken", "spinach", "lemon"]

    print(f"Testing Hybrid Search + Macro Rerank for: {ingredients}")

    try:
        # This calls the full colleague stack:
        # 1. Sparse search (BM25)
        # 2. Dense search (MiniLM)
        # 3. RRF Fusion
        # 4. Cross-Encoder Reranking
        # 5. Macro Fit scoring
        results, pids = search_ingredients_with_macro_rerank(
            tools,
            ingredients=ingredients,
            macro=my_goals,
            top_k=3
        )

        print(f"\nSuccessfully retrieved and reranked {len(results)} recipes:")
        for i, res in enumerate(results):
            print(f"{i + 1}. {res.title_hint} (Score: {res.final_score:.2f})")
            print(f"   Macros: {res.calories} kcal | Fit: {res.macro_fit:.2f}")
            print(f"   ID: {res.parent_id}\n")

    except Exception as e:
        print(f"Pipeline Failed: {e}")


if __name__ == "__main__":
    fast_test()