import qdrant_client
from nutriai.schemas import ParentChunk, ChildChunk


class CulinaryTools:
    def __init__(self, qdrant_host="localhost", qdrant_port=6333):
        self.client = qdrant_client.QdrantClient(qdrant_host, port=qdrant_port)
        self.collection_name = "recipe_chunks"

    def search_by_ingredients(self, ingredients: list[str], limit=3):
        """
        Finds recipes that contain the most matches from a list of ingredients.
        Uses the 'parent_id' logic from schemas.py.
        """
        results = []
        for ing in ingredients:
            # Search child chunks (kind="ingredient")
            hits = self.client.search(
                collection_name=self.collection_name,
                query_text=ing,
                query_filter={
                    "must": [{"key": "kind", "match": {"value": "ingredient"}}]
                },
                limit=5
            )
            results.extend(hits)

        # Group by parent_id and count hits
        parent_counts = {}
        for hit in results:
            pid = hit.payload['parent_id']
            parent_counts[pid] = parent_counts.get(pid, 0) + 1

        # Sort by most matches
        sorted_parents = sorted(parent_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_parents[:limit]

    def get_full_recipe(self, parent_id: str):
        """Fetches the ParentChunk text for the LLM to read."""
        parent = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[f"{parent_id}::parent"]
        )
        return parent[0].payload['text'] if parent else "Recipe not found."