import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.nutriai.reranking import MacroTargets, search_ingredients_with_macro_rerank
from src.nutriai.retrieval import CulinaryTools
from difflib import get_close_matches


MASTER_INGREDIENTS = {
    "salmon", "chicken", "turkey", "beef", "steak", "pork", "tofu", "shrimp",
    "spinach", "broccoli", "peas", "rice", "beans", "eggs", "pasta", "potatoes",
    "lemon", "garlic", "onion", "tomato", "chickpeas", "lentils", "kale",
    # Add these:
    "lamb", "tuna", "cod", "shrimp", "bacon", "ham", "sausage",
    "carrot", "celery", "cucumber", "zucchini", "mushroom", "pepper",
    "cheese", "cream", "butter", "milk", "yogurt", "flour", "oats"
}

MACRO_WORDS = {
    'i', 'want', 'a', 'meal', 'dinner', 'lunch', 'breakfast', 'with', 'and',
    'at', 'least', 'make', 'it', 'be', 'max', 'minimum', 'maximum', 'grams',
    'gram', 'calories', 'calorie', 'kcal', 'protein', 'carbs', 'carb', 'fat',
    'fats', 'of', 'for', 'me', 'the', 'some', 'my', 'can', 'you', 'have',
    'high', 'low', 'around', 'about', 'under', 'over', 'please', 'healthy',
    'quick', 'easy', 'simple', 'diet', 'food', 'recipe', 'dish', 'cook'
}

class NutriAgent:
    def __init__(self, model_id="HuggingFaceTB/SmolLM-1.7B-Instruct"):
        print(f"--- Loading SmolLM-1.7B ---")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Optimized for your 6GB VRAM
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )

        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

        URL = "https://a75b8deb-f8d8-4e71-92de-038169e741b9.eu-central-1-0.aws.cloud.qdrant.io"
        KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwic3ViamVjdCI6ImFwaS1rZXk6M2VkMDY3OTktYzkyYS00MDA1LThlN2EtNDE2YTFhZTU2NzZjIn0.L-dJM96g0DUTyF1WwnTaHKzxqPE4eJCG4vc3NbF1OOQ"

        self.tools = CulinaryTools(url=URL, api_key=KEY)

    def parse_action(self, llm_output: str):
        """Regex to find: Action: search_by_ingredients(['eggs', 'flour'])"""
        action_match = re.search(r"Action:\s*(\w+)\((.*)\)", llm_output)
        if action_match:
            return action_match.group(1), action_match.group(2)
        return None, None

    def extract_ingredients_manually(self, query: str):
        query_clean = re.sub(r'[^\w\s]', '', query.lower())
        # Also strip standalone numbers
        tokens = [t for t in query_clean.split() if not t.isdigit()]

        found = []
        for word in tokens:
            if word in MASTER_INGREDIENTS:
                found.append(word)
            else:
                close = get_close_matches(word, MASTER_INGREDIENTS, n=1, cutoff=0.82)
                if close:
                    found.append(close[0])

        if not found:
            found = [w for w in tokens if w not in MACRO_WORDS and len(w) > 2]

        return list(set(found))

    def extract_macros(self, query: str) -> MacroTargets:
        targets = {}
        cal_match = re.search(r'(\d+)\s*(?:cal|kcal|calorie)', query, re.I)
        pro_match = re.search(r'(\d+)\s*(?:g(?:rams?)?\s+(?:of\s+)?)?protein', query, re.I)
        carb_match = re.search(r'(\d+)\s*(?:g|grams)?\s*carb', query, re.I)
        fat_match = re.search(r'(\d+)\s*(?:g|grams)?\s*fat', query, re.I)

        if cal_match: targets['target_calories'] = float(cal_match.group(1))
        if pro_match: targets['target_protein_g'] = float(pro_match.group(1))
        if carb_match: targets['target_carbs_g'] = float(carb_match.group(1))
        if fat_match: targets['target_fat_g'] = float(fat_match.group(1))

        # Fall back to reasonable defaults so reranker isn't flying blind
        if not targets:
            targets = {
                'target_calories': 600.0,
                'target_protein_g': 30.0,
                'target_carbs_g': 60.0,
                'target_fat_g': 20.0,
            }

        return MacroTargets(**targets)

    def run(self, user_query: str):
        user_macros = self.extract_macros(user_query)
        ingredients = self.extract_ingredients_manually(user_query)

        print(f"--- Sanitized Search: {ingredients} ---")

        from src.nutriai.reranking import search_ingredients_with_macro_rerank, rerank_parents

        # 1. This call uses your self.tools (Cloud Qdrant)
        _, pids = search_ingredients_with_macro_rerank(
            self.tools,
            ingredients=ingredients,
            macro=user_macros,
            top_k=20
        )

        # 2. Rerank based on the actual content in the Cloud
        results = rerank_parents(
            self.tools,
            pids,
            query=user_query,
            macro=user_macros,
        )[:5]

        if not results:
            return "I couldn't find anything in the cloud database."

        # 3. USE THE BEST MATCH DIRECTLY
        # Avoid the 'if matched' check which is too restrictive for real-world data
        best_match = results[0]
        recipe_text = self.tools.get_full_recipe(best_match.parent_id)
        recipe_text = self.tools.get_full_recipe(best_match.parent_id)
        print(f"--- Final pick: '{best_match.title_hint}' ---")

        messages = [
            {
                "role": "system",
                "content": "You are a helpful chef assistant. Present recipes clearly and concisely. Never invent ingredients or nutrition numbers not present in the recipe."
            },
            {
                "role": "user",
                "content": (
                    f"Read this recipe and present it to the user. Only use what's written here.\n\n"
                    f"=== RECIPE ===\n{recipe_text}\n=== END ===\n\n"
                    f"List the ingredients, summarize the steps, and state the calories and protein."
                )
            }
        ]

        formatted = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        final_res = self.pipe(
            formatted,
            max_new_tokens=400,
            repetition_penalty=1.2,
            do_sample=True,
            temperature=0.4
        )[0]['generated_text']

        return final_res[len(formatted):].strip()


if __name__ == "__main__":
    agent = NutriAgent()
    query = "I have chicken and spinach, what can I make?"
    result = agent.run(query)
    print(f"\n[FINAL RESPONSE]\n{result}")