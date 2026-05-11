import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# Import the existing tool class you just shared
from nutriai.retrieval import CulinaryTools


class NutriAgent:
    def __init__(self, model_id="HuggingFaceTB/SmolLM-1.7B-Instruct"):
        print(f"--- Loading SmolLM-1.7B to 1660 Super (4-bit) ---")
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

        # Initialize the existing retrieval tools
        # It will automatically use your .env or default localhost settings
        self.tools = CulinaryTools()

    def parse_action(self, llm_output: str):
        """Regex to find: Action: search_by_ingredients(['eggs', 'flour'])"""
        action_match = re.search(r"Action:\s*(\w+)\((.*)\)", llm_output)
        if action_match:
            return action_match.group(1), action_match.group(2)
        return None, None

    def run(self, user_query: str):
        # 1. Generate the Reasoning & Action
        prompt = f"User: {user_query}\nThought:"

        # We use a stop sequence so it doesn't hallucinate the observation
        raw_output = self.pipe(
            prompt,
            max_new_tokens=256,
            stop_sequence="Observation:",
            do_sample=True,
            temperature=0.2
        )[0]['generated_text']

        print(f"\n[AGENT THOUGHT]\n{raw_output}")

        # 2. Parse and Execute Tool
        name, args_str = self.parse_action(raw_output)

        if name == "search_by_ingredients":
            import ast
            # Clean the string arguments into a Python list
            ingredients = ast.literal_eval(args_str)

            # CALLING THE EXISTING RETRIEVAL
            print(f"--- Searching for: {ingredients} ---")
            hits = self.tools.search_by_ingredients(ingredients, limit=3)

            if not hits:
                return "Observation: No recipes found. Final Answer: I couldn't find anything matching those items."

            # Grab the first (best) match's full text
            best_recipe_id = hits[0][0]
            recipe_content = self.tools.get_full_recipe(best_recipe_id)

            # 3. Final Step: Give the content back to the LLM for a final summary
            return f"Observation: Found {len(hits)} matches. Here is the best one:\n{recipe_content}"

        return raw_output


if __name__ == "__main__":
    # Quick Test
    agent = NutriAgent()
    query = "I have chicken and spinach, what can I make?"
    result = agent.run(query)
    print(f"\n[FINAL RESPONSE]\n{result}")