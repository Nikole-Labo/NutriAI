import sys
import os
import torch

# 1. Setup Pathing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.nutriai.finetuning.agent_core import NutriAgent


def main():
    print("--- NutriAI: SSVV Lab Edition ---")
    print("Initializing SmolLM-1.7B on RTX 4070...")

    agent = NutriAgent()

    print("\n[SYSTEM ONLINE] Type 'exit' to quit.")

    while True:
        user_query = input("\nYou: ").strip()
        if user_query.lower() in ["exit", "quit"]:
            break
        try:
            response = agent.run(user_query)  # Use your existing run method
            print(f"\nNutriAI: {response}")
        except Exception as e:
            print(f"\n[Error]: {e}")


if __name__ == "__main__":
    main()