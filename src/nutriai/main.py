"""CLI entry: SmolLM agent + Qdrant retrieval (configure Qdrant in project root ``.env``)."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure ``nutriai`` is importable when running: python -m nutriai.main
_SRC = Path(__file__).resolve().parent.parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from nutriai.finetuning.agent_core import NutriAgent


def main() -> None:
    print("--- NutriAI: SSVV Lab Edition ---")
    print("Initializing SmolLM-1.7B...")

    agent = NutriAgent()

    print("\n[SYSTEM ONLINE] Type 'exit' to quit.")

    while True:
        user_query = input("\nYou: ").strip()
        if user_query.lower() in ("exit", "quit"):
            break
        try:
            response = agent.run(user_query)
            print(f"\nNutriAI: {response}")
        except Exception as e:
            print(f"\n[Error]: {e}")


if __name__ == "__main__":
    main()
