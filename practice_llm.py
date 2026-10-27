import random
from datetime import datetime


class Solution:
    def main(self):
        concepts=[
            "Python: Practice loops, functions, and classes.",
                "Machine Learning: Understand training and prediction.",
                "Deep Learning: Learn how neural networks work.",
                "LLMs: Understand tokens, prompts, and transformers.",
                "RAG: Learn document loading, chunking, embeddings, and retrieval.",
                "AI Agents: Understand tools, decisions, and actions.",
                "LangChain: Build applications using language models.",
                "LangGraph: Design workflows with state and conditional edges.",
                "Ollama: Run language models locally.",
                "Evaluation: Test accuracy, hallucinations, and reliability."
        ]

        today = datetime.now().strftime("%A, %d %B %Y")
        concept = random.choice(concepts)

        print("="*45)
        print("     AI ENGINEERING DAILY REMINDER")
        print("=" * 45)
        print(f"Today: {today}")
        print("\nREMEMBER YOUR MISSION:")
        print("Become a skilled AI Engineer through daily practice.")
        print("\nTODAY'S CONCEPT:")
        print(concept)
        print("\nYOUR WHITEBOARD CHALLENGE:")
        print("1. Write the concept in your own words.")
        print("2. Draw how it works.")
        print("3. Write a small Python example.")
        print("4. Explain it without looking at your notes.")
        print("\nConsistency beats cramming. Keep building!")
        print("=" * 45)


if __name__ == "__main__":
    Solution().main()

