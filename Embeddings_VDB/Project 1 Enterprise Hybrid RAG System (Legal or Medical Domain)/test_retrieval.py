import os
import sys

# Add the project root to path so we can import src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.retriever import get_hybrid_retriever

def test_retrieval(question):
    print(f"\n--- Testing Retrieval for: {question} ---")
    retriever = get_hybrid_retriever()
    docs = retriever.invoke(question)
    print(f"Retrieved {len(docs)} documents.")
    for i, doc in enumerate(docs):
        print(f"Doc {i+1} Metadata: {doc.metadata}")
        print(f"Doc {i+1} Content: {doc.page_content[:200]}...\n")

if __name__ == "__main__":
    test_retrieval("What is the initial pharmacologic agent for type 2 diabetes?")
    test_retrieval("How is hypertension diagnosed?")
