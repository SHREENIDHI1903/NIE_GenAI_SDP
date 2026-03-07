import os
import sys

# Add the project root to path so we can import src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.retriever import get_hybrid_retriever
from src.llm_pipeline import get_llm

def debug_rag(question):
    print(f"\n--- Testing Question: {question} ---")
    retriever = get_hybrid_retriever()
    llm = get_llm()

    # 1. Test Retrieval
    docs = retriever.invoke(question)
    print(f"\nRetrieved {len(docs)} documents:")
    for i, doc in enumerate(docs):
        print(f"Doc {i+1} Content Snippet: {doc.page_content[:200]}...")
        # print(f"Doc {i+1} Metadata: {doc.metadata}")

    context = "\n\n".join(doc.page_content for doc in docs)

    # 2. Test Prompt
    prompt_template = """Answer the question based strictly on the provided context below. 
If the context does not contain the answer, reply exactly with: "I don't know". Do not attempt to guess or use outside knowledge.

Question: {question}

Context: {context}

Helpful Answer:"""
    
    prompt = prompt_template.format(question=question, context=context)

    # 3. Test LLM
    print("\nGenerating Answer...")
    response = llm.invoke(prompt)
    print(f"Raw Response: '{response}'")

if __name__ == "__main__":
    # Test valid questions from the data
    debug_rag("What is the initial pharmacologic agent for type 2 diabetes?")
    debug_rag("How is hypertension diagnosed?")
    # Test out of context
    debug_rag("What is dolo 650?")
