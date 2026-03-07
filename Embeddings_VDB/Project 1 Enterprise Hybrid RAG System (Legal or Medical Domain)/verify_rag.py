import os
import sys

# Add the project root to path so we can import src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.retriever import get_hybrid_retriever
from src.llm_pipeline import get_llm

def verify_rag(question):
    print(f"\n--- Verifying Question: {question} ---")
    retriever = get_hybrid_retriever()
    llm = get_llm()

    docs = retriever.invoke(question)
    print(f"Retrieved {len(docs)} documents.")
    
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt_template = """You are a professional medical assistant. Answer the user's question accurately using ONLY the context provided below.

Rules:
1. If the answer is present in the context, provide a concise and helpful response.
2. If the context does not contain the answer, reply exactly with: "I'm sorry, I don't have information on that in my current medical guidelines."
3. Do not use any external knowledge.

Context:
{context}

Question: {question}

Helpful Answer:"""
    
    prompt = prompt_template.format(question=question, context=context)
    response = llm.invoke(prompt)
    print(f"Response: '{response}'")

if __name__ == "__main__":
    # Test valid questions
    verify_rag("What is the preferred initial pharmacologic agent for type 2 diabetes?")
    verify_rag("How is hypertension diagnosed?")
    # Test out of context
    verify_rag("What is dolo 650?")
    verify_rag("What do you know?")
