import os
import pickle
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever

VECTOR_STORE_DIR = "vector_store"
FAISS_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index")
BM25_PATH = os.path.join(VECTOR_STORE_DIR, "bm25_retriever.pkl")

def get_hybrid_retriever():
    if not os.path.exists(FAISS_PATH) or not os.path.exists(BM25_PATH):
        raise FileNotFoundError("Vector stores not found. Please run ingest.py first.")

    # Load FAISS
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    faiss_vstore = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    # Using similarity search for dense retrieval
    faiss_retriever = faiss_vstore.as_retriever(search_kwargs={"k": 4})

    # Load BM25
    with open(BM25_PATH, 'rb') as f:
        bm25_retriever = pickle.load(f)
    bm25_retriever.k = 2

    # Combine using Reciprocal Rank Fusion
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.3, 0.7] # Give slightly more weight to semantic search by default
    )
    
    return ensemble_retriever
