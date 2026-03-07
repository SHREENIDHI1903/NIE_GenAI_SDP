import os
import pickle
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

DATA_DIR = "data"
VECTOR_STORE_DIR = "vector_store"
FAISS_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index")
BM25_PATH = os.path.join(VECTOR_STORE_DIR, "bm25_retriever.pkl")

def ingest_data():
    print("Loading documents from directory...")
    loader = DirectoryLoader(DATA_DIR, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()

    if not documents:
        print("No documents found in data directory.")
        return

    print(f"Loaded {len(documents)} documents. Splitting texts...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks.")

    print("Initializing embedding model (all-MiniLM-L6-v2)...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Creating FAISS Vector Store...")
    faiss_vstore = FAISS.from_documents(chunks, embeddings)
    faiss_vstore.save_local(FAISS_PATH)
    print(f"FAISS index saved to {FAISS_PATH}")

    print("Creating BM25 Retriever for Sparse Search...")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    
    with open(BM25_PATH, 'wb') as f:
        pickle.dump(bm25_retriever, f)
    print(f"BM25 index saved to {BM25_PATH}")

    print("Ingestion complete!")

if __name__ == "__main__":
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    ingest_data()
