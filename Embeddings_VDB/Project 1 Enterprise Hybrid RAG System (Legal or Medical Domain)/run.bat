@echo off
echo Starting Enterprise Hybrid RAG System...
if not exist "venv\Scripts\activate.bat" (
    echo Virtual environment not found. Please ensure it is created.
    exit /b 1
)

call venv\Scripts\activate.bat

if not exist "vector_store\faiss_index" (
    echo Vector store not found. Running ingestion first...
    python src\ingest.py
)

echo Starting Streamlit App...
streamlit run app.py
