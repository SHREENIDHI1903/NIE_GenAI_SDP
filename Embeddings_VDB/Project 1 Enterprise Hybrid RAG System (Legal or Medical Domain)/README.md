# Enterprise Hybrid RAG System (Medical Domain)

Welcome to the **Enterprise Hybrid RAG System**! This project is designed as an educational tool for students to understand how modern Retrieval-Augmented Generation (RAG) pipelines work in an enterprise setting, specifically applied to the medical domain (Hypertension and Diabetes).

## 🚀 Key Features

-   **Hybrid Retrieval**: Combines **Semantic Search** (Dense) and **Keyword Search** (Sparse) for maximum accuracy.
-   **Local LLM Execution**: Uses `google/flan-t5-base` to run entirely on your local CPU/GPU without needing OpenAI API keys.
-   **Local Embeddings**: Uses `all-MiniLM-L6-v2` via HuggingFace for fast, local vectorization.
-   **Streamlit UI**: A clean, interactive diagnostic interface.

---

## 🏗️ Architecture Overview

The system follows a classic RAG architecture with a "Hybrid" twist:

### 1. Data Ingestion (`src/ingest.py`)
-   **Loading**: Reads clinical guidelines from `data/*.txt`.
-   **Splitting**: Breaks long documents into smaller chunks (500 characters with 50-character overlap) for better retrieval.
-   **Vector Storage (FAISS)**: Creates high-dimensional embeddings for each chunk to capture "meaning".
-   **Sparse Storage (BM25)**: Indexing chunks based on keyword frequency to handle exact medical terms.

### 2. Retrieval Pipeline (`src/retriever.py`)
-   **FAISS (Dense Search)**: Finds chunks that are semantically similar to the question (even if keywords don't match exactly).
-   **BM25 (Sparse Search)**: Finds chunks that contain the exact medical codes or drug names mentioned in the query.
-   **Ensemble Retriever**: Merges results from both using **Reciprocal Rank Fusion (RRF)** to get the "best of both worlds".

### 3. LLM Generation (`src/llm_pipeline.py`)
-   Takes the retrieved context and the user's question.
-   Packages them into a structured prompt.
-   The `flan-t5` model generates an answer *strictly* based on the provided context.

---

## 💻 Installation & Setup

### 1. Prerequisites
-   Python 3.10 or higher.
-   A virtual environment is recommended.

### 2. Setup
Clone the repository and install dependencies:
```bash
# Create virtual environment
python -m venv venv

# Activate venv (Windows)
.\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Ingest Data
Before running the app, you must create the vector store:
```bash
python src/ingest.py
```

### 4. Run the App
```bash
streamlit run app.py
```
*Note: You can also use the `run.bat` file to automate these steps.*

---

## 🧠 Core Concepts for Students

### Why Hybrid RAG?
-   **Semantic Search (Dense)** is great for understanding intent (e.g., "how to treat high sugar" matches "diabetes management").
-   **Keyword Search (Sparse)** is essential for medical domain names (e.g., "Metformin", "130/80 mm Hg").
-   By combining them, we ensure that the system handles both conversational language and strict terminology.

### The "I Don't Know" Problem
We've implemented strict grounding. If the LLM cannot find the answer in the provided documents, it is instructed *not* to hallucinate and instead reply with:
> *"I'm sorry, I don't have information on that in my current medical guidelines."*

---

## 📂 Project Structure

```text
├── app.py                # Main Streamlit UI
├── src/
│   ├── ingest.py         # Document processing & indexing
│   ├── retriever.py      # Hybrid retrieval logic
│   └── llm_pipeline.py   # LLM initialization
├── data/                 # Raw clinical guideline text files
├── vector_store/         # Saved FAISS and BM25 indices
├── requirements.txt      # Project dependencies
└── run.bat               # Windows shortcut for running the system
```

---

## 🛠️ Debugging & Testing

We've provided scripts to help you see "under the hood":
-   `debug_rag.py`: Tests the entire pipeline with a single question and shows raw output.
-   `test_retrieval.py`: Shows you exactly which document chunks are being pulled from the database.
-   `verify_rag.py`: A consistency check for system accuracy.

Happy Coding! 🏥📚

---

## 🔮 Future Scope (Ideas for Students)

Want to take this project to the next level? Here are some advanced features you can implement:

1.  **Multi-Modal RAG**: Extend the ingestion to handle PDF tables, images of clinical charts (using OCR), and even structured medical data (CSV).
2.  **Contextual Re-ranking**: Use a **Cross-Encoder** (like `BGE-Reranker`) after the hybrid retrieval to re-rank the top chunks for even higher precision.
3.  **Conversation Memory**: Implement `langchain` memory (e.g., `ConversationBufferMemory`) so the system can answer follow-up questions like "What are its side effects?" after a question about "Metformin".
4.  **Advanced Evaluation**: Use tools like **RAGAS** or **TruLens** to quantitatively measure Faithfulness, Answer Relevance, and Context Precision.
5.  **Multi-Query Retrieval**: Implement a "Query Expansion" step where the LLM rewrites the user's question into 3 different versions to improve retrieval coverage.
6.  **Agentic RAG**: Turn the system into an **Agent** that can decide when to search the local database versus when to use a web-search tool (like Tavily) for the latest 2026 medical news.
