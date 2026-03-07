# Technical Documentation - System Deep Dive

This document provides a detailed technical explanation of the components and logic within the Enterprise Hybrid RAG System.

---

## 1. Data Ingestion: `src/ingest.py`

The ingestion script is responsible for transforming raw medical guidelines into a format suitable for hybrid retrieval.

### Document Processing
- **Loading**: It uses `DirectoryLoader` with `TextLoader` to fetch all `.txt` files from the `data/` directory.
- **Chunking**: `RecursiveCharacterTextSplitter` divides documents into 500-character chunks with a 50-character overlap. 
  - *Why 500?* Small enough to stay focused on a single medical fact, large enough to provide context.
  - *Why overlap?* Ensures that facts split across chunks are preserved (contextual continuity).

### Storage
- **FAISS (Dense/Semantic)**: Generates 384-dimensional embeddings using `all-MiniLM-L6-v2`. These are stored in a FAISS index, allowing for multi-dimensional similarity search.
- **BM25 (Sparse/Keyword)**: Creates an index based on the BM25 algorithm (Best Match 25). It uses the exact frequency of words to find matches, which is critical for medical terms like "Metformin" or specific blood pressure values.

---

## 2. Hybrid Retrieval: `src/retriever.py`

This module implements the core "intelligence" that finds the right context for the LLM.

### The Ensemble Logic
- **FAISS Retriever**: Set to `k=4`. It retrieves the top 4 chunks based on semantic meaning.
- **BM25 Retriever**: Set to `k=2`. It retrieves the top 2 chunks based on exact keyword matches.
- **EnsembleRetriever**: Combines both results using **Reciprocal Rank Fusion (RRF)**.
  - **Weights**: `0.7` for FAISS and `0.3` for BM25.
  - *Rationale*: We prioritize semantic understanding (meaning) but still give significant weight to exact keyword matches (technical terms).

---

## 3. LLM Pipeline: `src/llm_pipeline.py`

The generation layer uses a local transformer model to answer questions based on the retrieved context.

### Model Selection
- **Model**: `google/flan-t5-base`. 
- **Type**: Text-to-Text-Transfer-Transformer (Seq2Seq). It is fine-tuned on instruction tasks, making it excellent at following prompt instructions like "Answer strictly based on context."

### Generation Parameters
- **`temperature=0.1`**: Keeps responses deterministic and factual.
- **`max_length=512`**: Ensures the model has enough "room" to explain complex medical criteria.
- **`repetition_penalty=1.1`**: Prevents the model from getting stuck in loops or repeating the prompt.

---

## 4. Main Application: `app.py`

The frontend and the RAG "Chain" are defined here using Streamlit.

### RAG Chain Construction (LCEL)
The system uses **LangChain Expression Language (LCEL)** to pipe data through the pipeline:
```python
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)
```
1.  **Context Loading**: The retriever fetches docs, and `format_docs` joins them into a single string.
2.  **Prompting**: The context and question are injected into the medical assistant prompt template.
3.  **Inference**: The LLM generates the final answer.
4.  **Parsing**: The `StrOutputParser` ensures we get a clean string output.

### UI Features
- **Caching**: `@st.cache_resource` is used on `load_resources()`. Since the LLM and retrievers are heavy, this ensures they are only loaded once when the app starts, not on every user interaction.
- **Grounding**: The prompt specifically instructs the model to say *"I'm sorry, I don't have information on that..."* if the answer isn't in the context, preventing medical misinformation.
- **Transparency**: The "View Retrieved Context" expander allows users (and developers) to see exactly what evidence the system used to generate its answer.

---

## 5. Directory Structure Recap

```text
├── app.py                # UI + Chain orchestration
├── src/
│   ├── ingest.py         # Data preparation & Vector DB creation
│   ├── retriever.py      # Hybrid (Dense + Sparse) search logic
│   └── llm_pipeline.py   # LLM initialization (FLAN-T5)
├── vector_store/         # On-disk FAISS and BM25 pkl files
└── requirements.txt      # Dependency list (LangChain, FAISS, Transformers)
```
