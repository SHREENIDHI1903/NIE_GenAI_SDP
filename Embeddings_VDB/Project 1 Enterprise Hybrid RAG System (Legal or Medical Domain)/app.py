import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import sys
import os

# Add the project root to path so we can import src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.retriever import get_hybrid_retriever
from src.llm_pipeline import get_llm

st.set_page_config(page_title="Medical Enterprise Hybrid RAG", layout="wide", page_icon="🏥")
st.title("🏥 Enterprise Hybrid RAG System (Medical Domain)")
st.markdown("Ask questions about Hypertension or Type 2 Diabetes based on the ingested clinical guidelines.")

with st.sidebar:
    st.header("📋 System Info")
    st.info("This RAG system is trained on specific medical guidelines for Hypertension and Type 2 Diabetes.")
    
    st.header("💡 Example Questions")
    st.markdown("""
    - What is the initial drug for Diabetes?
    - How is Hypertension diagnosed?
    - What are the HbA1c goals for adults?
    - List first-line therapy for Hypertension.
    """)
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

@st.cache_resource
def load_resources():
    hybrid_retriever = get_hybrid_retriever()
    llm = get_llm()
    return hybrid_retriever, llm

try:
    retriever, llm = load_resources()
except Exception as e:
    st.error(f"Error loading resources: {e}")
    st.info("Ensure the vector stores exist. Please run 'python src/ingest.py' from terminal first.")
    st.stop()

# Define prompt configuration
prompt_template = """You are a professional medical assistant. Answer the user's question accurately using ONLY the context provided below.

Rules:
1. If the answer is present in the context, provide a concise and helpful response.
2. If the context does not contain the answer, reply exactly with: "I'm sorry, I don't have information on that in my current medical guidelines."
3. Do not use any external knowledge.

Context:
{context}

Question: {question}

Helpful Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Setup RAG chain
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)

# Chat User Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a medical question..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    with st.chat_message("assistant"):
        with st.spinner("Analyzing context and generating answer..."):
            try:
                # 1. Retrieve the relevant documents for transparency
                retrieved_docs = retriever.invoke(prompt)
                context_used = format_docs(retrieved_docs)
                
                # 2. Let the chain generate the exact answer
                raw_answer = rag_chain.invoke(prompt)
                # Some local models regurgitate the prompt. We isolate the output text.
                # 'raw_answer' contains everything after "Helpful Answer:" if generated nicely.
                st.markdown(raw_answer)
                
                # 3. Expose sources
                with st.expander("View Retrieved Context & Citations"):
                    st.info("This is the exact context pulled by combining FAISS and BM25 retrievers:")
                    st.write(context_used)
                
                st.session_state.messages.append({"role": "assistant", "content": raw_answer})
            
            except Exception as e:
                st.error(f"Error during execution: {e}")
