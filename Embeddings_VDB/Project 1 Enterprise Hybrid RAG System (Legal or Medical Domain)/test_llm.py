import os
import sys

# Add the project root to path so we can import src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.retriever import get_hybrid_retriever
from src.llm_pipeline import get_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

hybrid_retriever = get_hybrid_retriever()
llm = get_llm()

prompt_template = """Answer the question based strictly on the provided context below. 
If the context does not contain the answer, reply exactly with: "I don't know". Do not attempt to guess or use outside knowledge.

Question: {question}

Context: {context}

Helpful Answer:"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": hybrid_retriever | format_docs, "question": RunnablePassthrough()}
    | PROMPT
    | llm
    | StrOutputParser()
)

print(rag_chain.invoke("what is dolo650?"))
