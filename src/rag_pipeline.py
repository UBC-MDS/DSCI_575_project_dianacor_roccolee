import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import pandas as pd
import pyarrow.parquet as pq
from langchain_core.documents import Document
from utils import *
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--query', 
                    default="what is a good gaming mouse to buy if I am left handed?",
                    type=str, 
                    help='Search query')
args = parser.parse_args()


def build_prompt(SYSTEM_PROMPT, query, context):
    return f"""{SYSTEM_PROMPT}

context:
{context}

question: 
{query}

Answer based on the Amazon datasets: """

def build_context(docs):
    return "\n\n".join(
        f"Product ASIN: {doc.metadata.get('parent_asin', 'N/A')}\n"
        f"Title: {doc.metadata.get('product_title', '')}\n"
        for doc in docs
    )

def build_vect_retriever(faiss_folder = "data/processed/langchain_semantic_index",
                        model= "sentence-transformers/all-MiniLM-L6-v2",
                      ):
    embeddings = HuggingFaceEmbeddings(
        model_name= model)

    vectorstore = FAISS.load_local(
            faiss_folder, 
            embeddings, allow_dangerous_deserialization=True
        )
    
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}) # Fetch 5 most similar documents

    return retriever

def run_chain(query = "what is a good gaming mouse to buy if I am left handed?",
            system_prompt = """
                            You are a helpful Amazon shopping assistant.
                            Answer the question using ONLY the following context (which contains real product reviews + metadata).
                            Always cite the product ASIN when possible. If the answer isn't in the context, say so.
                            """,
            retriever = build_vect_retriever()):

    docs = retriever.invoke(query)
    context = build_context(docs)

    generator = pipeline(
                    "text-generation",
                    model="Qwen/Qwen2.5-1.5B",
                    max_new_tokens=256,
                    do_sample=True,
                )
    llm = HuggingFacePipeline(pipeline=generator)
    text_prompt = build_prompt(system_prompt, query, context)
    full_prompt = ChatPromptTemplate.from_template(text_prompt)

    rag_chain = (
            {
                "context": retriever |  RunnableLambda(build_context),
                "query": RunnablePassthrough()
            }
            | full_prompt
            | llm
            | StrOutputParser()
        )
    
    response = rag_chain.invoke(query)
    response_cut = response.split("Assistant:", 1)[-1].strip()

    print(response)
    print(response_cut)


run_chain(query = args.query,
            system_prompt = """
                            You are a helpful Amazon shopping assistant.
                            Answer the question using ONLY the following context (which contains real product reviews + metadata).
                            Always cite the product ASIN when possible. If the answer isn't in the context, say so.
                            """,
            retriever = build_vect_retriever())