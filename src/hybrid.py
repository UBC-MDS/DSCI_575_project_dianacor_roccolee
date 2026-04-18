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
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

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
        f"Average Rating: {doc.metadata.get('average_rating', '')}\n"
        for doc in docs
    )

def build_hybrid_retriever(faiss_folder = "data/processed/langchain_semantic_index",
                        docs_path = "data/processed/product_documents.parquet",
                        model= "sentence-transformers/all-MiniLM-L6-v2"
                      ):
    embeddings = HuggingFaceEmbeddings(
        model_name= model)

    vectorstore = FAISS.load_local(
            faiss_folder, 
            embeddings, allow_dangerous_deserialization=True
        )
    
    docs = pd.read_parquet(docs_path)
    documents = [
        Document(
            page_content=row["document"],
            metadata={"product_title": row["product_title"], "parent_asin": row["parent_asin"], "average_rating": row["average_rating"]}
        )
        for _, row in docs.iterrows()
    ]

    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 5

    semantic_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5}) # Fetch 5 most similar documents

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[0.4, 0.6]  # Example: asigning 40% importance to BM25, 60% to Semantic Search
    )

    return hybrid_retriever

def run_hybrid_chain(query = "1080p gaming monitor",
            system_prompt = """
                            You are a helpful Amazon shopping assistant.
                            Answer the question using ONLY the following context (which contains real product reviews + metadata).
                            Always cite the product ASIN when possible. If the answer isn't in the context, say so.
                            """,
            hybrid_retriever = build_hybrid_retriever()):

    docs = hybrid_retriever.invoke(query)
    context = build_context(docs)

    generator = pipeline(
                    "text-generation",
                    model="Qwen/Qwen2.5-1.5B",
                    max_new_tokens=128,
                    do_sample=True,
                )
    llm = HuggingFacePipeline(pipeline=generator)
    text_prompt = build_prompt(system_prompt, query, context)
    full_prompt = ChatPromptTemplate.from_template(text_prompt)

    hybrid_rag_chain = (
            {
                "context": hybrid_retriever |  RunnableLambda(build_context),
                "query": RunnablePassthrough()
            }
            | full_prompt
            | llm
            | StrOutputParser()
        )
    
    
    response = hybrid_rag_chain.invoke(query)
    response_cut = response.split("Assistant:", 1)[-1].strip()

    return (response, response_cut)


def main():

    hybrid_retriever = build_hybrid_retriever()

    test_queries = pd.read_csv("results/queries.csv")
    results = []

    response, response_cut = run_hybrid_chain(query = "1080p gaming monitor",
                                        system_prompt = """
                                                        You are a helpful Amazon shopping assistant.
                                                        Answer the question using ONLY the following context (which contains real product reviews + metadata). 
                                                        Include the product's average rating as part of your reasoning for selecting a certain product, the higher the rating the better the product.
                                                        Always cite the product ASIN when possible. If the answer isn't in the context, say so.
                                                        """,
                                        hybrid_retriever = hybrid_retriever)
    print(response)

    # for q in test_queries['queries']:
    #     response, response_cut = run_hybrid_chain(query = q,
    #                                     system_prompt = """
    #                                                     You are a helpful Amazon shopping assistant.
    #                                                     Answer the question using ONLY the following context (which contains real product reviews + metadata). 
    #                                                     Include the product's average rating as part of your reasoning for selecting a certain product, the higher the rating the better the product.
    #                                                     Always cite the product ASIN when possible. If the answer isn't in the context, say so.
    #                                                     """,
    #                                     hybrid_retriever = hybrid_retriever)
    #     results.append({
    #         'query': q,
    #         'response': response_cut
    #     })
    # results_df = pd.DataFrame(results)
    # results_df.to_csv("results/m2_query_results.csv")
    # print(results_df)
    # return results_df

if __name__ == "__main__":
    main()
