# import faiss
# from langchain_community.vectorstores import FAISS
# from langchain_community.embeddings import HuggingFaceEmbeddings
# import pandas as pd
# import pyarrow.parquet as pq
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from transformers import pipeline
# from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
# from langchain_core.runnables import RunnablePassthrough, RunnableLambda
import argparse
import os
from utils import build_llm_model, build_vect_retriever, run_chain
from dotenv import load_dotenv, find_dotenv


def parse_args():
    '''To accept arguments directly via bash/terminal commands'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--faiss-folder",
                        type=str,
                        default="data/retrievers/semantic_index",
                        help="Path to the folder containing the saved FAISS index.")
    parser.add_argument( "--embedding-model",
                        type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="HuggingFace embedding model used to load the FAISS index." \
                        " Must be the same as the model used to originally build the index.")
    parser.add_argument("--local-model",
                        type=str,
                        default=False)
    parser.add_argument("--k",
                        type=int,
                        default=5,
                        help="Number of documents each retriever fetches per query.")
    parser.add_argument("--query",
                        type=str,
                        default="What is a good gaming mouse to buy if I am left handed?",
                        help="Single query test string.")
    return parser.parse_args()

def main():
    """main script function of parsing args, build the retriever, and run test query."""
    args = parse_args()

    semantic_retriever = build_vect_retriever(faiss_folder = args.faiss_folder,
                                              model=args.embedding_model,
                                              k=args.k)

    if args.local_model:

        llm = build_llm_model(local_call = True, local_model = "Qwen/Qwen2.5-1.5B",  max_tokens = 256)
        
        response = run_chain(args.query,
                retriever= semantic_retriever,
                llm_model = llm,
                # system_prompt= custom_prompt,
                )
        response_cut = response.split("Assistant:", 1)[-1].strip()
        print(response_cut)
    else:
        #api key check
        load_dotenv(find_dotenv())
        groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            raise ValueError("Missing GROQ_API_KEY in .env file")
        
        #process query with API model
        llm = build_llm_model(local_call = False, api_model = "qwen/qwen3-32b")
        response = run_chain(args.query,
                retriever= semantic_retriever,
                llm_model = llm,
                # system_prompt= custom_prompt,
                )
        print(response)

if __name__ == "__main__":
    main()