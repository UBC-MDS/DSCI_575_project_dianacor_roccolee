import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import pandas as pd
import pyarrow.parquet as pq
from langchain_core.documents import Document
from utils import *
import argparse

parser = argparse.ArgumentParser(description='Semantic search for Amazon products')
parser.add_argument('--query', type=str, help='Search query')
parser.add_argument('--k', type=int, default=5, help='Number of results to return')
parser.add_argument('--index_exists', type=bool, default=False, help='Tell the function whether indexes already exist')
args = parser.parse_args()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

if args.index_exists:
    vectorstore = FAISS.load_local(
        "./data/processed/langchain_semantic_index", embeddings, allow_dangerous_deserialization=True
    )
    if args.query:
        results = vectorstore.similarity_search(args.query, k=3)
        for r in results:
            print(r.metadata["product_title"])
else:
    docs = load_documents("./data/processed/product_documents.parquet")
    split_docs = split_documents(docs)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    vectorstore.save_local("./data/processed/langchain_semantic_index")
    if args.query:
        results = vectorstore.similarity_search(args.query, k=3)
        for r in results:
            print(r)
            # print(r.metadata["product_title"])

