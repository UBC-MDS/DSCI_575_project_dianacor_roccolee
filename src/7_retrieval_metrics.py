from utils import *
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
import pickle
from rank_bm25 import BM25Okapi


index = faiss.read_index("data/processed/semantic_search_index.faiss")

test_queries = pd.read_csv("results/queries.csv")
model = SentenceTransformer("all-MiniLM-L6-v2")
docs = pd.read_parquet("data/processed/product_documents.parquet")


with open("data/processed/bm25_index.pkl", "rb") as f:
    bm25_data = pickle.load(f)

bm25 = bm25_data["bm25"]
doc_names = bm25_data["doc_names"]
master_retrieval = []

for query in test_queries['queries']:
    bm25_results = bm25_search(query, bm25, doc_names, k=5)
    sem_results = semantic_search(docs, model, index, query, k=5)
    for sem_r, bm25_r in zip(sem_results, bm25_results):
        master_retrieval.append({
            'query': query,
            'semantic_search_results': sem_r['product_title'],
            'bm25_search_results': bm25_r['product_title']
        })

master_retrieval = pd.DataFrame(master_retrieval)
master_retrieval

master_retrieval.to_csv('results/query_results_milestone1.csv')