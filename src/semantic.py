import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer
import pyarrow.parquet as pq
import numpy as np
import faiss
from utils import *

parser = argparse.ArgumentParser(description='Semantic search for Amazon products')
parser.add_argument('--query', type=str, help='Search query')
parser.add_argument('--k', type=int, default=5, help='Number of results to return')
args = parser.parse_args()

table = pq.read_table("data/processed/product_documents.parquet")
docs = table.to_pandas()

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = docs["document"].to_list()
embeddings = model.encode(sentences)

embeddings_array = np.array(embeddings).astype('float32')
dimension = embeddings_array.shape[1]

index = faiss.IndexFlatL2(dimension)
index.add(embeddings_array)

if args.query:
    
    results = semantic_search(docs, model, index, args.query, k=5)
    for r in results:
        print(r['product_title'])
        print(f"Distance: {r['distance']:.4f}")
        print("---")

faiss.write_index(index, "./data/processed/semantic_search_index.faiss")