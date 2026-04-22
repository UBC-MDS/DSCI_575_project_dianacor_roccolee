from utils import build_vect_retriever, bm25_search
import pandas as pd
import pickle
import argparse
from itertools import zip_longest

def parse_args():
    """
    Serves as a centralized entry point for defining and managing the command-line 
    arguments. These arguments will be parsed and be passed directly into functions 
    being called within the script. Defaults are set for all arguments. This means 
    the script can be run without any user-specified command-line arguments.

    Returns:
        argparse.ArgumentParser:
            Configured parser instance used to define and retrieve CLI arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--bm25-pkl",
                        type=str,
                        default="data/retrievers/bm25_index.pkl",
                        help="Path to the pickled BM25Retriever.")
    parser.add_argument("--faiss-folder",
                        type=str,
                        default="data/retrievers/semantic_index",
                        help="Path to the folder containing the saved FAISS index.")
    parser.add_argument( "--embedding-model",
                        type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="HuggingFace embedding model used to load the FAISS index." \
                        " Must be the same as the model used to originally build the index.")
    parser.add_argument("--queries-csv",
                        type=str,
                        default="results/queries.csv",
                        help="Path to CSV file to run though all example queries.")
    parser.add_argument("--k",
                        type=int,
                        default=5,
                        help="Number of documents each retriever fetches per query.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    test_queries = pd.read_csv("results/queries.csv")

    semantic_retriever = build_vect_retriever(faiss_folder = args.faiss_folder,
                            model= args.embedding_model, 
                            k=args.k)
    
    with open(args.bm25_pkl, "rb") as f:
        bm25_retriever = pickle.load(f)
    bm25_retriever.k = args.k

    master_retrieval = []
    for query in test_queries['queries']:

        bm25_results = bm25_search(query, bm25_retriever)
        sem_results = semantic_retriever.invoke(query)


        for sem_r, bm25_r in zip_longest(sem_results, bm25_results):

            # extra processing b/c  BM25 returns (Document, score) tuples
            if bm25_r:
                bm25_doc, bm25_score = bm25_r
                bm25_title = bm25_doc.metadata.get('product_title')
            else:
                bm25_title = None

            #append results
            master_retrieval.append({
                'query': query,
                'semantic_search_results': (sem_r.metadata.get('product_title') if sem_r else None),
                'bm25_search_results': bm25_title,
            })

    master_retrieval = pd.DataFrame(master_retrieval)
    master_retrieval.to_csv('results/query_results_milestone1.csv')