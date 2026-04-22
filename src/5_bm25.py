import os
import pickle
import argparse
from utils import load_documents, split_documents, build_bm25_retriever, bm25_search, print_top_results



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
    parser = argparse.ArgumentParser(description="Build a LangChain BM25 index from a parquet file.")
    parser.add_argument("--input",
                        default="data/processed/product_documents.parquet",
                        help="Path to input parquet file")
    parser.add_argument("--output-dir", 
                        default="data/retrievers/",  
                        help="Directory to save output files")
    parser.add_argument("--text-col",   
                        default="document")
    parser.add_argument("--chunk-size",   
                        default=1000,
                        type=int,
                        help="Size of each chunk")
    parser.add_argument("--chunk-overlap",   
                        default=100,
                        type=int,
                        help="Overlap between chunks")
    parser.add_argument("--prevent-rewrite",
                        type=bool, 
                        default=True,
                        help="Prevent rewriting the index if one exists")
    # For extra feature to test query after building index
    parser.add_argument("--query",
                        type=str,
                        default=None,
                        help="Optional query to test after building")
    parser.add_argument("--k",
                        default=5,
                        type=int,
                        help="Number of results to return for query")
    return parser.parse_args()

if __name__ == "__main__":
    print("[STATUS] Starting BM25 index process")
    
    args = parse_args()
    index_path = f"{args.output_dir}bm25_index.pkl"

    if not args.prevent_rewrite or not os.path.exists(index_path):
    # Document loading and splitting
        print("[STATUS] Loading and chunking/splitting docs")
        whole_docs = load_documents(parquet_path = args.input, 
                                    text_col= args.text_col)
        split_docs = split_documents(whole_docs, 
                                    args.chunk_size,
                                    args.chunk_overlap)

        #create Langchain BM25 retriever
        print("[STATUS] Building BM25 index")
        bm25_retriever = build_bm25_retriever(split_docs)
        
        with open(f"{index_path}", "wb") as f:
            pickle.dump(bm25_retriever, f)
        print(f"[SAVED] LangChain BM25 retriever as .pkl file")

        with open(f"{args.output_dir}corpus.pkl", "wb") as f:
            pickle.dump(bm25_retriever.docs, f)
        print("[SAVED] Saved BM25 sub-docs corpus")
    else: 
        print("[STATUS] Index already exists, loading index from file.")
        with open(index_path, "rb") as f:
            bm25_retriever = pickle.load(f)
        print("[DONE] Index loaded")
    print("[STATUS] Finished BM25 process.")
    
    # Extra feature if user also passed in a test query
    if args.query:

        print("\n")
        print(f"[EXTRA] Query also received: {args.query}")

        top_k = bm25_search(args.query, bm25_retriever, k=args.k)

        print(f"[RESULTS] Top {args.k} for query below:")
        print_top_results(top_k)