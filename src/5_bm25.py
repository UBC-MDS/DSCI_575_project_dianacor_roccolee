from utils import load_documents, split_documents, build_bm25_retriever, bm25_search, print_top_results
import pickle
import argparse


def parse_args():
    '''To accept arguments directly via bash/terminal commands'''

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
    # For extra feature to test query after building index
    parser.add_argument("--query",
                        default=None,
                        help="Optional query to test after building")
    parser.add_argument("--k",
                        default=5,
                        type=int,
                        help="Number of results to return for query")
    return parser.parse_args()

def main():
    '''Main function to build a LangChain BM25 retriever from a parquet file of product documents
    and optionally tests it with a query.'''
    print("[STATUS] Starting BM25 index process")
    args = parse_args()

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
    
    with open(f"{args.output_dir}langc_bm25_index.pkl", "wb") as f:
        pickle.dump(bm25_retriever, f)
    print(f"[SAVED] LangChain BM25 retriever as .pkl file")

    with open(f"{args.output_dir}corpus.pkl", "wb") as f:
        pickle.dump(bm25_retriever.docs, f)
    print("[SAVED] Saved BM25 sub-docs corpus")
    print("[STATUS] Finished BM25 process.")
    
    # Extra feature if user also passed in a test query
    if args.query:

        print("\n")
        print(f"[EXTRA] Query also received: {args.query}")

        top_k = bm25_search(args.query, bm25_retriever, k=args.k)

        print(f"[RESULTS] Top {args.k} for query below:")
        print_top_results(top_k)


if __name__ == "__main__":
    main()