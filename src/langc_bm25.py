from utils import *
import pickle
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Build a LangChain BM25 index from a parquet file.")
    parser.add_argument("--input",
                        default="data/processed/product_documents.parquet",
                        help="Path to input parquet file")
    parser.add_argument("--output-dir", 
                        default="data/processed/",  
                        help="Directory to save output files")
    parser.add_argument("--text-col",   
                        default="document")
    parser.add_argument("--query",
                        default=None,
                        help="Optional query to test after building")
    parser.add_argument("--k",
                        default=5,
                        type=int,
                        help="Number of results to return for query")
    return parser.parse_args()

def main():
    args = parse_args()

    # Document loading and splitting
    print("[STATUS] Loading and chunking/splitting docs")
    whole_docs = load_documents(parquet_path = args.input, text_col= args.text_col)
    split_docs = split_documents(whole_docs)

    #create Langchain BM25 retriever
    print("[STATUS] Building BM25 index")
    bm25_retriever = langc_bm25_retriever(split_docs)
    
    
    with open(f"{args.output_dir}langc_bm25_index.pkl", "wb") as f:
        pickle.dump(bm25_retriever, f)
    print(f"[DONE] LangChain BM25 retriever saved as langc_bm25_index.pkl to {args.output_dir}")

    if args.query:
        print("\n")
        print(f"[EXTRA] Query also received: {args.query}")
        top_k = langc_bm25_search(args.query, bm25_retriever, k=args.k)
        print(f"[RESULTS] Top {args.k} for query below:")
        return_top_results(top_k)

if __name__ == "__main__":
    main()