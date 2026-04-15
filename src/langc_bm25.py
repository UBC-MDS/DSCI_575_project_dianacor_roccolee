from utils import load_documents, split_documents, langc_bm25_retriever
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

        query = args.query
        if hasattr(bm25_retriever, "preprocess_func"):
            query_tokens = bm25_retriever.preprocess_func(query)
        else:
            query_tokens = query.lower().split() #just as a fall back for tokenizing 

        bm25_scores = bm25_retriever.vectorizer.get_scores(query_tokens)
        bm25_docs = bm25_retriever.docs
        ranked = sorted(zip(bm25_docs, bm25_scores), 
                        key=lambda x: x[1], 
                        reverse=True)

        top_k = ranked[:k]

        results = []

        for doc, score in top_k:
            results.append({{
                "score": score,
                "text": doc.page_content,
                "metadata": doc.metadata
            })


if __name__ == "__main__":
    main()