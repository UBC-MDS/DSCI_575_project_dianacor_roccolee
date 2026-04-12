import argparse
import pickle
import pandas as pd
from rank_bm25 import BM25Okapi
from utils import *


def parse_args():
    parser = argparse.ArgumentParser(description="Build a BM25 index from a parquet file.")
    parser.add_argument("--input",
                        default="data/processed/product_documents.parquet",
                        help="Path to input parquet file")
    parser.add_argument("--output-dir", 
                        default="data/processed/",  
                        help="Directory to save output files")
    parser.add_argument("--text-col",   
                        default="document")
    parser.add_argument("--name-col",   
                        default="product_title")
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
    df = pd.read_parquet(args.input, columns=[args.text_col, args.name_col])

    print(f"Tokenizing {len(df)} documents and building corpus")
    tokenized_corpus = [tokenize(text) for text in df["document"]]
    doc_names = df["product_title"].tolist()

    print("Building BM25 index")
    bm25 = BM25Okapi(tokenized_corpus)

    print("Saving index")
    with open(f"{args.output_dir}bm25_index.pkl", "wb") as f:
        pickle.dump({"bm25": bm25, "doc_names": doc_names}, f)

    print("Saving corpus")
    with open(f"{args.output_dir}tokenized_corpus.pkl", "wb") as f:
        pickle.dump(tokenized_corpus, f)

    print(f"Everything saved to {args.output_dir}")

    if args.query:
        results = bm25_search(args.query, bm25, doc_names, k=args.k)
        for r in results:
            print(r["product_title"])
            print(f"Score: {r['distance']:.4f}")
            print("---")


if __name__ == "__main__":
    main()