
import argparse
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from utils import load_documents, split_documents, print_top_results, build_vect_retriever


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
    parser = argparse.ArgumentParser(description='Semantic search for Amazon products')
    parser.add_argument("--input",
                        default="data/processed/product_documents.parquet",
                        help="Path to input parquet file")
    parser.add_argument("--output-dir", 
                        default="data/retrievers/",  
                        help="Directory to save output files")
    parser.add_argument('--embeddings_model',
                        type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",)
    parser.add_argument('--prevent-rewrite', 
                        type=bool, 
                        default=True, 
                        help='Tell the function whether indexes already exist')
    parser.add_argument('--query', 
                        type=str,
                        default=None, 
                        help='Search query')
    parser.add_argument('--k', 
                        type=int, 
                        default=5, 
                        help='Number of results to return')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    index_path = os.path.join(args.output_dir, "semantic_index")
    embeddings = HuggingFaceEmbeddings(model_name=args.embeddings_model)

    if not args.prevent_rewrite or not os.path.isdir(index_path):
        
        print("[STATUS] Loading and chunking/splitting docs")
        docs = load_documents(args.input)
        split_docs = split_documents(docs)
        print("[STATUS] Building semantic index")

        vectorstore = FAISS.from_documents(split_docs, embeddings)
        vectorstore.save_local(index_path)
        print(f"[SAVED] LangChain semantic retriever")
    
    else:
        print("[STATUS] Index already exists, loading index from file.")
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        print("[DONE] Index loaded")
    print("[STATUS] Finished semantic process.")

    if args.query:
            print("\n")
            print(f"[EXTRA] Query also received: {args.query}")
            top_k_results = vectorstore.similarity_search(args.query, k=args.k)

            print(f"[RESULTS] Top {args.k} for query below:")
            for rank, result in enumerate(top_k_results, start=1):
                print(f"Rank {rank}: {result.metadata.get('product_title')}")