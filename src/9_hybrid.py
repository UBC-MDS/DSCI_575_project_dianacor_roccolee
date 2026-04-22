import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
from utils import build_hybrid_retriever, build_llm_model, run_chain, hybrid_run_queries
from prompts import SYSTEM_PROMPT_1, SYSTEM_PROMPT_2, SYSTEM_PROMPT_3
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("Missing GROQ_API_KEY in .env file")


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
    parser = argparse.ArgumentParser(description="Hybrid BM25 + Semantic RAG pipeline for Amazon product search.")
    parser.add_argument("--faiss-folder",
                        type=str,
                        default="data/retrievers/semantic_index",
                        help="Path to the folder containing the saved FAISS index.")
    parser.add_argument("--bm25-pkl",
                        type=str,
                        default="data/retrievers/bm25_index.pkl",
                        help="Path to the pickled BM25Retriever.")
    parser.add_argument("--bm25-weight",
                        type=float,
                        default=0.4,
                        help="Weight assigned to BM25 in the ensemble (semantic weight = 1 - bm25_weight).")
    parser.add_argument( "--embedding-model",
                        type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="HuggingFace embedding model used to load the FAISS index." \
                        " Must be the same as the model used to originally build the index.")
    parser.add_argument("--llm-model",
                        type=str,
                        default="qwen/qwen3-32b",
                        help="The Groq model for llm step.")
    parser.add_argument("--llm-model-2",
                        type=str,
                        default="openai/gpt-oss-120b",
                        help="The Groq model for llm comparison.")
    parser.add_argument("--k",
                        type=int,
                        default=5,
                        help="Number of documents each retriever fetches per query.")
    parser.add_argument("--query",
                        type=str,
                        default=None, #"1080p gaming monitor with high refresh rate and good color accuracy"
                        help="Single query test string. Ignored if --queries-csv is provided.")
    parser.add_argument("--queries-csv",
                        type=str,
                        default="results/queries.csv",
                        help="Path to CSV file to run though all example queries.")
    parser.add_argument("--output-csv",
                        type=str,
                        default="results/query_results_milestone2.csv",
                        help="Path to write the results CSV.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    hybrid_retriever = build_hybrid_retriever(
                        faiss_folder=args.faiss_folder,
                        bm25_pkl_path=args.bm25_pkl,
                        embedding_model=args.embedding_model,
                        bm25_weight=args.bm25_weight,
                        semantic_weight= 1 - args.bm25_weight,
                        k=args.k)
    
    if args.query: #if there's a single query provided
        
        using_model = args.llm_model
        llm = build_llm_model(local_call = False,
                                api_model = using_model)
        response = run_chain(
            query=args.query,
            # system_prompt=args.system_prompt, #custom prompt
            retriever=hybrid_retriever,
            llm_model=llm)
        
        print(f"Model: {using_model}... Query: {args.query}, \n Response returned: {response}")
    else:
        results_df = hybrid_run_queries(
            test_queries_path=args.queries_csv,
            hybrid_retriever=hybrid_retriever,
            system_prompt= SYSTEM_PROMPT_2, #custom prompt,
            model_1 = args.llm_model, 
            model_2 = args.llm_model_2)

        results_df.to_csv(args.output_csv, index=False)
        print(f"Results saved to {args.output_csv}")

        response = results_df