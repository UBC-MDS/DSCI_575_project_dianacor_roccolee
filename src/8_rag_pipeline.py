import argparse
import os
from utils import build_llm_model, build_vect_retriever, run_chain
from dotenv import load_dotenv, find_dotenv


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
    parser.add_argument("--faiss-folder",
                        type=str,
                        default="data/retrievers/semantic_index",
                        help="Path to the folder containing the saved FAISS index.")
    parser.add_argument( "--embedding-model",
                        type=str,
                        default="sentence-transformers/all-MiniLM-L6-v2",
                        help="HuggingFace embedding model used to load the FAISS index." \
                        " Must be the same as the model used to originally build the index.")
    parser.add_argument("--local-model",
                        type=str,
                        default=False)
    parser.add_argument("--k",
                        type=int,
                        default=5,
                        help="Number of documents each retriever fetches per query.")
    parser.add_argument("--query",
                        type=str,
                        default="What is a good gaming mouse to buy if I am left handed?",
                        help="Single query test string.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    semantic_retriever = build_vect_retriever(faiss_folder = args.faiss_folder,
                                              model=args.embedding_model,
                                              k=args.k)

    if args.local_model:
        llm = build_llm_model(local_call = True, local_model = "Qwen/Qwen2.5-1.5B",  max_tokens = 256)
        response = run_chain(args.query,
                retriever= semantic_retriever,
                llm_model = llm,
                # system_prompt= custom_prompt,
                )
        response_cut = response.split("Assistant:", 1)[-1].strip()
        print(response_cut)
    else:
        #api key check
        load_dotenv(find_dotenv())
        groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            raise ValueError("Missing GROQ_API_KEY in .env file")
        
        #process query with API model
        llm = build_llm_model(local_call = False, api_model = "qwen/qwen3-32b")
        response = run_chain(args.query,
                retriever= semantic_retriever,
                llm_model = llm,
                # system_prompt= custom_prompt,
                )
        print(response)