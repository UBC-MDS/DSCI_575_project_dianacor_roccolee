# ========================================== IMPORTS ==========================================
############################## Core Imports  ##############################

import pickle  # used in: build_hybrid_retriever
from pathlib import Path  # used in: download_request
import requests  # used in: download_request
import pandas as pd  # used in: load_documents, hybrid_run_queries
from tqdm import tqdm  # used in: download_request

############################## LangChain core objects  ##############################

from langchain_core.documents import Document  # used in: load_documents, build_context
from langchain_core.prompts import ChatPromptTemplate  # used in: run_chain
from langchain_core.runnables import RunnablePassthrough, RunnableLambda  # used in: run_chain
from langchain_core.output_parsers import StrOutputParser  # used in: run_chain

############################## LangChain retrieval / vector stores ##############################

import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter  # used in: split_documents
from langchain_community.vectorstores import FAISS  # used in: build_vect_retriever
from langchain_community.embeddings import HuggingFaceEmbeddings  # used in: build_vect_retriever
from langchain_community.retrievers import BM25Retriever  # used in: build_bm25_retriever
from langchain_classic.retrievers import EnsembleRetriever  # used in: build_hybrid_retriever

############################## LLM integrations ##############################

from langchain_groq import ChatGroq  # used in: build_llm_model

from langchain_huggingface import HuggingFacePipeline  # used in: build_llm_model
from transformers import pipeline  # used in: build_llm_model

# ========================================== FUNCTIONS ==========================================
############################## CONFIG HELPERS (column readers) ##############################

def read_meta_txt_columns(filepath = "notebooks/meta_columns.txt"):
    """
    Read metadata column names from a text file.

    Each line in the file represents one column name. This centralizes schema
    configuration so downstream scripts (parquet conversion, document creation,
    indexing) stay consistent without hardcoding column lists in multiple places.

    Args:
        filepath (str): Path to metadata column definition file.

    Returns:
        list[str]: Clean list of column names.
    """
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def read_review_txt_columns(filepath = "notebooks/review_columns.txt"):
    """
    Read review column names from a text file.

    Used to standardize feature selection for review datasets across
    preprocessing and document creation scripts.

    Args:
        filepath (str): Path to review column definition file.

    Returns:
        list[str]: Clean list of column names.
    """
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]
    

############################## DATA DOWNLOAD UTILITIES  ##############################
# For huggingface_datadownload.py script

def file_name_source_map(base_url, subset, meta, reviews):
    """
    Build a mapping of local filenames to remote dataset URLs.

    Ensures downloaded dataset structure matches expected naming conventions
    used in manual downloads.

    Args:
        base_url (str): Root dataset URL.
        subset (str): Dataset subset/category.
        meta (bool): Whether to include metadata files.
        reviews (bool): Whether to include review files.

    Returns:
        dict[str, str]: Mapping of filename -> download URL.
    """
    files = {}
    if reviews:
        files[f"{subset}.jsonl.gz"] = f"{base_url}/review_categories/{subset}.jsonl.gz" # to match the same file naming as if manually downloaded from website
    if meta:
        files[f"meta_{subset}.jsonl.gz"] = f"{base_url}/meta_categories/meta_{subset}.jsonl.gz" # to match the same file naming as if manually downloaded from website
    return files


def download_request(specific_url, output, filename):
    """
    Stream-download a file with progress bar and skip if already exists.

    This avoids redundant downloads and supports large dataset retrieval
    via chunked streaming.

    Args:
        specific_url (str): Remote file URL.
        output (str | Path): Output directory.
        filename (str): Name of saved file.
    """
    fullpath = Path(output) / filename

    if fullpath.exists(): # prevent from a taxing re-download
        print(f"Already exists, skipped: {filename}")
        return

    print(f"Downloading: {filename}")
    request = requests.get(specific_url, stream=True) #
    request.raise_for_status()

    # Following code below that handles request-downloads is from Claude as it's a nice-to-have and out of scope of assignment
    total = int(request.headers.get("content-length", 0))
    with open(fullpath, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024, desc=fullpath.name
    ) as bar:
        for chunk in request.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
            f.write(chunk)
            bar.update(len(chunk))

    print(f"Saved: {filename} in {output}")

############################## DOCUMENT LOADING & SPLITTING ##############################

def load_documents(parquet_path = "data/processed/product_documents.parquet", 
                    text_col= "document"):
    """
    Load a parquet dataset and convert rows into LangChain Documents.

    Each row becomes a Document where:
    - page_content = main text field
    - metadata = all other columns

    Args:
        parquet_path (str): Path to parquet dataset.
        text_col (str): Column containing main document text.

    Returns:
        list[Document]: LangChain documents for downstream retrieval.
    """
    data = pd.read_parquet(parquet_path)

    documents = []
    for idx, row in data.iterrows():
        metacols = [col for col in data.columns if col != text_col]
        metadata = {col: row[col] for col in metacols}
        doc = Document(page_content=str(row[text_col]), 
                       metadata=metadata)
        documents.append(doc)

    print(f"[DONE] {len(documents)} documents loaded")
    return documents

def split_documents(documents,
                    chunk_size = 500,
                    chunk_overlap = 100):
    """
    Split documents into smaller semantic chunks.

    Uses recursive splitting to respect natural text boundaries such as:
    paragraphs into -> sentences into -> words.

    Args:
        documents (list[Document]): Input documents.
        chunk_size (int): Max chunk size.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        list[Document]: Chunked documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"[DONE] Split into {len(split_docs)} chunks/sub-documents")
    return split_docs

############################## BM25 RETRIEVAL ##############################

def build_bm25_retriever(split_docs, 
                         k: int = 5):
    """
    Build a BM25 keyword-based retriever which is useful for 
    lexical matching (exact terms overlapping, like TF-IDF but better).

    Args:
        split_docs (list[Document]): Input documents.
        k (int): Top-k retrieval size.

    Returns:
        BM25Retriever: Configured retriever.
    """
    bm25_retriever = BM25Retriever.from_documents(split_docs)
    bm25_retriever.k = k
    return bm25_retriever

def bm25_search(query, bm25_retriever, k = 5):
    """
    Search using BM25 retriever and return top-k documents.

    Args:
        query (str): Search query.
        bm25_retriever (BM25Retriever): Trained retriever.
        k (int): Number of results.

    Returns:
        list[tuple[Document, float]]: Ranked results.
    """
    #tokenize the query safeguard
    if hasattr(bm25_retriever, "preprocess_func"):
            query_tokens = bm25_retriever.preprocess_func(query)
    else:
        query_tokens = query.lower().split() #just as a fall back for tokenizing 

    
    bm25_scores = bm25_retriever.vectorizer.get_scores(query_tokens)
    bm25_docs = bm25_retriever.docs

    ranked = sorted(zip(bm25_docs, bm25_scores), 
                    key=lambda x: x[1], 
                    reverse=True)

    top_k_docs = ranked[:k]
    return top_k_docs

def print_top_results(top_k_docs):
    """
    Helper function to print BM25 results in a nice readable format.

    Args:
        top_k_docs (list[tuple[Document, float]]): Top-k documents and their scores.

    Returns:
        list[dict]: Formatted results.
    
    Side effect: prints results to console (intended for reviewing).
    """
    results = []
    for doc, score in top_k_docs:
            results.append({
        "parent_asin": doc.metadata.get("parent_asin"),
        "product_title": doc.metadata.get("product_title"),
        "average_rating": doc.metadata.get("average_rating"),
        "score": float(score)
    })
    for r in results:
            print(f"Product ID: {r['parent_asin']}")
            print(f"Product Title: {r['product_title']}")
            print(f"Rating: {r['average_rating']}")
            print(f"Score: {r['score']:.4f}")
            print("---")
    return results

############################## SEMANTIC RETRIEVAL ##############################

def build_vect_retriever(faiss_folder = "data/retrievers/semantic_index",
                        model= "sentence-transformers/all-MiniLM-L6-v2", 
                        k=5):
    """
    Load a FAISS vectorstore from disk and return a retriever that can be used in the RAG chain.

    Args:
        faiss_folder (str): Path to saved FAISS index.
        model (str): Embedding model name.
        k (int): Top-k retrieval size.

    Returns:
        Retriever: LangChain retriever wrapper.
    """
    embeddings = HuggingFaceEmbeddings(model_name= model)
    vectorstore = FAISS.load_local(faiss_folder, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever

############################## HYBRID RETRIEVER ##############################

def build_hybrid_retriever(
    faiss_folder="data/retrievers/semantic_index",
    bm25_pkl_path="data/processed/bm25_retriever.pkl",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    bm25_weight=0.4,
    semantic_weight=0.6,
    k=5,
):
    """
    Create a hybrid retriever combining BM25 + FAISS semantic search.
    Args:
        faiss_folder (str): Path to FAISS index.
        bm25_pkl_path (str): Path to pickled BM25 retriever.
        embedding_model (str): Model name for embeddings.
        bm25_weight (float): Weight for BM25 in ensemble.
        semantic_weight (float): Weight for semantic retriever in ensemble.
        k (int): Number of documents each retriever returns.
    Returns:
        EnsembleRetriever: Weighted hybrid retriever.
    """
    #semantic load
    semantic_retriever = build_vect_retriever(faiss_folder = faiss_folder,
                                              model=embedding_model,
                                              k=k)

    #bm25 load
    with open(bm25_pkl_path, "rb") as f:
        bm25_retriever = pickle.load(f)
    bm25_retriever.k = k

    #hybrid ensemble
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, semantic_retriever],
        weights=[bm25_weight, semantic_weight])

    return hybrid_retriever

def hybrid_run_queries(test_queries_path, hybrid_retriever, system_prompt, 
                       model_1 = "qwen/qwen3-32b", 
                       model_2 = "openai/gpt-oss-120b"):
    """
    Iterate over every query/model combination to
    test multiple test queries across given LLMs to evaluate.
    Args:
        test_queries_path (str): Path to CSV of test queries.
        hybrid_retriever (EnsembleRetriever): The combined retriever to use.
        system_prompt (str): Prompt instructions for the LLM.
    Returns:
        pd.DataFrame: Query-response comparison table.
    """
    
    test_queries = pd.read_csv(test_queries_path)
    results = []
    models = {"model_1": ChatGroq(model=model_1),
            "model_2": ChatGroq(model=model_2)}

    for q in test_queries["queries"]:
        model_1_response = run_chain(
            query=q,
            retriever=hybrid_retriever,
            llm_model = models["model_1"],
            system_prompt = system_prompt)
        model_1_answer = model_1_response.split("</think>", 1)[-1].strip()

        model_2_response = run_chain(
            query=q,
            retriever=hybrid_retriever,
            llm_model = models["model_2"],
            system_prompt = system_prompt)
        model_2_answer = model_2_response.split("</think>", 1)[-1].strip()
        results.append({"query": q, f"{model_1}\'s response": model_1_answer, f"{model_2}\'s response": model_2_answer})

    return pd.DataFrame(results)
############################## RAG + LLM UTILITIES ##############################

def build_prompt(system_prompt, query, context):
    """
    Construct a full prompt for LLM inference.
    Combines system instructions, retrieved context, and user query
    and is passed into the ChatPromptTemplate in run_chain.
    Args:
        system_prompt (str): Instructions for the LLM.
        query (str): User's original question.
        context (str): Retrieved information to ground the answer.
    Returns:
        str: Final prompt string.
    """
    return f"""{system_prompt}

context:
{context}

question:
{query}

Answer based on the Amazon datasets from the context provided: """

def build_context(docs):
    """
    Convert retrieved documents into a single LLM-ready context string.
    Args:
        docs (list[Document]): Retrieved documents.
    Returns:
        str: Formatted context block.
    """
    return "\n\n".join(
        f"Product ASIN: {doc.metadata.get('parent_asin', 'N/A')}\n"
        f"Title: {doc.metadata.get('product_title', 'N/A')}\n"
        f"Average Rating: {doc.metadata.get('average_rating', 'N/A')}\n"
        f"Content: {doc.page_content}\n"
        for doc in docs
    )

############################## RAG + LLM PIPELINE ##############################

def build_llm_model(local_call = False, local_model = "Qwen/Qwen2.5-1.5B", max_tokens = 256, api_model = "qwen/qwen3-32b"):
   """
    Initialize either a local HF model or Groq-hosted LLM.

    Args:
        local_call (bool): Use local model if True.
        local_model (str): HuggingFace model name if local model = True.
        api_model (str): Remote API model name if local model = False.

    Returns:
        LLM: LangChain-compatible LLM wrapper.
    """
   if local_call:
        generator = pipeline(
            "text-generation",
            model=local_model,
            max_new_tokens=max_tokens,
            do_sample=True,
        )
        return HuggingFacePipeline(pipeline=generator)
   else:
        return ChatGroq(model=api_model)

def run_chain(
    query,
    retriever= None,
    llm_model = None,
    system_prompt="""You are a helpful Amazon shopping assistant.
        Answer the question using ONLY the following context (which contains real product reviews + metadata).
        Always cite the product ASIN when possible. If the answer isn't in the context, say so."""
        ):
    """
    Run a single query through the full RAG (Retrieval-Augmented Generation) pipeline
    and provides the final LLM output/response.

    Steps:
    1. Retrieve relevant documents
    2. Build context
    3. Format prompt
    4. Generate response via LLM

    Returns:
        str: Final model output.
    """
    # docs = retriever.invoke(query)
    # context = build_context(docs)
    # text_prompt = build_prompt(system_prompt, query, context)
    # full_prompt = ChatPromptTemplate.from_template(text_prompt)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "context:\n{context}\n\nquestion:\n{query}")
    ])

    rag_chain = (
        {
            "context": retriever | RunnableLambda(build_context),
            "query": RunnablePassthrough(),
        }
        | prompt
        | llm_model
        | StrOutputParser()
    )
    return rag_chain.invoke(query)


