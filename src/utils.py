import requests
from tqdm import tqdm
from pathlib import Path
import string
import nltk
from nltk.corpus import stopwords
from langchain_core.documents import Document
import faiss
from langchain_community.vectorstores import FAISS
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter


import pandas as pd
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))

############################## For convert_parquet.py and create_documents.py script ##############################

# helper function from Claude to keep the columns being used in other scripts consistent 
# and easily editable in one place via a text file where milestone exploration notebook is
# (instead of hardcoding them in multiple scripts which could change) 

    #specific with default meta columns path 
def read_meta_txt_columns(filepath = "notebooks/meta_columns.txt"):
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]

    #specific with default review columns path 
def read_review_txt_columns(filepath = "notebooks/review_columns.txt"):
    with open(filepath, 'r') as f:
        return [line.strip() for line in f if line.strip()]
    

############################## For huggingface_datadownload.py script ##############################


def file_name_source_map(base_url, subset, meta, reviews):
    files = {}
    if reviews:
        files[f"{subset}.jsonl.gz"] = f"{base_url}/review_categories/{subset}.jsonl.gz" # to match the same file naming as if manually downloaded from website
    if meta:
        files[f"meta_{subset}.jsonl.gz"] = f"{base_url}/meta_categories/meta_{subset}.jsonl.gz" # to match the same file naming as if manually downloaded from website
    return files


def download_request(specific_url, output, filename):
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

############################## Query using semantic search ##############################
def semantic_search(docs, model, index, query, k=5):
    query_embedding = model.encode([query]).astype('float32')
    distances, indices = index.search(query_embedding, k)
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        results.append({
            'parent_asin': docs.iloc[idx]['parent_asin'],
            'product_title': docs.iloc[idx]['product_title'],
            'distance': dist
        })
    return results

############################## Query using BM25 search ##############################

def tokenize(document) -> list[str]:
    """
    Custom tokenized function for BM25.
    Does whitespace split, makes lowercase, remove punctuation and stopwords.
    """

    document = document.lower()
    document = document.translate(str.maketrans("", "", string.punctuation)) # removes punctuation 
    tokens = document.split()
    tokens = [t for t in tokens if t not in STOP_WORDS]
    return tokens


def bm25_search(query, bm25, doc_names, k = 5):
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    ranked_indices = sorted(range(len(scores)), 
                            key= lambda i: scores[i], 
                            reverse=True) # higher score is better
    top_k_indices  = ranked_indices[:k]
    return [{"product_title": doc_names[i], "distance": scores[i]} for i in top_k_indices]




############################## Langchain Utils Functions ##############################
def load_documents(parquet_path = "data/processed/product_documents.parquet", 
                    text_col= "document"):

    data = pd.read_parquet(parquet_path)

    documents = []
    for idx, row in data.iterrows():
        metacols = [col for col in data.columns if col != text_col]
        metadata = {col: row[col] for col in metacols}
        doc = Document(page_content=str(row[text_col]), 
                       metadata=metadata)
        documents.append(doc)

    print(f"[DONE]{len(documents)} documents loaded")
    return documents

def split_documents(documents,
                    chunk_size = 500,
                    chunk_overlap = 100):
    """
    Split documents using RecursiveCharacterTextSplitter — recursively tries to split at natural boundaries (paragraphs, newlines)
    rather than cutting arbitrarily.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"[DONE] Split into {len(split_docs)} chunks/sub-documents")
    return split_docs

def build_vectorstore(split_docs: list[Document],
                      model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> FAISS:
    """
    Embed chunks with a sentence-transformer and store in FAISS.
    Lecture uses the same model: sentence-transformers/all-MiniLM-L6-v2

    After this step:
      • Each chunk → vector representation
      • Similar meanings → vectors close in space  (lecture note)
    """
    # Lecture code: HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # Lecture code: FAISS.from_documents(split_docs, embeddings)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    print("FAISS vector store built — knowledge base is now searchable by meaning")
    return vectorstore


def langc_bm25_retriever(split_docs, 
                         k: int = 5):
    bm25_retriever = BM25Retriever.from_documents(split_docs)
    bm25_retriever.k = k
    return bm25_retriever

def langc_bm25_search(query, bm25_retriever, k = 5):
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
    return top_k

def return_top_results(top_k):
    results = []
    for doc, score in top_k:
            results.append({
        "product_title": doc.metadata.get("product_title"),
        "parent_asin": doc.metadata.get("parent_asin"),
        "score": float(score)
    })
    for r in results:
            print(f"Product ID: {r['parent_asin']}")
            print(r["product_title"])
            print(f"Score: {r['score']:.4f}")
            print("---")

    return results

############################## RAG pipeline Functions ##############################

def build_context(docs):
        return "\n\n".join(
            f"Product ASIN: {doc.metadata.get('parent_asin', 'N/A')}\n"
            f"Title: {doc.metadata.get('product_title', '')}\n"
            # f"Rating: {doc.metadata['rating']}/5]\n" # Need to add back as part of web app requirements
            for doc in docs
        )