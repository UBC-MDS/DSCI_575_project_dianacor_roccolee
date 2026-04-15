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

    results = []
    for i in top_k_indices:
        results.append({
            "product_title": doc_names[i],
            "distance": scores[i]
        })
    return results

def load_documents(parquet_path: str, 
                    text_col: str = "document"):

    data = pd.read_parquet(parquet_path)

    documents = []
    for idx, row in data.iterrows():
        metacols = [col for col in data.columns if col != text_col]
        metadata = {col: row[col] for col in metacols}
        doc = Document(page_content=str(row[text_col]), 
                       metadata=metadata)
        documents.append(doc)

    print(f"{len(documents)} documents loaded")
    return documents

def split_documents(documents: list[Document],
                    chunk_size: int = 500,
                    chunk_overlap: int = 100) -> list[Document]:
    """
    Split documents using RecursiveCharacterTextSplitter
    It recursively tries to split at natural boundaries (paragraphs, newlines)
    rather than cutting arbitrarily.

    chunk_size / chunk_overlap are a tradeoff:
      Too small → loses context; Too large → retrieval less precise
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split into {len(split_docs)} chunks")
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