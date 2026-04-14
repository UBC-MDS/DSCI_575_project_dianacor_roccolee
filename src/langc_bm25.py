import pandas as pd
from utils import load_documents, split_documents
# from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
# from langchain_community.vectorstores import FAISS
# from langchain_classic.retrievers import EnsembleRetriever   # pip install langchain-classic
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough, RunnableParallel
# from transformers import pipeline


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

    print(f"{len(documents)} documents loaded")
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
    print(f"Split into {len(split_docs)} chunks")
    return split_docs

documents = load_documents()
split_docs = split_documents(documents)

def langc_bm25_retriever(split_docs, 
                         k: int = 5):
    bm25_retriever = BM25Retriever.from_documents(split_docs)
    bm25_retriever.k = k
    return bm25_retriever

= langc_bm25_retriever(split_docs)