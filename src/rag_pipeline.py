import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import pandas as pd
import pyarrow.parquet as pq
from langchain_core.documents import Document
from utils import *

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
        "../data/processed/langchain_semantic_index", embeddings, allow_dangerous_deserialization=True
    )

# Convert vector store to a retriever
retriever = vectorstore.as_retriever(
search_type="similarity",
search_kwargs={"k": 5} # Fetch 5 most similar documents
)

# Use in a chain
query = "left-handed gaming mouse"
docs = retriever.invoke(query)