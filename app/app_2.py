# ========================================== IMPORTS ==========================================
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pickle
from shiny import App, ui, reactive, render
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv

from utils import (bm25_search,
    build_vect_retriever,
    build_hybrid_retriever,
    build_llm_model,
    run_chain,)

load_dotenv(find_dotenv())
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("Missing GROQ_API_KEY in .env file")

# ========================================== Global Variables ==========================================
RETRIEVER_FOLDER = "data/retrievers/"
BM25_FILE_PATH = os.path.join(RETRIEVER_FOLDER, "bm25_index.pkl")
SEM_INDEX_PATH = os.path.join(RETRIEVER_FOLDER, "semantic_index")
EMBEDDINGS_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

BM_WEIGHT = 0.4
SEMANTIC_WEIGHT = 0.6
K = 5

GROQ_MODEL = "qwen/qwen3-32b"
# ========================================== Retrievers loading ==========================================

# Semantic loading
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
vectorstore = FAISS.load_local(SEM_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
semantic_retriever = build_vect_retriever(faiss_folder=SEM_INDEX_PATH, model=EMBEDDINGS_MODEL, k=K)

# BM25 loading
with open(BM25_FILE_PATH, "rb") as f:
    bm25_retriever = pickle.load(f)

# Hybrid loading
hybrid_retriever = build_hybrid_retriever(
    faiss_folder=SEM_INDEX_PATH,
    bm25_pkl_path=BM25_FILE_PATH,
    embedding_model=EMBEDDINGS_MODEL,
    bm25_weight=BM_WEIGHT,
    semantic_weight=SEMANTIC_WEIGHT,
    k=K,
)
# LLM loading
llm = build_llm_model(api_model = GROQ_MODEL)

# ========================================== Shiny Code ==========================================
# -- Result card ---------------------------------------------------------------
# Hits are LangChain Documents: content in .page_content, fields in .metadata

def result_card(i, doc):
    meta = doc.metadata if hasattr(doc, "metadata") else {}
    rating = meta.get("average_rating")
    rating_badge = (
        ui.span(f"Rating: {rating:.1f}", class_="rating-badge")
        if rating is not None else ui.span()
    )
    return ui.div(
        ui.div(
            ui.span(f"#{i+1}", class_="rank-badge"),
            rating_badge,
            class_="card-header-row"
        ),
        ui.h4(meta.get("product_title", "Unknown Product"), class_="product-title"),
        class_="result-card"
    )


# -- UI ------------------------------------------------------------------------

app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.link(
            rel="stylesheet",
            href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;800&family=DM+Mono:wght@300;400&display=swap"
        ),
        ui.tags.style("""
        :root {
            --bg: #0e0f13;
            --surface: #16181f;
            --surface2: #1e2029;
            --border: #2a2d3a;
            --accent: #e8ff47;
            --accent2: #47c8ff;
            --text: #f0f0f0;
            --muted: #6b7080;
            --radius: 12px;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: var(--bg); color: var(--text); font-family: 'DM Mono', monospace; min-height: 100vh; }
        .app-wrap { max-width: 860px; margin: 0 auto; padding: 48px 24px; }
        .app-header { margin-bottom: 32px; }
        .app-title { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2.4rem; letter-spacing: -1px; line-height: 1; }
        .app-title span { color: var(--accent); }
        .app-sub { color: var(--muted); font-size: 0.82rem; margin-top: 8px; }

        .nav-tabs { border-bottom: 1.5px solid var(--border); margin-bottom: 28px; }
        .nav-tabs .nav-link {
            font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.85rem;
            color: var(--muted); background: none; border: none;
            padding: 10px 24px; border-radius: 0; transition: color 0.18s;
        }
        .nav-tabs .nav-link:hover { color: var(--text); }
        .nav-tabs .nav-link.active {
            color: var(--accent) !important; background: none !important;
            border-bottom: 2.5px solid var(--accent) !important;
        }
        .tab-content { padding-top: 8px; }

        .search-wrap { display: flex; gap: 10px; margin-bottom: 36px; margin-top: 20px; }
        .search-wrap input[type=text] {
            flex: 1; background: var(--surface); border: 1.5px solid var(--border);
            border-radius: var(--radius); color: var(--text); font-family: 'DM Mono', monospace;
            font-size: 0.95rem; padding: 14px 18px; outline: none; transition: border-color 0.18s;
        }
        .search-wrap input[type=text]:focus { border-color: var(--accent); }
        .search-wrap input[type=text]::placeholder { color: var(--muted); }
        .search-btn {
            background: var(--accent); border: none; border-radius: var(--radius);
            color: #0e0f13; font-family: 'Syne', sans-serif; font-weight: 700;
            font-size: 0.9rem; padding: 14px 28px; cursor: pointer; transition: opacity 0.18s;
        }
        .search-btn:hover { opacity: 0.85; }

        .selectize-input {
            background: var(--surface) !important; border: 1.5px solid var(--border) !important;
            color: var(--text) !important; font-family: 'DM Mono', monospace !important;
            border-radius: var(--radius) !important; margin-bottom: 20px;
        }
        .selectize-dropdown {
            background: var(--surface) !important; border: 1.5px solid var(--border) !important;
            color: var(--text) !important;
        }
        .selectize-dropdown .option:hover { background: var(--surface2) !important; }

        .results-label { font-family: 'Syne', sans-serif; font-size: 0.75rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 16px; }
        .result-card { background: var(--surface); border: 1.5px solid var(--border); border-radius: var(--radius); padding: 22px 24px; margin-bottom: 14px; transition: border-color 0.18s; }
        .result-card:hover { border-color: var(--accent); }
        .card-header-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
        .rank-badge { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 0.78rem; color: var(--accent); }
        .rating-badge { font-family: 'Syne', sans-serif; font-weight: 600; font-size: 0.78rem; color: var(--accent2); margin-left: auto; }
        .product-title { font-family: 'Syne', sans-serif; font-size: 1.05rem; font-weight: 600; color: var(--text); line-height: 1.35; }

        .rag-answer-label { font-family: 'Syne', sans-serif; font-size: 0.75rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: var(--accent2); margin-bottom: 10px; }
        .rag-answer {
            background: var(--surface2); border: 1.5px solid var(--accent2);
            border-radius: var(--radius); padding: 22px 24px;
            font-size: 0.92rem; line-height: 1.7; color: var(--text);
        }

        .empty-state { text-align: center; padding: 60px 0; color: var(--muted); font-size: 0.85rem; }
        .empty-state .icon { font-size: 2.5rem; margin-bottom: 12px; }
        """)
    ),
    ui.div(
        ui.div(
            ui.h1(ui.HTML('<span>Amazon Electronics</span> Product Search'), class_="app-title"),
            ui.p("Amazon product retrieval using 3 approaches: Semantic, BM25, and a Hybrid RAG", class_="app-sub"),
            class_="app-header"
        ),
        ui.navset_tab(
            ui.nav_panel("Pure Retrieval Search Only",
                ui.input_select(
                    "search_mode", None,
                    choices={"semantic": "Semantic", "bm25": "BM25"},
                    selected="semantic",
                    width="200px"
                ),
                ui.div(
                    ui.input_text("search_query", None,
                                  placeholder="e.g. Best gaming headset for under $30",
                                  width="100%"),
                    ui.input_action_button("search_btn", "Search", class_="search-btn"),
                    class_="search-wrap"
                ),
                ui.div(ui.output_ui("search_results")),
            ),
            ui.nav_panel("RAG Mode",
                ui.div(
                    ui.input_text("rag_query", None,
                                  placeholder="e.g. What's a good waterproof laptop look like?",
                                  width="100%"),
                    ui.input_action_button("rag_btn", "Ask", class_="search-btn"),
                    class_="search-wrap"
                ),
                ui.div(ui.output_ui("rag_results")),
            ),
            id="main_tabs"
        ),
        class_="app-wrap"
    )
)


# -- Server --------------------------------------------------------------------

def server(input, output, session):

    @output
    @render.ui
    @reactive.event(input.search_btn)
    def search_results():
        query = input.search_query().strip()
        if not query:
            return ui.div(ui.p("Enter a query and hit Search"), class_="empty-state")

        mode = input.search_mode()
        if mode == "semantic":
            hits = vectorstore.similarity_search(query, k=K)
            label = "Semantic"
        else:
            # bm25_search returns list[tuple[Document, float]] — extract just the docs
            raw = bm25_search(query, bm25_retriever, k=K)
            hits = [doc for doc, _score in raw]
            label = "BM25"

        if not hits:
            return ui.div(ui.p("No results found."), class_="empty-state")

        return ui.div(
            ui.p(f"Top {len(hits)} results — {label} mode", class_="results-label"),
            *[result_card(i, doc) for i, doc in enumerate(hits)]
        )

    @output
    @render.ui
    @reactive.event(input.rag_btn)
    def rag_results():
        query = input.rag_query().strip()
        if not query:
            return ui.div(
                ui.p("Ask a question to get a RAG-powered answer"),
                class_="empty-state"
            )
        try:
            answer = run_chain(query, retriever=hybrid_retriever, llm_model=llm)
            if not answer:
                return ui.div(ui.p("No answer generated."), class_="empty-state")
            return ui.div(
                ui.p("Answer", class_="rag-answer-label"),
                ui.div(answer, class_="rag-answer"),
            )
        except Exception as e:
            return ui.div(ui.p(f"Error: {str(e)}"), class_="empty-state")


app = App(app_ui, server)