import faiss
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from shiny import App, ui, reactive, render
import numpy as np
from sentence_transformers import SentenceTransformer
import pyarrow.parquet as pq
from app_utils import *
import pickle
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

# Load data & models
table = pq.read_table("./data/processed/product_documents.parquet")
docs = table.to_pandas()

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("./data/processed/semantic_search_index.faiss")

with open("./data/processed/bm25_index.pkl", "rb") as f:
    bm25_data = pickle.load(f)

bm25 = bm25_data["bm25"]
doc_names = bm25_data["doc_names"]

generator = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    max_new_tokens=256,
    do_sample=False,
)
llm = HuggingFacePipeline(pipeline=generator)
print("[DONE] BUILDING GENERATOR")
hybrid_retriever = build_hybrid_retriever()

# -- Result card ---------------------------------------------------------------

def result_card(i, r):
    rating = r.get("average_rating") if hasattr(r, "get") else None
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
        ui.h4(r["product_title"], class_="product-title"),
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

        /* Shiny nav tabs */
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

        /* Search bar */
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

        /* Mode select */
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

        /* Result cards */
        .results-label { font-family: 'Syne', sans-serif; font-size: 0.75rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 16px; }
        .result-card { background: var(--surface); border: 1.5px solid var(--border); border-radius: var(--radius); padding: 22px 24px; margin-bottom: 14px; transition: border-color 0.18s; }
        .result-card:hover { border-color: var(--accent); }
        .card-header-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
        .rank-badge { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 0.78rem; color: var(--accent); }
        .rating-badge { font-family: 'Syne', sans-serif; font-weight: 600; font-size: 0.78rem; color: var(--accent2); margin-left: auto; }
        .product-title { font-family: 'Syne', sans-serif; font-size: 1.05rem; font-weight: 600; color: var(--text); line-height: 1.35; }

        /* RAG answer */
        .rag-answer-label { font-family: 'Syne', sans-serif; font-size: 0.75rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: var(--accent2); margin-bottom: 10px; }
        .rag-answer {
            background: var(--surface2); border: 1.5px solid var(--accent2);
            border-radius: var(--radius); padding: 22px 24px;
            font-size: 0.92rem; line-height: 1.7; color: var(--text);
        }

        /* Empty states */
        .empty-state { text-align: center; padding: 60px 0; color: var(--muted); font-size: 0.85rem; }
        .empty-state .icon { font-size: 2.5rem; margin-bottom: 12px; }
        """)
    ),
    ui.div(
        ui.div(
            ui.h1(ui.HTML('Product <span>Search</span>'), class_="app-title"),
            ui.p("Amazon product retrieval · Semantic & BM25 · RAG", class_="app-sub"),
            class_="app-header"
        ),
        ui.navset_tab(
            ui.nav_panel("Search Only",
                ui.input_select(
                    "search_mode", None,
                    choices={"semantic": "Semantic", "bm25": "BM25"},
                    selected="semantic",
                    width="200px"
                ),
                ui.div(
                    ui.input_text("search_query", None,
                                  placeholder="e.g. best gaming headset for under $30",
                                  width="100%"),
                    ui.input_action_button("search_btn", "Search", class_="search-btn"),
                    class_="search-wrap"
                ),
                ui.div(ui.output_ui("search_results")),
            ),
            ui.nav_panel("RAG Mode",
                ui.div(
                    ui.input_text("rag_query", None,
                                  placeholder="e.g. What's a good waterproof laptop?",
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
            return ui.div(ui.div("🔍", class_="icon"), ui.p("Enter a query and hit Search"), class_="empty-state")

        mode = input.search_mode()
        if mode == "semantic":
            hits = semantic_search(docs, model, index, query, k=5)
            label = "Semantic"
        else:
            hits = bm25_search(docs, bm25, query, k=5)
            label = "BM25"

        if not hits:
            return ui.div(ui.div("🤷", class_="icon"), ui.p("No results found."), class_="empty-state")

        return ui.div(
            ui.p(f"Top {len(hits)} results — {label} mode", class_="results-label"),
            *[result_card(i, r) for i, r in enumerate(hits)]
        )

    @output
    @render.ui
    @reactive.event(input.rag_btn)
    def rag_results():
        query = input.rag_query().strip()
        if not query:
            return ui.div(
                ui.div("💬", class_="icon"),
                ui.p("Ask a question to get a RAG-powered answer"),
                class_="empty-state"
            )
        try:
            answer = run_hybrid_chain(query, hybrid_retriever, llm, tokenizer)
            if not answer:
                return ui.div(ui.div("😕", class_="icon"), ui.p("No answer generated."), class_="empty-state")
            return ui.div(
                ui.p("Answer", class_="rag-answer-label"),
                ui.div(answer, class_="rag-answer"),
            )
        except Exception as e:
            return ui.div(ui.div("❌", class_="icon"), ui.p(f"Error: {str(e)}"), class_="empty-state")


app = App(app_ui, server)