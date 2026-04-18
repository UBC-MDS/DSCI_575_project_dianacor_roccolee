from shiny import App, ui, reactive, render
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pyarrow.parquet as pq
from utils import *
 
# Load data & models
table = pq.read_table("./data/processed/product_documents.parquet")
docs = table.to_pandas()
 
model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("./data/processed/semantic_search_index.faiss")

################ TO DO: ADD FIRST 200 CHARACTERS OF THE REVIEW TEXT TO EACH RESULT CARD

# Search function
 
def result_card(i, r):
    return ui.div(
        ui.div(
            ui.span(f"#{i+1}", class_="rank-badge"),
            class_="card-header-row"
        ),
        ui.h4(r["product_title"], class_="product-title"),
        class_="result-card"
    )
 
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
        .app-header { margin-bottom: 40px; }
        .app-title { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2.4rem; letter-spacing: -1px; line-height: 1; }
        .app-title span { color: var(--accent); }
        .app-sub { color: var(--muted); font-size: 0.82rem; margin-top: 8px; }
        /* label display fix */
        .shiny-input-radiogroup .shiny-options-group { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 20px; }
        .shiny-input-radiogroup input[type=radio] { display: none; }
        .shiny-input-radiogroup label.radio {
            display: inline-flex !important;
            align-items: center;
            padding: 8px 18px;
            border: 1.5px solid var(--border);
            border-radius: 999px;
            cursor: pointer;
            font-family: 'Syne', sans-serif;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--muted);
            transition: all 0.18s;
            background: var(--surface);
        }
        .shiny-input-radiogroup input[type=radio]:checked + span { color: #0e0f13; }
        .shiny-input-radiogroup label.radio:has(input:checked) { background: var(--accent); border-color: var(--accent); color: #0e0f13; }
        .search-wrap { display: flex; gap: 10px; margin-bottom: 36px; }
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
        .results-label { font-family: 'Syne', sans-serif; font-size: 0.75rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 16px; }
        .result-card { background: var(--surface); border: 1.5px solid var(--border); border-radius: var(--radius); padding: 22px 24px; margin-bottom: 14px; transition: border-color 0.18s; }
        .result-card:hover { border-color: var(--accent); }
        .card-header-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
        .rank-badge { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 0.78rem; color: var(--accent); }
        .product-title { font-family: 'Syne', sans-serif; font-size: 1.05rem; font-weight: 600; color: var(--text); margin-bottom: 16px; line-height: 1.35; }
        .empty-state { text-align: center; padding: 60px 0; color: var(--muted); font-size: 0.85rem; }
        .empty-state .icon { font-size: 2.5rem; margin-bottom: 12px; }
        """)
    ),
    ui.div(
        ui.div(
            ui.h1(ui.HTML('Product <span>Search</span>'), class_="app-title"),
            ui.p("Amazon product retrieval · Semantic Search", class_="app-sub"),
            class_="app-header"
        ),
        ui.input_radio_buttons(
            "mode", None,
            choices=["Semantic", "BM25", "Hybrid"],
            selected="Semantic",
            inline=True
        ),
        ui.div(
            ui.input_text("query", None, placeholder="e.g. best gaming headset for under $30", width="100%"),
            ui.input_action_button("search", "Search", class_="search-btn"),
            class_="search-wrap"
        ),
        ui.div(ui.output_ui("results")),
        class_="app-wrap"
    )
)
 
def server(input, output, session):
    @output
    @render.ui
    @reactive.event(input.search)
    def results():
        query = input.query().strip()
        if not query:
            return ui.div(
                ui.div("Search", class_="icon"),
                ui.p("Enter a query and hit Search"),
                class_="empty-state"
            )
 
        mode = input.mode()
 
        if mode != "Semantic":
            return ui.div(
                ui.div("🚧", class_="icon"),
                ui.p(f"{mode} search is coming soon."),
                class_="empty-state"
            )
 
        hits = semantic_search(docs, model, index, query, k=3)
 
        if not hits:
            return ui.div(ui.div("No results", class_="icon"), ui.p("No results found."), class_="empty-state")
 
        return ui.div(
            ui.p(f"Top 3 results - {mode} mode", class_="results-label"),
            *[result_card(i, r) for i, r in enumerate(hits)]
        )
 
app = App(app_ui, server)