from shiny import App, ui, reactive, render
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pyarrow.parquet as pq
from app_utils import *
import pickle


# Load data & models
table = pq.read_table("./data/processed/product_documents.parquet")
docs = table.to_pandas()

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("./data/processed/semantic_search_index.faiss")

with open("./data/processed/bm25_index.pkl", "rb") as f:
    bm25_data = pickle.load(f)

bm25 = bm25_data["bm25"]
doc_names = bm25_data["doc_names"]

# ── Result card (Search Only) ──────────────────────────────────────────────────

def result_card(i, r):
    # Safely pull average rating if the column exists
    rating = r.get("average_rating") if hasattr(r, "get") else r.get("average_rating", None)
    rating_badge = (
        ui.span(f"Rating: {rating:.1f}", class_="rating-badge")
        if rating is not None else ui.span()
    )
    # # First 200 chars of review text
    # preview = ""
    # for col in ("review_text", "reviewText", "body", "text"):
    #     if col in r and r[col]:
    #         preview = str(r[col])[:200]
    #         break

    preview = None
    
    return ui.div(
        ui.div(
            ui.span(f"#{i+1}", class_="rank-badge"),
            rating_badge,
            class_="card-header-row"
        ),
        ui.h4(r["product_title"], class_="product-title"),
        ui.p(preview + ("…" if len(preview) == 200 else ""), class_="review-preview") if preview else ui.span(),
        class_="result-card"
    )


# ── UI ─────────────────────────────────────────────────────────────────────────

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

        /* ── Mode tabs ── */
        .mode-tabs { display: flex; gap: 0; margin-bottom: 32px; border: 1.5px solid var(--border); border-radius: var(--radius); overflow: hidden; width: fit-content; }
        .mode-tab {
            padding: 10px 24px;
            font-family: 'Syne', sans-serif; font-weight: 700; font-size: 0.85rem;
            cursor: pointer; background: var(--surface); color: var(--muted);
            border: none; transition: all 0.18s; user-select: none;
        }
        .mode-tab.active { background: var(--accent); color: #0e0f13; }
        .mode-tab:not(:last-child) { border-right: 1.5px solid var(--border); }

        /* ── Sub-mode radio pills (Search Only) ── */
        .shiny-input-radiogroup > label { display: none; }
        .shiny-input-radiogroup .shiny-options-group { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 20px; }
        .shiny-input-radiogroup input[type=radio] { display: none; }
        .shiny-input-radiogroup label.radio {
        display: inline-flex !important;
        align-items: center;
        padding: 8px 18px;
        border: 1.5px solid var(--border);
        border-radius: 999px;
        cursor: pointer;
        font-family: 'Syne', sans-serif; font-size: 0.85rem; font-weight: 600;
        color: var(--muted); transition: all 0.18s; background: var(--surface);
        }
        .shiny-input-radiogroup label.radio:has(input:checked) { background: var(--accent); border-color: var(--accent); color: #0e0f13; }

        /* ── Search bar ── */
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

        /* ── Result cards ── */
        .results-label { font-family: 'Syne', sans-serif; font-size: 0.75rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin-bottom: 16px; }
        .result-card { background: var(--surface); border: 1.5px solid var(--border); border-radius: var(--radius); padding: 22px 24px; margin-bottom: 14px; transition: border-color 0.18s; }
        .result-card:hover { border-color: var(--accent); }
        .card-header-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
        .rank-badge { font-family: 'Syne', sans-serif; font-weight: 800; font-size: 0.78rem; color: var(--accent); }
        .rating-badge { font-family: 'Syne', sans-serif; font-weight: 600; font-size: 0.78rem; color: var(--accent2); margin-left: auto; }
        .product-title { font-family: 'Syne', sans-serif; font-size: 1.05rem; font-weight: 600; color: var(--text); margin-bottom: 10px; line-height: 1.35; }
        .review-preview { font-size: 0.82rem; color: var(--muted); line-height: 1.6; margin-top: 6px; }

        /* ── RAG answer panel ── */
        .rag-answer-wrap { margin-bottom: 28px; }
        .rag-answer-label { font-family: 'Syne', sans-serif; font-size: 0.75rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: var(--accent2); margin-bottom: 10px; }
        .rag-answer {
            background: var(--surface2); border: 1.5px solid var(--accent2);
            border-radius: var(--radius); padding: 22px 24px;
            font-size: 0.92rem; line-height: 1.7; color: var(--text);
        }
        .rag-thinking { color: var(--muted); font-size: 0.85rem; font-style: italic; }
        .rag-sources-label { font-family: 'Syne', sans-serif; font-size: 0.72rem; font-weight: 600; letter-spacing: 2px; text-transform: uppercase; color: var(--muted); margin: 20px 0 10px; }

        /* ── Empty / loading states ── */
        .empty-state { text-align: center; padding: 60px 0; color: var(--muted); font-size: 0.85rem; }
        .empty-state .icon { font-size: 2.5rem; margin-bottom: 12px; }

        /* ── Panel visibility ── */
        .panel { display: none; }
        .panel.visible { display: block; }
        """),
        # Tiny JS to handle the mode-tab toggle (flips a hidden Shiny input)
        ui.tags.script("""
        function switchMode(mode) {
            document.querySelectorAll('.mode-tab').forEach(t => t.classList.remove('active'));
            document.getElementById('tab-' + mode).classList.add('active');
            document.querySelectorAll('.panel').forEach(p => p.classList.remove('visible'));
            document.getElementById('panel-' + mode).classList.add('visible');
        }
        document.addEventListener('DOMContentLoaded', function() {
            switchMode('search');
        });
        """)
    ),
    ui.div(
        # Header
        ui.div(
            ui.h1(ui.HTML('Product <span>Search</span>'), class_="app-title"),
            ui.p("Amazon product retrieval · Semantic & BM25 · RAG", class_="app-sub"),
            class_="app-header"
        ),

        # Mode tabs
        ui.div(
            ui.tags.button("Search Only", id="tab-search",  class_="mode-tab active",
                           onclick="switchMode('search')"),
            ui.tags.button("RAG Mode",    id="tab-rag",     class_="mode-tab",
                           onclick="switchMode('rag')"),
            class_="mode-tabs"
        ),


        # ── PANEL: Search Only ────────────────────────────────────────────────
        ui.div(
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
            id="panel-search", class_="panel visible"
        ),

        # ── PANEL: RAG Mode ───────────────────────────────────────────────────
        ui.div(
            ui.div(
                ui.input_text("rag_query", None,
                              placeholder="e.g. What's a good waterproof laptop?",
                              width="100%"),
                ui.input_action_button("rag_btn", "Ask", class_="search-btn"),
                class_="search-wrap"
            ),
            ui.div(ui.output_ui("rag_results")),
            id="panel-rag", class_="panel"
        ),

        class_="app-wrap"
    )
)


# ── Server ─────────────────────────────────────────────────────────────────────

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

    # @output
    # @render.ui
    # @reactive.event(input.rag_btn)
    # async def rag_results():
    #     query = input.rag_query().strip()
    #     if not query:
    #         return ui.p("Enter a question.")
        
    #     ui.notification_show("Running RAG pipeline...", duration=None, id="rag_notif")
    #     answer = await asyncio.to_thread(run_hybrid_chain, query)
    #     ui.notification_remove("rag_notif")
        
    #     return ui.p(answer)
    
    @output
    @render.ui
    @reactive.event(input.rag_btn)
    def rag_results():
        query = input.rag_query().strip()
        if not query:
            return ui.p("Enter a question.")
        
        answer = run_hybrid_chain(query)
        return ui.p(answer)

    # @output
    # @render.ui
    # @reactive.event(input.rag_btn)
    # def rag_results():
    #     query = input.rag_query().strip()
    #     if not query:
    #         return ui.p("Enter a question.")
        
    #     answer = run_hybrid_chain(query)
    #     return ui.p(answer)
    # def rag_results():
    #     query = input.rag_query().strip()
    #     if not query:
    #         return ui.div(ui.div("💬", class_="icon"), ui.p("Ask a question to get a RAG-powered answer"), class_="empty-state")

    #     answer = run_hybrid_chain(query)
    #     print(f"[RAG] query received: {query}")
    #     answer = run_hybrid_chain(query)
    #     hits = semantic_search(docs, model, index, query, k=5)
    #     print(f"[RAG] answer generated: {answer[:100]}")

    #     return ui.div(
    #         ui.p("Answer", class_="rag-answer-label"),
    #         ui.div(answer, class_="rag-answer"),
    #         ui.p("Retrieved sources", class_="rag-sources-label"),
    #         *[result_card(i, r) for i, r in enumerate(hits)]
    #     )


    # # ── RAG Mode ──────────────────────────────────────────────────────────────

    # @output
    # @render.ui
    # @reactive.event(input.rag_btn)
    # def rag_results():
    #     query = input.rag_query().strip()
    #     if not query:
    #         return ui.div(
    #             ui.div("💬", class_="icon"),
    #             ui.p("Ask a question to get a RAG-powered answer"),
    #             class_="empty-state"
    #         )

    #     # 1. Retrieve context docs via semantic search
    #     hits = semantic_search(docs, model, index, query, k=5)

    #     # 2. Build context string for the LLM
    #     context_parts = []
    #     for i, r in enumerate(hits):
    #         title = r.get("product_title", "Unknown product")
    #         review = ""
    #         for col in ("review_text", "reviewText", "body", "text"):
    #             if col in r and r[col]:
    #                 review = str(r[col])[:400]
    #                 break
    #         context_parts.append(f"[{i+1}] {title}\n{review}")
    #     context = "\n\n".join(context_parts)

    #     # 3. Generate answer — swap in your preferred LLM call here
    #     answer = rag_generate(query, context)   # defined in utils.py

    #     # 4. Render
    #     return ui.div(
    #         # Answer panel
    #         ui.div(
    #             ui.p("Answer", class_="rag-answer-label"),
    #             ui.div(answer, class_="rag-answer"),
    #             class_="rag-answer-wrap"
    #         ),
    #         # Source cards
    #         ui.p("Retrieved sources", class_="rag-sources-label"),
    #         *[result_card(i, r) for i, r in enumerate(hits)]
    #     )


app = App(app_ui, server)