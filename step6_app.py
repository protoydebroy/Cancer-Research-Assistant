"""
CANCER CHATBOT — STEP 6 (with CONVERSATION MEMORY + MOBILE OPTIMIZATIONS)
============================================================================
Streamlit web app with multi-turn conversation memory.

Features:
  • Multi-turn memory via query rewriting + sliding-window history
  • Cancer-type metadata filtering for precise retrieval
  • Streaming responses with source citations
  • Polished UI with custom theme
  • Mobile-friendly sidebar toggle and responsive layout

Run:
  streamlit run step6_app.py
"""

import os
import time
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from step4_retrieve import Retriever

# ── Config ────────────────────────────────────────────────────────────────────
load_dotenv()

MODEL_NAME       = "llama-3.3-70b-versatile"
TEMPERATURE      = 0.2
MAX_TOKENS       = 1024
DEFAULT_TOP_K    = 5
MEMORY_WINDOW    = 4
REWRITE_TIMEOUT  = 8

CANCER_TYPES = [
    "All cancers", "breast", "cervical", "oral", "bladder", "eye",
    "pancreas", "colorectal", "esophageal", "kidney", "laryngeal",
    "liver", "lung", "ovarian", "prostate", "skin", "stomach",
    "testicular", "thyroid", "uterine",
]

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an oncology research assistant helping doctors with cancer-related research questions. You answer based on excerpts from a clinical reference corpus covering 19 cancer types.

GROUND RULES — non-negotiable:

1. ONLY answer using the provided <context> chunks. If the context doesn't contain enough information, say so clearly: "The provided documents do not cover this in detail." Do not invent facts.

2. CITE every clinical claim using the format [source.pdf, p.N]. Multiple citations are fine: [breast.pdf, p.3][breast.pdf, p.7].

3. STRUCTURE answers for clinicians:
   - Lead with a direct one-sentence answer.
   - Then expand with relevant detail: staging, biomarkers, line of treatment, key metrics.
   - Use bullet points sparingly, only for genuinely list-like content.

4. CLINICAL TONE — assume the reader is a physician. Use proper terminology. Do not over-explain basics.

5. CONVERSATION CONTINUITY — When the doctor asks a follow-up (e.g., "what about stage 4?", "and the side effects?"), use the prior conversation as context. Maintain continuity of cancer type, patient context, and clinical scenario across turns unless the doctor changes topic.

6. SAFETY — always end with: "Note: This is a research aid, not clinical decision support. Verify with current guidelines and patient-specific factors."

7. If asked something outside oncology / the provided documents, politely decline and stay on-topic."""


REWRITE_SYSTEM_PROMPT = """You rewrite follow-up questions in a conversation into self-contained search queries.

Given a chat history and a new user question, output ONE rewritten query that:
- Is a complete, self-contained question that makes sense without the chat history
- Resolves all pronouns and references ("it", "that cancer", "this stage")
- Preserves all medical specifics from earlier turns (cancer type, stage, biomarker, line of therapy)
- Adds nothing the user didn't imply
- Stays focused on a single information need

Output ONLY the rewritten query as a single line. No quotes, no preamble, no explanation.

If the new question is already self-contained or starts a new topic, return it unchanged.

Examples:

History:
  user: First-line treatment for HER2+ breast cancer?
  assistant: [discusses trastuzumab + pertuzumab + docetaxel]
New question: what about stage 4?
Rewritten: What is the first-line treatment for stage 4 HER2+ breast cancer?

History:
  user: How is colorectal cancer staged?
  assistant: [discusses TNM staging]
New question: and the survival rates?
Rewritten: What are the survival rates by stage for colorectal cancer?

History:
  user: Tell me about breast cancer biomarkers.
  assistant: [discusses HER2, ER, PR, Ki-67]
New question: How is lung cancer screening done?
Rewritten: How is lung cancer screening done?
"""


# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cancer Research Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="auto",   # auto: expand on desktop, collapse on mobile
)


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — visual polish + mobile responsive
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Hide default Streamlit chrome ── */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* ── Page background gradient ── */
    .stApp {
        background:
            radial-gradient(ellipse at top left, rgba(99, 102, 241, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse at bottom right, rgba(168, 85, 247, 0.06) 0%, transparent 50%),
            #0e1117;
    }

    /* ── Hero header ── */
    .hero-container {
        padding: 1.5rem 0 0.5rem 0;
        margin-bottom: 0.5rem;
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #818cf8 0%, #a78bfa 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 0.4rem;
        font-weight: 400;
    }
    .hero-badges {
        margin-top: 0.8rem;
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
    }
    .hero-badge {
        background: rgba(99, 102, 241, 0.12);
        color: #a5b4fc;
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 500;
        border: 1px solid rgba(99, 102, 241, 0.25);
    }
    .hero-badge.green {
        background: rgba(34, 197, 94, 0.1);
        color: #86efac;
        border-color: rgba(34, 197, 94, 0.25);
    }
    .hero-badge.amber {
        background: rgba(251, 191, 36, 0.08);
        color: #fcd34d;
        border-color: rgba(251, 191, 36, 0.22);
    }

    /* ── Sidebar styling ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1419 0%, #0a0d12 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    [data-testid="stSidebar"] h1 {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }
    [data-testid="stSidebar"] .section-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #64748b;
        font-weight: 600;
        margin: 1.2rem 0 0.5rem 0;
    }

    /* ── Sidebar toggle button (the > arrow) — make it visible always ── */
    [data-testid="collapsedControl"] {
        background: linear-gradient(135deg, #6366f1 0%, #a78bfa 100%) !important;
        border-radius: 10px !important;
        padding: 6px !important;
        top: 0.8rem !important;
        left: 0.6rem !important;
        z-index: 999 !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4) !important;
        transition: transform 0.15s ease !important;
    }
    [data-testid="collapsedControl"]:hover {
        transform: scale(1.05) !important;
    }
    [data-testid="collapsedControl"] svg {
        fill: white !important;
        width: 22px !important;
        height: 22px !important;
    }

    /* ── Mobile-specific overrides ── */
    @media (max-width: 768px) {
        /* Bigger sidebar toggle on mobile so it's impossible to miss */
        [data-testid="collapsedControl"] {
            padding: 10px !important;
            top: 1rem !important;
            left: 0.6rem !important;
            box-shadow: 0 4px 16px rgba(99, 102, 241, 0.6) !important;
        }
        [data-testid="collapsedControl"] svg {
            width: 26px !important;
            height: 26px !important;
        }

        /* Smaller hero title on mobile */
        .hero-title {
            font-size: 1.7rem !important;
        }
        .hero-subtitle {
            font-size: 0.85rem !important;
        }
        .hero-container {
            padding-top: 3.5rem !important;  /* leave room for the toggle button */
        }

        /* Tighter chat messages on mobile */
        [data-testid="stChatMessage"] {
            padding: 0.7rem 0.9rem !important;
        }

        /* Make example buttons full-width on mobile */
        .stButton > button {
            font-size: 0.85rem !important;
            padding: 0.6rem 0.8rem !important;
        }

        /* Latency strip wraps better on mobile */
        .latency-strip {
            font-size: 0.7rem !important;
            padding: 0.3rem 0.6rem !important;
        }
        .latency-pill {
            margin-right: 0.4rem !important;
        }
    }

    /* ── Chat message styling ── */
    [data-testid="stChatMessage"] {
        background: rgba(30, 41, 59, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 14px !important;
        padding: 1rem 1.2rem !important;
        margin: 0.5rem 0 !important;
    }

    /* ── Chat input ── */
    [data-testid="stChatInput"] textarea {
        background: rgba(30, 41, 59, 0.6) !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        border-radius: 12px !important;
        color: #e2e8f0 !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: rgba(99, 102, 241, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1) !important;
    }

    /* ── Example prompt buttons ── */
    .stButton > button {
        background: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        color: #cbd5e1 !important;
        border-radius: 10px !important;
        padding: 0.7rem 1rem !important;
        font-size: 0.9rem !important;
        text-align: left !important;
        transition: all 0.2s ease !important;
        white-space: normal !important;
        height: auto !important;
        min-height: 3rem !important;
    }
    .stButton > button:hover {
        background: rgba(99, 102, 241, 0.15) !important;
        border-color: rgba(99, 102, 241, 0.5) !important;
        color: #e2e8f0 !important;
        transform: translateY(-1px) !important;
    }

    /* Sidebar action buttons (Clear/Export) */
    [data-testid="stSidebar"] .stButton > button {
        text-align: center !important;
        font-size: 0.85rem !important;
    }

    /* Download button (Export) */
    .stDownloadButton > button {
        background: rgba(30, 41, 59, 0.5) !important;
        border: 1px solid rgba(99, 102, 241, 0.2) !important;
        color: #cbd5e1 !important;
        border-radius: 10px !important;
        padding: 0.5rem 1rem !important;
        font-size: 0.85rem !important;
    }
    .stDownloadButton > button:hover {
        background: rgba(99, 102, 241, 0.15) !important;
        border-color: rgba(99, 102, 241, 0.5) !important;
    }

    /* ── Examples section heading ── */
    .examples-heading {
        color: #94a3b8;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin: 1.5rem 0 0.8rem 0;
    }

    /* ── Searched-as caption ── */
    .searched-as {
        background: rgba(99, 102, 241, 0.08);
        border-left: 3px solid #6366f1;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        color: #a5b4fc;
        font-size: 0.82rem;
        font-style: italic;
        margin-bottom: 0.6rem;
    }

    /* ── Latency strip ── */
    .latency-strip {
        background: rgba(15, 23, 42, 0.6);
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        font-size: 0.75rem;
        color: #64748b;
        font-family: 'SF Mono', Monaco, 'Cascadia Code', monospace;
        margin-top: 0.6rem;
        display: inline-block;
    }
    .latency-pill {
        color: #94a3b8;
        margin-right: 0.8rem;
    }
    .latency-pill .v {
        color: #cbd5e1;
        font-weight: 600;
    }

    /* ── Source items inside expander ── */
    .source-item {
        background: rgba(15, 23, 42, 0.5);
        border-left: 2px solid rgba(99, 102, 241, 0.4);
        padding: 0.6rem 0.9rem;
        border-radius: 6px;
        margin-bottom: 0.6rem;
    }
    .source-meta {
        font-size: 0.78rem;
        color: #94a3b8;
        font-family: 'SF Mono', Monaco, monospace;
        margin-bottom: 0.3rem;
    }
    .source-meta code {
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc !important;
        padding: 0.1rem 0.4rem;
        border-radius: 4px;
    }
    .source-text {
        color: #cbd5e1;
        font-size: 0.82rem;
        line-height: 1.5;
    }
    .similarity-pill {
        display: inline-block;
        background: rgba(34, 197, 94, 0.15);
        color: #86efac;
        padding: 0.05rem 0.5rem;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-left: 0.4rem;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: rgba(15, 23, 42, 0.4) !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-radius: 10px !important;
    }

    /* ── Status pills in sidebar ── */
    .status-line {
        font-size: 0.82rem;
        padding: 0.25rem 0;
        color: #cbd5e1;
    }

    /* ── Slider ── */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background: #6366f1 !important;
    }

    /* ── Toggle ── */
    [data-testid="stToggle"] {
        margin-top: 0.3rem;
    }

    /* ── Selectbox ── */
    .stSelectbox [data-baseweb="select"] {
        background: rgba(30, 41, 59, 0.5) !important;
    }

    /* ── Mobile hint banner ── */
    .mobile-hint {
        display: none;
        background: rgba(99, 102, 241, 0.1);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        font-size: 0.78rem;
        color: #a5b4fc;
        margin: 0.5rem 0;
    }
    @media (max-width: 768px) {
        .mobile-hint {
            display: block;
        }
    }
</style>
""", unsafe_allow_html=True)


# ── Cached resources ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading retrieval engine...")
def get_retriever():
    return Retriever()


@st.cache_resource(show_spinner=False)
def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


# ── Memory helpers ────────────────────────────────────────────────────────────
def get_recent_exchanges(messages: list, n: int = MEMORY_WINDOW) -> list:
    return messages[-(2 * n):] if messages else []


def rewrite_query_with_history(groq_client, chat_history, new_question):
    if not chat_history:
        return new_question

    history_lines = []
    for msg in chat_history:
        if msg["role"] == "user":
            history_lines.append(f"  user: {msg['content']}")
        elif msg["role"] == "assistant":
            content = msg.get("content", "")
            if len(content) > 300:
                content = content[:300] + "..."
            history_lines.append(f"  assistant: {content}")
    history_text = "\n".join(history_lines)

    user_msg = f"History:\n{history_text}\nNew question: {new_question}\nRewritten:"

    try:
        resp = groq_client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = [
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            temperature = 0.0,
            max_tokens  = 200,
            timeout     = REWRITE_TIMEOUT,
        )
        rewritten = resp.choices[0].message.content.strip().strip('"').strip("'")
        if not rewritten or len(rewritten) > 500:
            return new_question
        return rewritten
    except Exception:
        return new_question


def build_user_prompt(question: str, hits: list) -> str:
    if not hits:
        return (
            f"<context>\n  (No matching documents were retrieved.)\n</context>\n\n"
            f"<question>\n{question}\n</question>"
        )
    blocks = []
    for h in hits:
        blocks.append(
            f"[chunk {h['rank']}] source={h['source']} | page={h['page_num']} | "
            f"cancer_type={h['cancer_type']} | similarity={h['similarity']:.3f}\n"
            f"{h['text']}"
        )
    return (
        f"<context>\n" + "\n\n---\n\n".join(blocks) + f"\n</context>\n\n"
        f"<question>\n{question}\n</question>\n\n"
        f"Answer the question using only the context above. "
        f"Cite each clinical claim as [source, p.N]."
    )


def build_messages_with_memory(messages: list, current_user_prompt: str) -> list:
    out = [{"role": "system", "content": SYSTEM_PROMPT}]
    history = messages[:-1] if messages and messages[-1]["role"] == "user" else messages
    recent = get_recent_exchanges(history, n=MEMORY_WINDOW)
    for m in recent:
        out.append({"role": m["role"], "content": m["content"]})
    out.append({"role": "user", "content": current_user_prompt})
    return out


def render_sources(hits: list):
    """Render the source-list inside the expander with custom styling."""
    for h in hits:
        text_preview = h["text"][:380] + ("..." if len(h["text"]) > 380 else "")
        # Escape HTML in text
        safe_text = (text_preview.replace("&", "&amp;")
                                  .replace("<", "&lt;").replace(">", "&gt;"))
        st.markdown(
            f'<div class="source-item">'
            f'  <div class="source-meta">'
            f'    <strong>[{h["rank"]}]</strong> '
            f'    <code>{h["source"]}</code> · '
            f'    page {h["page_num"]} · '
            f'    {h["cancer_type"]}'
            f'    <span class="similarity-pill">sim {h["similarity"]:.2f}</span>'
            f'  </div>'
            f'  <div class="source-text">{safe_text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_latency_strip(rewrite_ms, retrieve_ms, llm_ms, total_ms):
    parts = []
    if rewrite_ms:
        parts.append(f'<span class="latency-pill">↻ rewrite <span class="v">{rewrite_ms:.0f}ms</span></span>')
    parts.append(f'<span class="latency-pill">⌕ retrieval <span class="v">{retrieve_ms:.0f}ms</span></span>')
    parts.append(f'<span class="latency-pill">✦ LLM <span class="v">{llm_ms:.0f}ms</span></span>')
    parts.append(f'<span class="latency-pill">∑ total <span class="v">{total_ms:.0f}ms</span></span>')
    st.markdown(
        f'<div class="latency-strip">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🩺 Settings")

    st.markdown('<div class="section-label">Cancer focus</div>', unsafe_allow_html=True)
    cancer_filter_label = st.selectbox(
        "Restrict search to a specific cancer:",
        CANCER_TYPES,
        index=0,
        label_visibility="collapsed",
    )
    cancer_filter = None if cancer_filter_label == "All cancers" else cancer_filter_label

    st.markdown('<div class="section-label">Retrieval depth</div>', unsafe_allow_html=True)
    top_k = st.slider(
        "Chunks per query (top-k):",
        min_value=1, max_value=10, value=DEFAULT_TOP_K,
        label_visibility="collapsed",
    )
    st.caption(f"Retrieving top {top_k} chunks per query")

    st.markdown('<div class="section-label">Memory</div>', unsafe_allow_html=True)
    use_memory = st.toggle(
        "Conversation memory",
        value=True,
        help="Lets the bot understand follow-ups like 'what about stage 4?' "
             "by rewriting them with prior context before retrieval.",
    )
    st.caption(f"Window: last {MEMORY_WINDOW} exchanges")

    st.markdown('<div class="section-label">Conversation</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑 Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.session_state.messages:
            export_data = json.dumps(
                [{"role": m["role"], "content": m["content"]}
                 for m in st.session_state.messages],
                indent=2,
            )
            st.download_button(
                "↓ Export",
                data=export_data,
                file_name=f"chat_{datetime.now():%Y%m%d_%H%M}.json",
                mime="application/json",
                use_container_width=True,
            )
        else:
            st.button("↓ Export", disabled=True, use_container_width=True)

    st.markdown('<div class="section-label">System status</div>', unsafe_allow_html=True)
    chroma_path = Path("data/chroma_db")
    chunks_path = Path("data/chunks.jsonl")
    emb_path    = Path("data/embeddings.npz")

    def status(ok, label, missing=""):
        icon = "🟢" if ok else "🔴"
        text = label if ok else f"{label} ({missing})"
        st.markdown(f'<div class="status-line">{icon} {text}</div>', unsafe_allow_html=True)

    status(chunks_path.exists(),     "Document chunks", "run step1")
    status(emb_path.exists(),        "Embeddings",      "run step2")
    status(chroma_path.exists(),     "Vector database", "run step3")
    status(bool(os.getenv("GROQ_API_KEY")), "API key",       "set in .env")

    st.markdown('<div class="section-label">About</div>', unsafe_allow_html=True)
    with st.expander("ℹ How this works"):
        st.markdown(
            "**Cancer Research Assistant** is a multi-turn RAG chatbot built over "
            "a corpus of 19 cancer-specific clinical reference documents.\n\n"
            "**Memory:** Query rewriting + sliding-window history "
            f"(last {MEMORY_WINDOW} exchanges).\n\n"
            "**Stack:** sentence-transformers · ChromaDB · Groq Llama 3.3 70B · "
            "Streamlit\n\n"
            "**Disclaimer:** Research aid only — not clinical decision support."
        )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

# Hero header
st.markdown("""
<div class="hero-container">
  <h1 class="hero-title">🩺 Cancer Research Assistant</h1>
  <p class="hero-subtitle">
    Grounded clinical answers across 19 cancer types with verifiable source citations
  </p>
  <div class="hero-badges">
    <span class="hero-badge">Llama 3.3 70B</span>
    <span class="hero-badge green">Multi-turn memory</span>
    <span class="hero-badge amber">Source-cited</span>
  </div>
</div>
<div class="mobile-hint">
  💡 Tap the <strong>purple ☰ button</strong> at the top-left to open settings (cancer filter, top-k, etc.)
</div>
""", unsafe_allow_html=True)


# Pre-flight checks
if not Path("data/chroma_db").exists():
    st.error(
        "Vector database not found at `data/chroma_db/`.\n\n"
        "Run the pipeline first:\n"
        "1. `python step1_ingest.py`\n"
        "2. `python step2_embed.py`\n"
        "3. `python step3_vectordb.py`"
    )
    st.stop()

groq_client = get_groq_client()
if groq_client is None:
    st.error(
        "GROQ_API_KEY not found.\n\n"
        "Add this to your `.env` file:\n\n"
        "```\nGROQ_API_KEY=gsk_yourActualKeyHere\n```\n\n"
        "Get a free key at [console.groq.com](https://console.groq.com)."
    )
    st.stop()

retriever = get_retriever()


# Example prompts when conversation is empty
if not st.session_state.messages and "pending_query" not in st.session_state:
    st.markdown('<div class="examples-heading">✨ Try asking</div>', unsafe_allow_html=True)
    examples = [
        "What is the first-line treatment for HER2+ breast cancer?",
        "How is stage 3 lung cancer staged and treated?",
        "What are common metastatic sites in colorectal cancer?",
        "Explain key biomarkers used in cervical cancer screening",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state.pending_query = ex
                st.rerun()


# Replay history
for msg in st.session_state.messages:
    avatar = "🧑‍⚕️" if msg["role"] == "user" else "🩺"
    with st.chat_message(msg["role"], avatar=avatar):
        if msg["role"] == "assistant":
            if msg.get("rewritten_query") and msg["rewritten_query"] != msg.get("original_query"):
                st.markdown(
                    f'<div class="searched-as">🔍 Searched as: {msg["rewritten_query"]}</div>',
                    unsafe_allow_html=True,
                )
        st.markdown(msg["content"])

        if msg["role"] == "assistant":
            if msg.get("hits"):
                with st.expander(f"📚  View {len(msg['hits'])} sources"):
                    render_sources(msg["hits"])
            if msg.get("latency"):
                lat = msg["latency"]
                render_latency_strip(
                    lat.get("rewrite_ms", 0),
                    lat["retrieve_ms"],
                    lat["llm_ms"],
                    lat["total_ms"],
                )


# Handle new input
user_query = st.chat_input("Ask a question (follow-ups understood)...")

if "pending_query" in st.session_state:
    user_query = st.session_state.pending_query
    del st.session_state.pending_query

if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user", avatar="🧑‍⚕️"):
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="🩺"):
        # Step 1: Query rewriting
        rewrite_ms = 0
        prior_history = st.session_state.messages[:-1]

        if use_memory and prior_history:
            with st.spinner("Resolving conversation context…"):
                t0 = time.time()
                search_query = rewrite_query_with_history(
                    groq_client,
                    get_recent_exchanges(prior_history, n=MEMORY_WINDOW),
                    user_query,
                )
                rewrite_ms = (time.time() - t0) * 1000
        else:
            search_query = user_query

        if search_query != user_query:
            st.markdown(
                f'<div class="searched-as">🔍 Searched as: {search_query}</div>',
                unsafe_allow_html=True,
            )

        # Step 2: Retrieve
        with st.spinner("Searching documents…"):
            t0 = time.time()
            hits = retriever.retrieve(
                search_query,
                top_k=top_k,
                cancer_type=cancer_filter,
            )
            retrieve_ms = (time.time() - t0) * 1000

        if not hits:
            warn_msg = (
                "I couldn't find anything relevant in the documents"
                + (f" for cancer type '{cancer_filter}'" if cancer_filter else "")
                + ". Try rephrasing or removing the cancer-type filter."
            )
            st.warning(warn_msg)
            st.session_state.messages.append({
                "role": "assistant", "content": warn_msg,
                "hits": [], "latency": None,
                "original_query": user_query, "rewritten_query": search_query,
            })
            st.stop()

        # Step 3: Generate answer with memory
        rag_prompt = build_user_prompt(search_query, hits)
        full_messages = build_messages_with_memory(
            st.session_state.messages,
            rag_prompt,
        )

        placeholder = st.empty()
        answer_chunks = []

        t0 = time.time()
        try:
            stream = groq_client.chat.completions.create(
                model       = MODEL_NAME,
                messages    = full_messages,
                temperature = TEMPERATURE,
                max_tokens  = MAX_TOKENS,
                stream      = True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    answer_chunks.append(delta)
                    placeholder.markdown("".join(answer_chunks) + "▌")
        except Exception as e:
            err = f"⚠️ LLM error: `{type(e).__name__}: {e}`"
            placeholder.error(err)
            st.session_state.messages.append({
                "role": "assistant", "content": err,
                "hits": hits, "latency": None,
                "original_query": user_query, "rewritten_query": search_query,
            })
            st.stop()

        llm_ms  = (time.time() - t0) * 1000
        total_ms = rewrite_ms + retrieve_ms + llm_ms
        answer = "".join(answer_chunks)
        placeholder.markdown(answer)

        with st.expander(f"📚  View {len(hits)} sources"):
            render_sources(hits)

        render_latency_strip(rewrite_ms, retrieve_ms, llm_ms, total_ms)

        st.session_state.messages.append({
            "role":            "assistant",
            "content":         answer,
            "hits":            hits,
            "original_query":  user_query,
            "rewritten_query": search_query,
            "latency": {
                "rewrite_ms":  rewrite_ms,
                "retrieve_ms": retrieve_ms,
                "llm_ms":      llm_ms,
                "total_ms":    total_ms,
            },
        })
