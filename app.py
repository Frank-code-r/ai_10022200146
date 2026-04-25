# app.py
# Author: Frank Afelete Kofi Dogli | Index: 10022200146
# ACity RAG Chatbot — CS4241 Introduction to Artificial Intelligence 2026

import os
import streamlit as st
from rag.pipeline import build_index, run_query
from rag.logger import get_session_log, format_log_for_display

st.set_page_config(
    page_title="ACity AI — RAG Chatbot",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 ACity AI — RAG Chatbot")
st.caption("CS4241 · Introduction to Artificial Intelligence · Academic City University · 2026")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
        help="Get yours at console.groq.com",
    )

    st.divider()
    st.header("🔧 Retrieval Settings")
    top_k = st.slider("Top-K chunks to retrieve", 3, 10, 5)
    use_hybrid = st.toggle("Hybrid Search (vector + keyword)", value=True)

    st.divider()
    st.header("🔄 Index")
    rebuild = st.button("Rebuild Index from Scratch")

    st.divider()
    st.header("💡 Example Queries")
    examples = [
        "What are the key priorities of the 2025 Ghana budget?",
        "How many votes did NDC get in the Ashanti region in 2020?",
        "What is the government's plan for education spending?",
        "Who won the most votes in the Eastern region in 2016?",
        "What does the budget say about infrastructure?",
    ]
    for ex in examples:
        if st.button(ex, key=ex):
            st.session_state["example_query"] = ex

# ── Load Index ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Building index... this takes 3-5 minutes on first run.")
def get_retriever():
    return build_index(force_rebuild=False)

if rebuild:
    st.cache_resource.clear()

try:
    store, retriever = get_retriever()
    st.session_state["retriever"] = retriever
    st.sidebar.success(f"✅ {store.index.ntotal} vectors loaded")
except Exception as e:
    import traceback
    st.error(f"Index build failed: {e}")
    st.code(traceback.format_exc())
    st.stop()

# ── Chat History ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Query Input ────────────────────────────────────────────────────────────────
query = st.chat_input("Ask about Ghana elections or the 2025 budget...")

# Handle example button clicks
if "example_query" in st.session_state:
    query = st.session_state.pop("example_query")

if query:
    if not api_key:
        st.warning("Please enter your Groq API key in the sidebar.")
        st.stop()

    # Show user message
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = run_query(
                query=query,
                retriever=st.session_state["retriever"],
                api_key=api_key,
                top_k=top_k,
                use_hybrid=use_hybrid,
            )

        # Answer
        st.markdown("### 💬 Answer")
        st.markdown(result["answer"])

        # Retrieved chunks
        with st.expander(f"📄 Retrieved Chunks ({len(result['retrieved_chunks'])})"):
            for i, (chunk, score) in enumerate(result["retrieved_chunks"], 1):
                source = "🗳️ Election Data" if chunk.source == "csv" else "💰 Budget PDF"
                st.markdown(f"**#{i} · {source} · Score: {score:.5f}**")
                st.text(chunk.text[:400])
                st.divider()

        # Final prompt
        with st.expander("🧠 Final Prompt Sent to LLM"):
            st.code(result["user_message"], language="markdown")

        # Pipeline log
        with st.expander("📋 Pipeline Log"):
            st.markdown(format_log_for_display(get_session_log()))

        st.session_state["messages"].append({
            "role": "assistant",
            "content": result["answer"],
        })

# ── Adversarial Testing Panel ──────────────────────────────────────────────────
st.divider()
with st.expander("⚔️ Part E — RAG vs Vector-Only Comparison"):
    st.markdown("Test the same query in both modes side by side.")
    adv_query = st.text_input("Enter a query:", key="adv_input")
    
    if st.button("Run Comparison"):
        if not api_key:
            st.warning("Add your Groq API key first.")
        elif adv_query:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**🔵 Hybrid RAG**")
                with st.spinner():
                    r1 = run_query(adv_query, st.session_state["retriever"], api_key, top_k=top_k, use_hybrid=True)
                st.markdown(r1["answer"])

            with col2:
                st.markdown("**🟠 Vector Only**")
                with st.spinner():
                    r2 = run_query(adv_query, st.session_state["retriever"], api_key, top_k=top_k, use_hybrid=False)
                st.markdown(r2["answer"])
