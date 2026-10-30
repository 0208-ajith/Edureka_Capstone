"""Streamlit UI: upload enterprise documents and ask questions about them (Task 2).

Run with:  streamlit run app.py
"""

from __future__ import annotations

import logging
import os

import streamlit as st
from dotenv import load_dotenv

from docqa.agents import AgentOrchestrator
from docqa.config import SUPPORTED_EXTENSIONS, Settings
from docqa.guardrails import GuardrailViolation, clean_question
from docqa.ingestion import IngestionError, ingest_document
from docqa.llm import ClaudeLLM, LLMError
from docqa.rag import Answer, answer_with_rag
from docqa.vector_store import VectorStore

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

AGENT_MODE = "Agentic (plan → retrieve → reason → validate)"
RAG_MODE = "Simple RAG (retrieve → generate)"

st.set_page_config(page_title="Enterprise Document Assistant", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner="Loading the vector store...")
def get_store(path: str, collection: str) -> VectorStore:
    return VectorStore(path, collection)


def configured_api_key() -> str | None:
    """API key from the environment / .env, or from Streamlit secrets when deployed."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if key:
        return key
    try:
        return st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:  # no secrets.toml present
        return None


def render_answer(message: dict) -> None:
    st.markdown(message["content"])
    answer: Answer | None = message.get("answer")
    if answer is None:
        return
    if answer.validation:
        v = answer.validation
        status = "grounded in the sources" if v["grounded"] else "partly unverified"
        st.caption(f"Validator: {status} · confidence {v['confidence']}")
    if answer.passages:
        with st.expander(f"Sources ({len(answer.passages)})"):
            for p in answer.passages:
                st.markdown(f"**[{p.label}] {p.source}** · {p.location} · relevance {p.score:.2f}")
                st.caption(p.text[:700] + ("..." if len(p.text) > 700 else ""))
    if answer.trace:
        with st.expander("Agent trace"):
            for step in answer.trace:
                st.markdown(f"- {step}")
            if answer.plan:
                st.markdown("**Plan**")
                st.json(answer.plan, expanded=False)
            if answer.validation:
                st.markdown("**Validation**")
                st.json(answer.validation, expanded=False)


settings = Settings.from_env()
store = get_store(settings.chroma_dir, settings.collection)
st.session_state.setdefault("messages", [])

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("1. Anthropic API key")
    api_key = configured_api_key()
    if api_key:
        st.success("API key loaded from configuration.")
    else:
        api_key = st.text_input("API key", type="password", help="Used only for this session.")
        if not api_key:
            st.info("Enter an Anthropic API key to ask questions.")

    st.header("2. Upload documents")
    uploads = st.file_uploader(
        "PDF, TXT, CSV or Excel (.xlsx)",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )
    if st.button("Ingest documents", disabled=not uploads, type="primary"):
        for upload in uploads:
            with st.spinner(f"Processing {upload.name}..."):
                try:
                    report = ingest_document(upload.name, upload.getvalue(), store, settings)
                    st.success(f"{report.source}: {report.chunks} chunks indexed.")
                except (GuardrailViolation, IngestionError) as exc:
                    st.error(str(exc))
                except Exception:
                    logging.exception("Ingestion failed for %s", upload.name)
                    st.error(f"{upload.name}: unexpected error while indexing. See the server log.")

    st.header("3. Knowledge base")
    sources = store.list_sources()
    if not sources:
        st.caption("No documents indexed yet.")
    for name, count in sources.items():
        col_name, col_button = st.columns([5, 2])
        col_name.markdown(f"{name}  \n<small>{count} chunks</small>", unsafe_allow_html=True)
        if col_button.button("Remove", key=f"remove-{name}"):
            store.delete_source(name)
            st.rerun()
    if sources and st.button("Clear knowledge base"):
        store.reset()
        st.rerun()

    st.header("Settings")
    mode = st.radio("Answering mode", [AGENT_MODE, RAG_MODE])
    top_k = st.slider("Passages per search (top-k)", 1, 10, settings.top_k)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    st.caption(f"Model: {settings.model}")

# ----------------------------------------------------------------------------- chat
st.title("Enterprise Document Assistant")
st.caption(
    "Ask questions about your uploaded documents. Answers are generated by Claude from retrieved "
    "passages, cite their sources, and (in agentic mode) are checked by a validator agent."
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        render_answer(message)

question = st.chat_input("Ask a question about your documents")
if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        answer: Answer | None = None
        try:
            question = clean_question(question, settings.max_question_chars)
            if not api_key:
                raise GuardrailViolation("Add an Anthropic API key in the sidebar first.")
            llm = ClaudeLLM(settings, api_key=api_key)
            history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            if mode == AGENT_MODE:
                with st.status("Agents working...", expanded=True) as status:
                    orchestrator = AgentOrchestrator(store, llm, settings)
                    answer = orchestrator.run(question, history=history, top_k=top_k, on_step=status.write)
                    status.update(label="Done", state="complete", expanded=False)
            else:
                with st.spinner("Retrieving and generating..."):
                    answer = answer_with_rag(question, store, llm, settings, top_k=top_k)
            reply = {"role": "assistant", "content": answer.text, "answer": answer}
        except GuardrailViolation as exc:
            reply = {"role": "assistant", "content": f"⚠️ {exc}"}
        except LLMError as exc:
            reply = {"role": "assistant", "content": f"⚠️ {exc}"}
        except Exception:
            logging.exception("Unexpected error while answering")
            reply = {"role": "assistant", "content": "⚠️ Something went wrong while answering. Please try again."}
        render_answer(reply)

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(reply)
