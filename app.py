import os
import re
import hashlib
from io import BytesIO

import faiss
import fitz  # PyMuPDF
import numpy as np
import requests
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# Cyber Law RAG Assistant
# Source: user-provided Prevention of Electronic Crimes Act PDF
# ============================================================

st.set_page_config(
    page_title="Pakistan Cyber Law RAG Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# IMPORTANT:
# Put the direct/public PDF download URL in Streamlit Secrets as:
# Default public source PDF. Override with Streamlit Secret CYBER_LAW_PDF_URL if needed.
CYBER_LAW_PDF_URL = "https://www.pakistancode.gov.pk/pdffiles/administrator6a061efe0ed5bd153fa8b79b8eb4cba7.pdf"
#
# For Colab/local use, you can also set:
# Default public source PDF: https://www.pakistancode.gov.pk/pdffiles/administrator6a061efe0ed5bd153fa8b79b8eb4cba7.pdf
#
# The uploaded PDF is used as the source for the app design, but it is
# intentionally NOT bundled because the requested GitHub project has only
# three files: app.py, requirements.txt, readme.md.


def get_secret(name, default=""):
    """Read a value from Streamlit secrets first, then environment variables."""
    try:
        value = st.secrets.get(name, "")
        if value:
            return value
    except Exception:
        pass
    return os.getenv(name, default)


def normalize_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def download_pdf(url):
    """Download a public PDF into memory."""
    if not url:
        raise ValueError(
            "No PDF URL configured. Add CYBER_LAW_PDF_URL to Streamlit Secrets "
            "or an environment variable."
        )

    headers = {"User-Agent": "Pakistan-Cyber-Law-RAG/1.0"}
    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()
    if "pdf" not in content_type and not response.content.startswith(b"%PDF"):
        raise ValueError(
            "The configured URL did not return a PDF. Please use a direct PDF URL."
        )

    if len(response.content) > 30 * 1024 * 1024:
        raise ValueError("PDF is larger than 30 MB. Please use a smaller source file.")

    return response.content


def extract_pages(pdf_bytes):
    """Extract text page-by-page so retrieved passages can show page numbers."""
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []

    for page_number, page in enumerate(document, start=1):
        text = normalize_text(page.get_text("text"))
        if text:
            pages.append(
                {
                    "page": page_number,
                    "text": text,
                }
            )

    document.close()

    if not pages:
        raise ValueError(
            "No selectable text was found in the PDF. This app expects a text-based PDF."
        )

    return pages


def make_chunks(pages, chunk_size=1200, overlap=180):
    """
    Split each page into overlapping word chunks.
    Keeping page boundaries makes legal citations easier to understand.
    """
    chunks = []

    for item in pages:
        words = item["text"].split()
        if not words:
            continue

        start = 0
        chunk_id = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            text = " ".join(words[start:end]).strip()

            if len(text) >= 80:
                chunks.append(
                    {
                        "id": len(chunks),
                        "page": item["page"],
                        "chunk": chunk_id,
                        "text": text,
                    }
                )

            if end >= len(words):
                break

            start = max(end - overlap, start + 1)
            chunk_id += 1

    return chunks


@st.cache_resource(show_spinner=False)
def build_knowledge_base(pdf_url):
    """
    Download PDF and create embeddings + FAISS index on first app startup.
    Streamlit caches this resource, so it is not rebuilt on every interaction.
    """
    pdf_bytes = download_pdf(pdf_url)

    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()[:12]
    pages = extract_pages(pdf_bytes)
    chunks = make_chunks(pages)

    if not chunks:
        raise ValueError("The PDF could not be split into searchable chunks.")

    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [item["text"] for item in chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return {
        "pdf_hash": pdf_hash,
        "pages": pages,
        "chunks": chunks,
        "index": index,
        "embedder": model,
        "page_count": len(pages),
        "chunk_count": len(chunks),
    }


def retrieve(question, knowledge_base, top_k=5, min_score=0.20):
    """Semantic search using cosine similarity via normalized FAISS inner product."""
    query_vector = knowledge_base["embedder"].encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores, indices = knowledge_base["index"].search(query_vector, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        item = knowledge_base["chunks"][int(idx)]
        results.append(
            {
                "score": float(score),
                "page": item["page"],
                "chunk": item["chunk"],
                "text": item["text"],
            }
        )

    # Prefer high-confidence evidence. If nothing passes the threshold,
    # return the best result so the LLM can explicitly say evidence is weak.
    filtered = [r for r in results if r["score"] >= min_score]
    return filtered if filtered else results[:2]


def build_context(results):
    blocks = []
    for i, result in enumerate(results, start=1):
        blocks.append(
            f"[SOURCE {i} | PDF page {result['page']} | similarity {result['score']:.3f}]\n"
            f"{result['text']}"
        )
    return "\n\n".join(blocks)


def answer_with_groq(question, results, api_key, model_name, max_tokens, temperature):
    client = Groq(api_key=api_key)
    context = build_context(results)

    system_prompt = """
You are a Pakistan cyber-law research assistant.

Your ONLY legal source for substantive legal claims in this response is the
retrieved text from the user's Pakistan cyber-law PDF. Do not silently replace
it with another law, website, memory, or current legal amendment.

Rules:
1. Answer the user's question using only the retrieved source passages.
2. Identify the relevant section(s) when the source supports doing so.
3. Explain the rule in simple language first, then give legal detail.
4. If the source contains a punishment, state the imprisonment/fine exactly as
   supported by the retrieved text. Do not invent a punishment.
5. Distinguish "may extend to" maximums from mandatory punishments.
6. If the retrieved passages do not establish an answer, say:
   "The provided PDF does not give enough information to answer this precisely."
   Then tell the user what additional section/topic is needed.
7. Do not provide instructions for committing cybercrime, evading law
   enforcement, bypassing security, or exploiting systems.
8. For a real case, emphasize that this is an informational RAG assistant,
   not a lawyer or a substitute for professional legal advice.
9. Use citations like [PDF p. 6, Section 10] only when supported by the
   retrieved passages. Never invent section numbers.
10. Do not claim that this PDF is the current version of Pakistani law unless
    the PDF itself establishes that fact.

Keep the answer focused and useful.
"""

    user_prompt = f"""
Question:
{question}

Retrieved legal source passages:
{context}

Write the answer now. If the passages are insufficient, clearly say so.
"""

    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_completion_tokens=max_tokens,
    )

    return completion.choices[0].message.content


def render_sources(results):
    with st.expander("📚 Retrieved legal evidence", expanded=False):
        for i, result in enumerate(results, start=1):
            st.markdown(
                f"**Source {i} — PDF page {result['page']} — "
                f"similarity {result['score']:.3f}**"
            )
            st.write(result["text"])
            st.divider()


# -------------------- UI --------------------

st.title("⚖️ Pakistan Cyber Law RAG Assistant")
st.caption(
    "Ask questions about cyber offences, punishments, investigation powers, "
    "digital evidence and related provisions in the configured Pakistan cyber-law PDF."
)

with st.sidebar:
    st.header("⚙️ Settings")

    api_key = get_secret("GROQ_API_KEY")
    pdf_url = get_secret("CYBER_LAW_PDF_URL") or "https://www.pakistancode.gov.pk/pdffiles/administrator6a061efe0ed5bd153fa8b79b8eb4cba7.pdf"

    model_name = st.selectbox(
        "Groq model",
        [DEFAULT_MODEL],
        index=0,
        help="The application is configured for Groq's openai/gpt-oss-20b model.",
    )

    response_size = st.slider(
        "Response size",
        min_value=200,
        max_value=4000,
        value=1200,
        step=100,
        help="Maximum completion tokens requested from Groq.",
    )

    temperature = st.slider(
        "Creativity",
        min_value=0.0,
        max_value=0.8,
        value=0.1,
        step=0.1,
        help="Low temperature is recommended for legal Q&A.",
    )

    top_k = st.slider(
        "Retrieved passages",
        min_value=2,
        max_value=10,
        value=5,
        step=1,
    )

    min_score = st.slider(
        "Minimum similarity",
        min_value=0.0,
        max_value=0.8,
        value=0.20,
        step=0.05,
        help="Lower values retrieve more broadly; higher values are stricter.",
    )

    show_sources = st.checkbox("Show retrieved evidence", value=True)
    show_debug = st.checkbox("Show technical details", value=False)

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("### About")
    st.write(
        "This app uses PyMuPDF → chunking → Sentence Transformers → FAISS → "
        "Groq to answer from the configured legal PDF."
    )
    st.info(
        "Legal notice: this tool is for education and information. "
        "It is not a lawyer and should not be relied on as legal advice."
    )

if not api_key:
    st.error(
        "GROQ_API_KEY is missing. Add it to Streamlit Secrets or your environment."
    )
    st.stop()

if not pdf_url:
    st.error(
        "CYBER_LAW_PDF_URL is missing. Add a direct/public PDF URL to the "
        "same cyber-law PDF you want the RAG system to use."
    )
    st.stop()

try:
    with st.spinner("Downloading the cyber-law PDF and building FAISS embeddings..."):
        kb = build_knowledge_base(pdf_url)

    st.success(
        f"Knowledge base ready: {kb['page_count']} pages • "
        f"{kb['chunk_count']} searchable chunks"
    )
except Exception as exc:
    st.error(f"Knowledge-base startup failed: {exc}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            if show_sources:
                render_sources(message["sources"])

question = st.chat_input(
    "Ask a question about a cyber offence, punishment, section, investigation power, etc."
)

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the cyber-law PDF and preparing the answer..."):
            try:
                results = retrieve(
                    question,
                    kb,
                    top_k=top_k,
                    min_score=min_score,
                )

                if not results:
                    response = (
                        "The provided PDF does not contain enough retrievable "
                        "information to answer this question precisely."
                    )
                else:
                    response = answer_with_groq(
                        question=question,
                        results=results,
                        api_key=api_key,
                        model_name=model_name,
                        max_tokens=response_size,
                        temperature=temperature,
                    )

                st.markdown(response)

                if show_sources:
                    render_sources(results)

                if show_debug:
                    st.caption(
                        f"Retrieved {len(results)} passages | "
                        f"Model: {model_name} | "
                        f"Embedding: {EMBEDDING_MODEL}"
                    )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                        "sources": results,
                    }
                )

            except Exception as exc:
                error_message = (
                    "I could not generate the answer. "
                    f"Technical error: {exc}"
                )
                st.error(error_message)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )
