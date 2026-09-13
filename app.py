import os
import re
import hashlib
from datetime import datetime, timezone

import faiss
import fitz  # PyMuPDF
import numpy as np
import requests
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# Pakistan Cyber Law - Automatic Update / Version-Aware RAG
# Source: Official Pakistan Code
# ============================================================

st.set_page_config(
    page_title="Pakistan Cyber Law AI Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_MODEL = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Official Pakistan Code consolidated PECA PDF.
# The Streamlit Secret CYBER_LAW_PDF_URL can override this URL.
DEFAULT_PDF_URL = (
    "https://www.pakistancode.gov.pk/pdffiles/"
    "administrator6a061efe0ed5bd153fa8b79b8eb4cba7.pdf"
)

MAX_PDF_MB = 30


def get_secret(name, default=""):
    """Read Streamlit Secrets first, then environment variables."""
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
    """Download and validate a public PDF."""
    if not url:
        raise ValueError("No cyber-law PDF URL is configured.")

    headers = {
        "User-Agent": "Pakistan-Cyber-Law-RAG/2.0",
        "Accept": "application/pdf,*/*",
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=90,
        allow_redirects=True,
    )
    response.raise_for_status()

    content = response.content
    content_type = response.headers.get("content-type", "").lower()

    if "pdf" not in content_type and not content.startswith(b"%PDF"):
        raise ValueError(
            "The configured source did not return a PDF. "
            "Use a direct PDF URL."
        )

    if len(content) > MAX_PDF_MB * 1024 * 1024:
        raise ValueError(f"PDF is larger than {MAX_PDF_MB} MB.")

    return content, response.headers


def source_hash(pdf_bytes):
    return hashlib.sha256(pdf_bytes).hexdigest()


def extract_pages(pdf_bytes):
    """Extract text page-by-page for page-aware legal citations."""
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
            "No selectable text was found in the PDF. "
            "A text-based PDF is required."
        )

    return pages


def make_chunks(pages, chunk_size=1200, overlap=180):
    """Create overlapping page-aware chunks."""
    chunks = []

    for item in pages:
        words = item["text"].split()
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

    if not chunks:
        raise ValueError("The PDF produced no searchable chunks.")

    return chunks


def detect_version_info(pages):
    """
    Detect amendment/version clues from the official consolidated PDF.
    This is metadata only; it does not invent a version.
    """
    full_text = "\n".join(page["text"] for page in pages)

    amendment_refs = sorted(
        set(
            re.findall(
                r"(?:Act\s+No\.\s*[A-Z0-9IVXLCDM]+\s+of\s+20\d{2})",
                full_text,
                flags=re.IGNORECASE,
            )
        )
    )

    dates = sorted(
        set(
            re.findall(
                r"\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b",
                full_text,
            )
        )
    )

    # Common Pakistan Code amendment marker.
    has_2025_amendment = bool(
        re.search(
            r"Prevention of Electronic Crimes.*Amendment.*2025",
            full_text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )

    has_act_ii_2025 = bool(
        re.search(
            r"Act\s+No\.\s*II\s+of\s+2025",
            full_text,
            flags=re.IGNORECASE,
        )
    )

    # The official consolidated document is still titled PECA 2016,
    # but it incorporates amendments. Therefore "latest PECA" should
    # be described as "PECA 2016 as amended", not as a replacement Act.
    if has_act_ii_2025 or has_2025_amendment:
        legal_version = (
            "Prevention of Electronic Crimes Act, 2016, "
            "as amended by the Prevention of Electronic Crimes "
            "(Amendment) Act, 2025 (Act No. II of 2025)"
        )
    else:
        legal_version = "Prevention of Electronic Crimes Act, 2016"

    return {
        "amendment_refs": amendment_refs,
        "dates": dates[-10:],
        "has_2025_amendment": has_2025_amendment,
        "has_act_ii_2025": has_act_ii_2025,
        "legal_version": legal_version,
    }


@st.cache_resource(show_spinner=False)
def build_knowledge_base(pdf_url, pdf_hash, pdf_bytes):
    """
    Build FAISS only for the exact downloaded PDF hash.
    If the official PDF changes, the hash changes and a new index is built.
    """
    pages = extract_pages(pdf_bytes)
    chunks = make_chunks(pages)
    version_info = detect_version_info(pages)

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
        "version_info": version_info,
        "source_url": pdf_url,
    }


def load_source(url, force_refresh=False):
    """
    Download the official source once per Streamlit session.
    Clicking Check for updates forces a fresh download.
    """
    if force_refresh or "source_snapshot" not in st.session_state:
        pdf_bytes, headers = download_pdf(url)
        pdf_hash = source_hash(pdf_bytes)

        st.session_state.source_snapshot = {
            "bytes": pdf_bytes,
            "hash": pdf_hash,
            "headers": dict(headers),
            "checked_at": datetime.now(timezone.utc),
        }

    return st.session_state.source_snapshot


def retrieve(question, knowledge_base, top_k=5, min_score=0.20):
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

    filtered = [r for r in results if r["score"] >= min_score]
    return filtered if filtered else results[:2]


def build_context(results):
    blocks = []
    for i, result in enumerate(results, start=1):
        blocks.append(
            f"[SOURCE {i} | PDF page {result['page']} | "
            f"similarity {result['score']:.3f}]\n{result['text']}"
        )
    return "\n\n".join(blocks)


def answer_with_groq(
    question,
    results,
    api_key,
    model_name,
    max_tokens,
    temperature,
    kb,
):
    client = Groq(api_key=api_key)
    context = build_context(results)
    version_info = kb["version_info"]

    if version_info["has_2025_amendment"]:
        version_note = (
            "The retrieved official consolidated source contains references "
            "to the Prevention of Electronic Crimes (Amendment) Act, 2025."
        )
    else:
        version_note = (
            "Do not assume that this source contains every later amendment."
        )

    system_prompt = f"""
You are a Pakistan cyber-law research assistant.

SOURCE POLICY
Your substantive legal claims must come ONLY from the retrieved passages
from the configured official Pakistan Code PDF.

VERSION AWARENESS
The application checks the downloaded official PDF by SHA-256 hash and rebuilds
the RAG index whenever the source PDF changes. Do not invent an amendment date
or claim that an amendment exists unless the retrieved source supports it.

Current source metadata:
- Source: Official Pakistan Code consolidated PDF
- Current legal version represented by this source: {version_info["legal_version"]}
- PDF SHA-256 prefix: {kb["pdf_hash"][:16]}
- Pages: {kb["page_count"]}
- Searchable chunks: {kb["chunk_count"]}
- Detected amendment references: {version_info["amendment_refs"]}
- {version_note}

ANSWER RULES
1. Answer using the retrieved legal passages.
2. Give the relevant section when the passages support it.
3. Explain the answer in simple language first.
4. State imprisonment/fines exactly as supported by the source.
5. Distinguish maximum penalties from mandatory penalties.
6. If the evidence is insufficient, say:
   "The current indexed source does not provide enough information to answer
   this precisely."
7. If the question asks what changed, only describe changes supported by the
   retrieved amendment/current-source text. Do not guess.
8. Never silently use outside websites, memory, or an older version.
9. Do not provide instructions for committing cybercrime, evading law
   enforcement, bypassing security, or exploiting systems.
10. Use citations such as [Official PDF p. X, Section Y] only when supported.
11. If the user asks for the "latest PECA", "current PECA", "latest version",
    or "current version", do NOT answer "PECA 2016" alone. If the source
    metadata identifies the 2025 amendment, answer:
    "The current consolidated source represents the Prevention of Electronic
    Crimes Act, 2016 as amended by the Prevention of Electronic Crimes
    (Amendment) Act, 2025 (Act No. II of 2025)."
    Explain that the 2025 law is an amending Act to PECA 2016, not a separate
    replacement PECA.
12. If the source itself says it is under review or refers users to Gazette
    notifications, mention that limitation when relevant.
13. This is an informational research assistant, not legal advice.

Keep the answer focused and useful.
"""

    user_prompt = f"""
Question:
{question}

Retrieved legal source passages:
{context}

Answer the question using only the source passages above.
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
                f"**Source {i} — Official PDF page {result['page']} — "
                f"similarity {result['score']:.3f}**"
            )
            st.write(result["text"])
            st.divider()


def render_version_panel(kb, source_snapshot):
    info = kb["version_info"]

    st.subheader("📌 Knowledge Base Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("PDF pages", kb["page_count"])

    with col2:
        st.metric("Searchable chunks", kb["chunk_count"])

    with col3:
        st.metric("Source hash", kb["pdf_hash"][:12])

    st.caption(
        "Source: Official Pakistan Code • "
        f"Last checked: {source_snapshot['checked_at'].strftime('%Y-%m-%d %H:%M UTC')}"
    )

    if info["has_2025_amendment"]:
        st.success(
            "Current legal version represented by this source: "
            f"**{info['legal_version']}**"
        )
    else:
        st.warning(
            "No 2025 PECA amendment reference was detected in this downloaded "
            "source. Do not assume later amendments are included."
        )

    with st.expander("🔎 Detected source/version information"):
        st.write("**Detected amendment references:**")
        if info["amendment_refs"]:
            for ref in info["amendment_refs"]:
                st.write(f"- {ref}")
        else:
            st.write("No amendment reference was automatically detected.")

        if info["dates"]:
            st.write("**Dates detected in source:**")
            st.write(", ".join(info["dates"]))


# ============================================================
# UI
# ============================================================

st.title("⚖️ Pakistan Cyber Law AI Assistant")
st.caption(
    "Version-aware RAG assistant using the official Pakistan Code PECA source. "
    "Ask about offences, punishments, sections, online content, investigation "
    "powers and amendments."
)

st.info(
    "📌 **Current source target:** Prevention of Electronic Crimes Act, 2016 "
    "as amended by the **Prevention of Electronic Crimes (Amendment) Act, "
    "2025 (Act No. II of 2025)**. The 2025 Act amends PECA 2016; it does not "
    "replace the Act with a separately numbered 'PECA 2025'."
)

api_key = get_secret("GROQ_API_KEY")
configured_pdf_url = get_secret("CYBER_LAW_PDF_URL")
pdf_url = configured_pdf_url or DEFAULT_PDF_URL

# The old third-party host caused an expired SSL certificate and also points
# to the older 2016-only document. Force the official consolidated source
# unless the user deliberately supplies another non-old-host source.
if "lawsofpakistan.com" in pdf_url.lower():
    st.sidebar.warning(
        "Your CYBER_LAW_PDF_URL still points to the old Laws of Pakistan PDF. "
        "The app is using the official Pakistan Code consolidated source instead."
    )
    pdf_url = DEFAULT_PDF_URL

with st.sidebar:
    st.header("⚙️ Settings")

    model_name = st.selectbox(
        "Groq model",
        [DEFAULT_MODEL],
        index=0,
    )

    response_size = st.slider(
        "Response size",
        min_value=200,
        max_value=4000,
        value=1200,
        step=100,
    )

    temperature = st.slider(
        "Creativity",
        min_value=0.0,
        max_value=0.8,
        value=0.1,
        step=0.1,
        help="Keep this low for legal Q&A.",
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
    )

    show_sources = st.checkbox("Show retrieved evidence", value=True)
    show_debug = st.checkbox("Show technical details", value=False)

    st.divider()

    if st.button("🔄 Check for updates", use_container_width=True):
        try:
            with st.spinner("Checking the official Pakistan Code source..."):
                old_hash = st.session_state.get(
                    "source_snapshot", {}
                ).get("hash")

                snapshot = load_source(pdf_url, force_refresh=True)

                if old_hash and old_hash == snapshot["hash"]:
                    st.success("No PDF change detected.")
                elif old_hash:
                    st.success(
                        "A new PDF version/change was detected. "
                        "The FAISS knowledge base will be rebuilt."
                    )
                    st.cache_resource.clear()
                else:
                    st.success("Source downloaded successfully.")

                st.session_state.pop("knowledge_base", None)
                st.rerun()

        except Exception as exc:
            st.error(f"Update check failed: {exc}")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.markdown("### 🔄 Update system")
    st.write(
        "On startup the app downloads the configured official PDF and calculates "
        "a SHA-256 hash. If the PDF changes, a new FAISS index is created."
    )

    st.info(
        "Legal notice: this tool is for education and research. "
        "It is not a lawyer and is not a substitute for professional legal advice."
    )

if not api_key:
    st.error(
        "GROQ_API_KEY is missing. Add it to Streamlit Secrets."
    )
    st.stop()

# Download/check source and build version-specific knowledge base.
try:
    snapshot = load_source(pdf_url)

    if (
        "knowledge_base" not in st.session_state
        or st.session_state.knowledge_base.get("pdf_hash") != snapshot["hash"]
    ):
        with st.spinner(
            "Building the version-aware FAISS knowledge base from the official source..."
        ):
            st.session_state.knowledge_base = build_knowledge_base(
                pdf_url,
                snapshot["hash"],
                snapshot["bytes"],
            )

    kb = st.session_state.knowledge_base

except Exception as exc:
    st.error(f"Knowledge-base startup failed: {exc}")
    st.info(
        "The question interface cannot search the law until the official PDF "
        "is downloaded successfully. Use 'Check for updates' after fixing the source."
    )
    st.stop()

render_version_panel(kb, snapshot)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            if show_sources:
                render_sources(message["sources"])

question = st.chat_input(
    "Ask a question about Pakistan cyber law, a section, punishment, or amendment..."
)

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the current indexed legal source..."):
            try:
                results = retrieve(
                    question,
                    kb,
                    top_k=top_k,
                    min_score=min_score,
                )

                if not results:
                    response = (
                        "The current indexed source does not provide enough "
                        "retrievable information to answer this question precisely."
                    )
                else:
                    response = answer_with_groq(
                        question=question,
                        results=results,
                        api_key=api_key,
                        model_name=model_name,
                        max_tokens=response_size,
                        temperature=temperature,
                        kb=kb,
                    )

                st.markdown(response)

                if show_sources:
                    render_sources(results)

                if show_debug:
                    st.caption(
                        f"PDF hash: {kb['pdf_hash'][:16]} | "
                        f"Retrieved: {len(results)} passages | "
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
                error_message = f"I could not generate the answer. Technical error: {exc}"
                st.error(error_message)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )
