# ⚖️ CyberlawGPT

A beginner-friendly **Retrieval-Augmented Generation (RAG)** application for asking questions about the Pakistan cyber-law PDF.

The application uses:

- **Python**
- **Streamlit**
- **FAISS** for vector search
- **Sentence Transformers** (`all-MiniLM-L6-v2`) for embeddings
- **PyMuPDF** for PDF text extraction
- **Groq**
- **`openai/gpt-oss-20b`** for the final answer

The source PDF used to design this application is the provided **Prevention of Electronic Crimes Act, 2016** document.

## Important source limitation

This application is deliberately grounded in the PDF configured through `CYBER_LAW_PDF_URL`.

It should **not** be treated as a complete or automatically up-to-date statement of Pakistani cyber law. If the PDF is an older version, the app will answer from that version.

For real legal matters, consult a qualified lawyer or the appropriate official authority.

---

# 1. Project contains only 3 files

```text
cyber-law-rag/
│
├── app.py
├── requirements.txt
└── readme.md
```

No FAISS index needs to be committed to GitHub.

The app downloads the PDF and creates the embeddings + FAISS index automatically when the application starts.

---

# 2. How the RAG pipeline works

```text
Cyber-law PDF
     ↓
Download on startup
     ↓
PyMuPDF extracts text page-by-page
     ↓
Text is divided into overlapping chunks
     ↓
Sentence Transformer creates embeddings
     ↓
FAISS stores the vectors
     ↓
User asks a question
     ↓
Question embedding
     ↓
FAISS semantic search
     ↓
Top relevant legal passages
     ↓
Groq GPT-OSS 20B
     ↓
Answer with PDF page/section references
```

The page information is preserved so the user can inspect the retrieved legal evidence.

---

# 3. Configure the PDF URL

Because the requested GitHub repository contains only three files, the PDF itself is **not committed to the repository**.

You need a publicly accessible **direct PDF download URL** for the exact cyber-law PDF you want to use.

Set:

```text
CYBER_LAW_PDF_URL = "YOUR_DIRECT_PUBLIC_PDF_URL"
```

The URL should directly return a PDF file, not an HTML webpage.

---

# 4. Streamlit Cloud deployment

Streamlit Community Cloud supports secrets outside your GitHub repository.

## Step 1 — Create GitHub repository

Create a repository and upload:

```text
app.py
requirements.txt
readme.md
```

## Step 2 — Deploy

On Streamlit Community Cloud:

1. Create a new app.
2. Select your GitHub repository.
3. Select `app.py` as the main file.
4. Open **Advanced settings**.
5. Add the following secrets:

```toml
GROQ_API_KEY = "your_groq_api_key"
CYBER_LAW_PDF_URL = "your_direct_public_pdf_url"
```

Do **not** put the Groq API key directly inside `app.py`.

## Step 3 — Deploy

The application will:

1. install the packages,
2. download the PDF,
3. extract the text,
4. load the embedding model,
5. create the FAISS index,
6. start the cyber-law assistant.

Streamlit recommends storing secrets in its Secrets management rather than committing them to GitHub.

---

# 5. Run on Google Colab

The same application can run in Colab.

## Install

```bash
!pip install -r requirements.txt
```

Set the environment variables:

```python
import os

os.environ["GROQ_API_KEY"] = "your_groq_api_key"
os.environ["CYBER_LAW_PDF_URL"] = "your_direct_public_pdf_url"
```

Then run:

```bash
!streamlit run app.py &>/content/streamlit.log &
```

For a public Colab demo, use your preferred tunneling method such as Cloudflare Tunnel or another supported tunnel.

---

# 6. Local Windows / VS Code

Open the project folder in VS Code.

Install dependencies:

```bash
pip install -r requirements.txt
```

Set environment variables in PowerShell:

```powershell
$env:GROQ_API_KEY="your_groq_api_key"
$env:CYBER_LAW_PDF_URL="your_direct_public_pdf_url"
```

Run:

```bash
streamlit run app.py
```

---

# 7. Groq model

The application uses:

```text
openai/gpt-oss-20b
```

This model is available through Groq and supports text generation and reasoning.

The app includes a **Response size** control. It controls the requested maximum completion tokens.

For legal Q&A, the default settings intentionally favor:

- low temperature,
- retrieval from the source PDF,
- explicit source evidence,
- conservative answers when evidence is insufficient.

---

# 8. Features

## Legal question answering

Examples:

```text
What is unauthorized access?

What is the punishment for unauthorized access under the Act?

What does Section 4 cover?

What is cyber stalking?

What is spoofing?

What is electronic fraud?

What is malicious code?

What are the provisions for unauthorized interception?

What is cyber terrorism?

What powers does an authorized officer have?

How is traffic data retained?

What does the Act say about search and seizure?

What is unlawful online content?

Can compensation be awarded to a victim?
```

The application retrieves the most relevant passages before generating the answer.

## Retrieval controls

The sidebar provides:

- Response size
- Temperature
- Number of retrieved passages
- Minimum similarity
- Show retrieved evidence
- Technical details
- Clear chat

---

# 9. Why FAISS is used

FAISS provides fast similarity search over the embedding vectors.

The application uses normalized embeddings and FAISS inner-product search, which is equivalent to cosine similarity for normalized vectors.

No external vector database is required.

---

# 10. Why Sentence Transformers is used

The application uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

This creates compact semantic embeddings that are suitable for semantic search.

The model is downloaded automatically the first time the application needs it.

---

# 11. Legal safety design

The prompt intentionally tells the language model:

- use only retrieved legal passages for substantive claims,
- identify sections when supported,
- do not invent punishments,
- distinguish maximum penalties from mandatory penalties,
- say when the PDF does not provide enough information,
- do not provide instructions for committing cybercrime,
- explain that the application is informational rather than legal advice.

This is important because an LLM should not be allowed to freely invent Pakistani legal provisions.

---

# 12. Example answer style

For a question such as:

```text
What is cyber stalking?
```

the system retrieves the relevant provision and produces an answer based on that evidence, with references such as:

```text
Under the provided PDF, cyber stalking covers specified repeated
contact, monitoring, watching/spying, or distributing a person's
photograph/video without consent in circumstances described by the Act.

Relevant provision: Section 21.
Source: PDF page 11.
```

The exact answer is generated from the retrieved passages rather than from a manually written database.

---

# 13. Troubleshooting

## Error: GROQ_API_KEY is missing

Add:

```toml
GROQ_API_KEY = "your_key"
```

to Streamlit Secrets.

## Error: CYBER_LAW_PDF_URL is missing

Add:

```toml
CYBER_LAW_PDF_URL = "https://example.com/file.pdf"
```

Use a direct PDF URL.

## Error: URL did not return a PDF

The URL probably points to a webpage instead of the PDF itself.

Use the actual PDF download URL.

## Error while installing FAISS

Make sure your deployment is using a supported Python version. Streamlit Community Cloud currently defaults to Python 3.12, and the package versions in `requirements.txt` are selected with that environment in mind.

## Slow first startup

This is normal.

The first startup can take time because the app needs to:

1. download the PDF,
2. download the Sentence Transformer model,
3. extract PDF text,
4. generate embeddings,
5. build the FAISS index.

Streamlit caching prevents the knowledge base from being rebuilt on every interaction during the same app process.

---

# 14. Important limitation

This is a **RAG research/education application**, not a legal decision system.

The quality of the answer depends on:

- the configured PDF,
- PDF text extraction quality,
- chunking,
- semantic retrieval,
- the retrieved evidence,
- and the language model.

If the required provision is not present in the configured PDF or is not retrieved, the application should say that the available source does not provide enough information instead of inventing an answer.

---

# 15. Final deployment checklist

Before deploying:

```text
[ ] app.py uploaded
[ ] requirements.txt uploaded
[ ] readme.md uploaded
[ ] Groq API key added to Streamlit Secrets
[ ] Direct PDF URL added to Streamlit Secrets
[ ] PDF URL opens/downloads a real PDF
[ ] Streamlit app starts successfully
[ ] Test Section 3
[ ] Test Section 21
[ ] Test spoofing
[ ] Test electronic fraud
[ ] Test search/seizure
[ ] Check retrieved evidence before relying on an answer
```

## Streamlit Secrets

```toml
GROQ_API_KEY = "your_groq_api_key"
CYBER_LAW_PDF_URL = "your_direct_public_pdf_url"
```

Never commit your Groq API key to GitHub.
