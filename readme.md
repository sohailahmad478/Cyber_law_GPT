⚖️ Pakistan Cyber Law AI Assistant — Version-Aware RAG
A beginner-friendly Retrieval-Augmented Generation (RAG) application for asking questions about Pakistan's cyber law using the official Pakistan Code source.

The project uses exactly 3 files:

cyber-law-rag/
├── app.py
├── requirements.txt
└── readme.md
🚀 Main improvement
The earlier version used a fixed PDF and could become outdated.

This upgraded version is automatic-update and version-aware:

Official Pakistan Code PDF
          ↓
Download source
          ↓
SHA-256 hash
          ↓
Detect amendment/version clues
          ↓
If PDF changed → rebuild FAISS
          ↓
PyMuPDF
          ↓
Chunking
          ↓
Sentence Transformers
          ↓
FAISS
          ↓
User question
          ↓
Relevant legal passages
          ↓
Groq GPT-OSS 20B
          ↓
Answer + source evidence
The current official Pakistan Code PDF contains references to the Prevention of Electronic Crimes (Amendment) Act, 2025, including an inserted definition and savings provisions. The app does not invent these changes; it reads the downloaded official source and detects amendment references from its text.

Important: automatic update means the app can detect when the configured PDF changes. It does not guarantee that every Gazette notification, court decision, regulation, or future legal development is represented in that PDF.

1. Features
🔄 Automatic source checking
When the app starts, it downloads the configured official PDF and calculates a SHA-256 hash.

Example:

PDF hash: a81f29c3e1ab...
If the PDF changes, its hash changes.

The app then rebuilds:

PDF → chunks → embeddings → FAISS
🔄 Manual "Check for updates"
The sidebar contains:

🔄 Check for updates
Click it when you want to force a fresh download.

If nothing changed:

No PDF change detected.
If the PDF changed:

A new PDF version/change was detected.
The FAISS knowledge base will be rebuilt.
📌 Version-aware status
The application shows:

PDF page count

searchable chunk count

source hash

last checked time

detected amendment references

whether a 2025 PECA amendment reference was found

📚 Evidence
Each answer can show the retrieved official PDF passages and page numbers.

🧠 RAG
The application uses:

PyMuPDF — PDF extraction

Sentence Transformers — embeddings

FAISS — semantic search

Groq — final answer

openai/gpt-oss-20b — language model

Streamlit — user interface

2. Official source
Default source:

https://www.pakistancode.gov.pk/pdffiles/administrator6a061efe0ed5bd153fa8b79b8eb4cba7.pdf
The application uses this as the default.

You can override it with:

CYBER_LAW_PDF_URL = "YOUR_DIRECT_PUBLIC_PDF_URL"
The URL must return a PDF directly.

Why Pakistan Code?
The application uses the official Pakistan Code source rather than the previously used third-party PDF host.

Pakistan Code's PECA page identifies the law as the Prevention of Electronic Crimes Act, 2016, and its consolidated PDF includes amendment material.

Pakistan Code also warns that its website content is for information purposes and may be under review, and that users may need to refer to the original Gazette notification when there is doubt. Therefore this application should be treated as a research/educational tool, not a definitive legal-advice system.

3. Streamlit Cloud deployment
Step 1 — GitHub
Create a GitHub repository and upload only:

app.py
requirements.txt
readme.md
Do not upload the PDF.

Do not upload your Groq API key.

Step 2 — Streamlit
Deploy the repository and select:

Main file: app.py
Step 3 — Secrets
Open Streamlit's Secrets settings and add:

GROQ_API_KEY = "your_groq_api_key"
You do not need CYBER_LAW_PDF_URL if you want to use the default official Pakistan Code source.

If you want to override it:

GROQ_API_KEY = "your_groq_api_key"
CYBER_LAW_PDF_URL = "https://example.com/your-direct-pdf.pdf"
4. Run locally in VS Code
Open the project folder:

cyber-law-rag/
├── app.py
├── requirements.txt
└── readme.md
Install:

pip install -r requirements.txt
Set the Groq key.

PowerShell:

$env:GROQ_API_KEY="your_groq_api_key"
Optional custom PDF:

$env:CYBER_LAW_PDF_URL="https://example.com/your-direct-pdf.pdf"
Run:

streamlit run app.py
5. Google Colab
Upload the three files to Colab.

Install:

!pip install -r requirements.txt
Set the API key:

import os
os.environ["GROQ_API_KEY"] = "your_groq_api_key"
Then run:

!streamlit run app.py &>/content/streamlit.log &
For a public Colab URL you can use a tunneling method such as Cloudflare Tunnel or another method available in your Colab environment.

For the easiest deployment, use GitHub + Streamlit Community Cloud.

6. What happens when PECA changes?
Suppose the official PDF changes.

Old source
SHA-256:
ABC123...
New source
SHA-256:
XYZ789...
The application sees:

ABC123 != XYZ789
Then it rebuilds:

New PDF
  ↓
New text
  ↓
New chunks
  ↓
New embeddings
  ↓
New FAISS index
Therefore the assistant does not continue using the old FAISS index for the new PDF.

7. Important limitation
This application detects changes to the configured PDF.

It does not automatically search the entire internet for every legal development.

For example, a new:

Gazette notification

amendment

ordinance

court judgment

regulation

government notification

may exist outside the configured PDF.

For a serious legal-research system, the next improvement would be a multi-source architecture:

Pakistan Code
      +
Gazette notifications
      +
Official amendment documents
      +
Official regulations
      +
Relevant court decisions
      ↓
Versioned legal knowledge base
      ↓
RAG
Do not claim that the system knows every current Pakistani cyber-law development unless those sources are actually included.

8. Questions you can test
Basic
What is the Prevention of Electronic Crimes Act 2016?
What is unauthorized access under PECA?
What punishment is provided for an offence against dignity?
Amendment-aware
Does the current source contain the 2025 PECA amendment?
What amendment references are present in the current source?
What is the definition of aspersion in the current PECA source?
What provisions were changed by the 2025 amendment?
For the last question, the assistant should answer only from retrieved amendment text. It should not guess.

Version test
What is the SHA-256 source version currently indexed?
The assistant UI, not the legal answer, is the authoritative place to see the actual source hash.

9. Legal safety
The system is intentionally designed not to:

invent sections

invent punishments

silently use an older law

claim unsupported amendments

provide instructions for committing cybercrime

pretend to be a lawyer

If the retrieved source is insufficient, the assistant should say so.

10. Project architecture
                 ┌─────────────────────────┐
                 │ Official Pakistan Code  │
                 │       PDF Source        │
                 └────────────┬────────────┘
                              │
                              ▼
                    Download + SHA-256
                              │
                              ▼
                    Version / amendment
                       metadata check
                              │
                              ▼
                         PyMuPDF
                              │
                              ▼
                         Chunking
                              │
                              ▼
                    Sentence Transformers
                              │
                              ▼
                           FAISS
                              │
             ┌────────────────┴───────────────┐
             │                                │
             ▼                                ▼
       User Question                    Source Evidence
             │
             ▼
        Semantic Search
             │
             ▼
       Relevant passages
             │
             ▼
        Groq GPT-OSS 20B
             │
             ▼
       Legal answer + citations
11. Final note
This is an educational/research RAG assistant, not legal advice.

For an actual legal case, verify the provision against the original Gazette notification and obtain advice from a qualified Pakistani lawyer or the appropriate official authority.

