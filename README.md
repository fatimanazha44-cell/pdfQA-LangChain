# RAG with LangChain

A Retrieval-Augmented Generation (RAG) system built with LangChain that lets you ask questions about any PDF document using a local LLM.

This project has two versions:

| File | Vector Store | Prompt Engineering | LLM |
|---|---|---|---|
| `app.py` | FAISS (local) | ❌ Basic — LLM answers freely | Groq (llama-3.1-8b-instant) |
| `add_pinecone_to_code.py` | Pinecone (cloud) | ✅ Strict — answers only from document | Groq (llama-3.1-8b-instant) |

---

## How It Works

1. Load a PDF file
2. Split it into chunks
3. Embed the chunks using `all-MiniLM-L6-v2`
4. Store embeddings in FAISS or Pinecone
5. Ask questions — the LLM answers using your document

---

## Project Structure

```
pdfQA-LangChain/
│
├── app.py                   # Version 1: FAISS + no custom prompt (basic)
├── add_pinecone_to_code.py  # Version 2: Pinecone + strict custom prompt (improved)
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/fatimanazha44-cell/pdfQA-LangChain.git
cd pdfQA-LangChain
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

```bash
cp .env.example .env
```

Fill in your `.env`:

```
GROQ_API_KEY=your_groq_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

> ⚠️ `PINECONE_API_KEY` only required for `add_pinecone_to_code.py`

---

## Usage

### Version 1 — FAISS (basic)

```bash
python app.py
```

- Enter your PDF path when prompted
- Ask questions — the LLM answers freely using its own knowledge + the document
- Type `quit` to exit

### Version 2 — Pinecone (improved)

```bash
python add_pinecone_to_code.py
```

- Enter your PDF path
- Choose whether to upload documents (`y`) or query existing index (`n`)
- Ask questions — the LLM answers strictly from the document only
- Type `quit` to exit

> Make sure you have a Pinecone index named `pdf-rag` created at [https://pinecone.io](https://pinecone.io)

---

## Key Concepts

### FAISS vs Pinecone

| | FAISS | Pinecone |
|---|---|---|
| Storage | Local (in-memory) | Cloud (persistent) |
| Setup | Zero config | Requires API key + index |
| Persistence | Lost on exit | Survives restarts |
| Scale | Small projects | Production-ready |

### Prompt Engineering — The Key Difference

`app.py` uses no custom prompt. The LLM can answer from its own training data, which means it may answer questions that have nothing to do with your PDF.

`add_pinecone_to_code.py` uses a strict custom prompt:

```
You are an AI assistant. Use ONLY the context below to answer.
If the answer is not in the context, say: "I don't know based on the document."
```

This prevents hallucination and keeps answers grounded strictly in your document. This is the correct approach for production RAG systems.

---

## Technologies

- [LangChain](https://www.langchain.com/)
- [Groq](https://groq.com/) — fast cloud LLM inference
- [FAISS](https://github.com/facebookresearch/faiss) — local vector store
- [Pinecone](https://www.pinecone.io/) — cloud vector store
- [HuggingFace Sentence Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — embeddings

---

## Part of a Learning Series

This is **Project 3** in a series of AI-integrated software engineering projects:

| # | Project | Focus |
|---|---|---|
| 1 | TV AI Summarizer | Multi-provider AI APIs (OpenAI, Groq, HuggingFace) |
| 2 | RAG without LangChain | Manual RAG pipeline from scratch |
| 3 | RAG with LangChain ← you are here | LangChain abstraction + vector stores + prompt engineering |
