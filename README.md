# RAG PDF Q&A Pipeline

A simple Retrieval-Augmented Generation (RAG) demo that:

- extracts text from a PDF
- chunks the text into overlapping sentence windows
- builds vector embeddings
- indexes vectors with FAISS
- answers user-style questions using Gemini based on retrieved chunks

## Project Structure

- `main.py` - runs the full pipeline
- `utilites/file_operations.py` - PDF text extraction with PyMuPDF
- `utilites/rag_operations.py` - chunking, embeddings, FAISS indexing, similarity search
- `utilites/qa.py` - Gemini answer generation from retrieved context
- `data/` - sample PDFs

## Requirements

- Python 3.12+
- A valid `GOOGLE_API_KEY` for Gemini

## Install

This repository includes `pyproject.toml` and `uv.lock`.

### Option 1: using uv (recommended)

```powershell
uv sync
```

### Option 2: using pip

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -e .
pip install faiss-cpu python-dotenv
```

> Notes:
> - `faiss` and `python-dotenv` are used by the code and may need to be installed depending on your environment.
> - NLTK tokenizer data is downloaded automatically at runtime by `utilites/rag_operations.py`.

## Environment Setup

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

## Run

```powershell
python .\main.py
```

By default, the script reads:

- `./data/Python_Tutorial_EDIT.pdf`

To use another PDF, update `file_path` in `main.py`.

## What the pipeline does

1. Extracts PDF text.
2. Splits text into sentence chunks (`chunk_size=5`, `overlap=1`).
3. Generates embeddings with `all-MiniLM-L6-v2`.
4. Builds a FAISS L2 index.
5. Runs example queries and generates answers using Gemini.

## Troubleshooting

- If you get API/auth errors:
  - verify `GOOGLE_API_KEY` is set correctly
  - check network access
- If `faiss` import fails:
  - install `faiss-cpu`
- If tokenizer errors appear:
  - rerun once and allow NLTK download
- If model calls fail intermittently:
  - retry; the code already tries fallback Gemini models

## Limitations

- This is a script-style prototype (not yet a packaged CLI/service).
- Query list is hardcoded in `main.py`.
- No persistent vector store yet; index is rebuilt every run.

## Next Improvements

- Add CLI args for file path and query input
- Persist and reload FAISS index
- Add tests for chunking and retrieval quality
- Improve prompt template and source attribution

