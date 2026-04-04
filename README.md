# RAG PDF Q&A Pipeline

A Retrieval-Augmented Generation (RAG) system that:
- Extracts text from PDFs using PyMuPDF
- Chunks text into overlapping sentence windows
- Generates embeddings with sentence-transformers
- Indexes embeddings with FAISS for fast similarity search
- Reranks retrieved chunks using a cross-encoder model
- Answers user questions using Ollama (llama3) based on retrieved context

## Project Structure

- `main.py` - Main pipeline that accepts PDF file paths as command-line arguments
- `utilites/file_operations.py` - PDF text extraction using PyMuPDF (fitz)
- `utilites/rag_operations.py` - Sentence-based chunking, embedding generation, FAISS indexing, similarity search, and chunk reranking
- `utilites/qa.py` - Answer generation using Ollama with llama3 model
- `data/` - Sample PDF files for demonstration

## Requirements

- Python 3.12+
- Ollama running locally on `http://localhost:11434` with the `llama3` model available

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
```

> Notes:
> - Dependencies are specified in `pyproject.toml`
> - NLTK tokenizer data (`punkt_tab`) is downloaded automatically at runtime by `utilites/rag_operations.py`
> - Sentence-transformers and cross-encoder models are downloaded on first use

## Environment Setup

Ensure Ollama is running locally:

```powershell
ollama serve
```

In another terminal, pull the llama3 model if not already available:

```powershell
ollama pull llama3
```

No additional `.env` file is required for this version.

## Run

Run the pipeline with one or more PDF file paths as arguments:

```powershell
python .\main.py ".\data\Python_Tutorial_EDIT.pdf"
```

Or process multiple PDFs:

```powershell
python .\main.py ".\data\Python_Tutorial_EDIT.pdf" ".\data\Udbhav_Full_Stack_Resume.pdf"
```

The script will process each file sequentially and present an interactive prompt for each PDF where you can ask questions.

## Pipeline Overview

For each PDF file provided:

1. **Extract** - Extracts all text from the PDF using PyMuPDF
2. **Chunk** - Splits text into sentence-based chunks (`chunk_size=5`, `overlap=1`)
3. **Embed** - Generates embeddings for each chunk using `all-MiniLM-L6-v2` model from sentence-transformers
4. **Index** - Builds a FAISS L2 index for fast similarity search
5. **Query Loop** - For each user query:
   - Generates embedding for the query
   - Retrieves top-10 similar chunks using FAISS
   - Reranks the top-10 chunks using `cross-encoder/ms-marco-MiniLM-L-6-v2`
   - Selects top-5 reranked chunks
   - Generates answer using Ollama (llama3) with the context and query

## Troubleshooting

- **Connection refused on port 11434:**
  - Ensure Ollama is running: `ollama serve` in a separate terminal
  - Check if llama3 model is available: `ollama pull llama3`

- **NLTK tokenizer errors:**
  - Rerun the script; it automatically downloads `punkt_tab` on first use

- **Embedding model download errors:**
  - Check internet connection; sentence-transformers downloads models on first use
  - Models are cached locally after first download

- **FAISS import fails:**
  - Ensure `faiss-cpu` is installed: `pip install faiss-cpu>=1.13.2`

- **Memory errors with large PDFs:**
  - The entire PDF is processed in memory; try with smaller PDFs first
  - Adjust `chunk_size` and `overlap` in `rag_operations.py` if needed

## Limitations

- Script-style interface (not a packaged service/API)
- Interactive query loop per PDF (one file at a time)
- FAISS index rebuilt on every run (no persistence)
- Entire PDF must fit in memory
- Fixed chunking parameters and model selection

## Future Improvements

- Batch query input from file instead of interactive loop
- Persist FAISS index to disk for faster subsequent runs
- Add tests for chunking, retrieval quality, and answer accuracy
- Improve answer source attribution with page/chunk references
- Support for other LLM backends (Ollama alternatives, API-based models)
- Configurable chunking strategy and model selection
- Streaming responses for long answers
- Multi-document context aggregation
