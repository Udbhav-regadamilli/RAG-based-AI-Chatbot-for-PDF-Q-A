import os
import pickle

import faiss
import fitz  # PyMuPDF
from rank_bm25 import BM25Okapi


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts text from a PDF file.

    Args:
        file_path (str): Path to PDF file

    Returns:
        str: Extracted text
    """
    doc = fitz.open(file_path)
    text = ""

    for page_num, page in enumerate(doc):
        page_text = page.get_text()

        # Optional: debug page-wise extraction
        # print(f"Page {page_num}: {len(page_text)} chars")

        text += page_text + "\n"

    return text


def clean_text(text):
    lines = text.split("\n")

    cleaned = []
    for line in lines:
        if "Sphinx" in line:
            continue
        if "documentation" in line.lower():
            continue
        if len(line.strip()) < 20:
            continue

        cleaned.append(line)

    return "\n".join(cleaned)


def save_index(index, chunks, filename):
    # Ensure model directory exists
    os.makedirs("./model", exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, f"./model/{filename}.index")

    # Save chunks
    with open(f"./model/{filename}.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ Saved index for {filename}")


def load_index(path="faiss_index"):
    # Extract clean filename
    filename = os.path.basename(path)
    filename = os.path.splitext(filename)[0]
    try:
        index = faiss.read_index(f"./model/{filename}.index")

        with open(f"./model/{filename}.pkl", "rb") as f:
            chunks = pickle.load(f)

        return index, chunks
    except Exception as e:
        print(f"Error loading index: {e}")
        return None, None


def build_bm25(chunks):
    tokenized = [chunk.lower().split() for chunk in chunks]
    return BM25Okapi(tokenized)


def save_bm25(bm25, filename):
    with open(f"./model/{filename}_bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)


def load_bm25(path):
    import pickle
    import os

    filename = os.path.basename(path)
    filename = os.path.splitext(filename)[0]

    with open(f"./model/{filename}_bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)

    return bm25
