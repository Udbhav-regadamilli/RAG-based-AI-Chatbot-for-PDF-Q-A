import fitz  # PyMuPDF
import pickle
import faiss
import os

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


def save_index(index, chunks, path):
    # Extract clean filename
    filename = os.path.basename(path)
    filename = os.path.splitext(filename)[0]

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