import fitz  # PyMuPDF

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