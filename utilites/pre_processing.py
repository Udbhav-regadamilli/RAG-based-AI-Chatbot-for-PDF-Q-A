import os

from utilites.file_operations import extract_text_from_pdf, save_index, load_index, build_bm25, save_bm25, clean_text
from utilites.rag_operations import chunk_text_sentences, generate_embeddings, create_faiss_index


def is_data_preprocessed(file_path):
    try:
        index, chunks = load_index(file_path)
        if index is not None and chunks is not None:
            return True
        else:
            return False
    except Exception as e:
        print(f"Error checking preprocessed data: {e}")
        return False


def pre_processing(file_path):
    print(f"Processing: {file_path}")
    pdf_text = extract_text_from_pdf(file_path)
    text = clean_text(pdf_text)
    chunks = chunk_text_sentences(text)
    embeddings = generate_embeddings(chunks)
    index, embeddings_np = create_faiss_index(embeddings)
    # Extract clean filename
    filename = os.path.basename(file_path)
    filename = os.path.splitext(filename)[0]

    save_index(index, chunks, filename)
    bm25 = build_bm25(chunks)
    save_bm25(bm25, filename)
    print(f"Ready for queries on {file_path}")