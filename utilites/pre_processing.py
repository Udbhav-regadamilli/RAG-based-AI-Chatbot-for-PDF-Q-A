from utilites.file_operations import extract_text_from_pdf, save_index, load_index
from utilites.rag_operations import chunk_text_sentences, generate_embeddings, create_faiss_index

def is_data_preprocessed(file_path):
    try:
        index, chunks = load_index(file_path)
        if index is not None and chunks is not None:
            print(f"Data already preprocessed for {file_path}.")
            return True
        else:
            print(f"No existing index found for {file_path}. Preprocessing needed.")
            return False
    except Exception as e:
        print(f"Error checking preprocessed data: {e}")
        return False


def pre_processing(file_path):
    print(f"Processing: {file_path}")
    pdf_text = extract_text_from_pdf(file_path)
    chunks = chunk_text_sentences(pdf_text)
    embeddings = generate_embeddings(chunks)
    index, embeddings_np = create_faiss_index(embeddings)
    save_index(index, chunks, path=file_path)
    print(f"Ready for queries on {file_path}")