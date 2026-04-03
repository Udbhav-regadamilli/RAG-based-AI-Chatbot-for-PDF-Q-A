from utilites.file_operations import extract_text_from_pdf
from utilites.qa import generate_answer
from utilites.rag_operations import chunk_text_sentences, generate_embeddings, create_faiss_index, search_similar


def rag_pipeline(file_path):
    print("Rag pipeline")
    print("Extracting text from PDF...")
    pdf_text = extract_text_from_pdf(file_path)
    print("Generating chunks from the extracted text...")
    chunks = chunk_text_sentences(pdf_text)
    print("Generating embeddings...")
    embeddings = generate_embeddings(chunks)
    print("Creating FAISS index...")
    index, embeddings_np = create_faiss_index(embeddings)
    print("FAISS index created successfully.")
    print("------------------------------------------------")
    print("Testing similarity...")
    # Test query
    queries = ["What is this document about?", "What are the key topics covered?", "Who developed python?",
               "Summarize the whole document"]
    for query in queries:
        print(f"\n🔍 Query: {query}")
        query_embedding = generate_embeddings([query])[0]
        similar_chunks = search_similar(index, query_embedding, chunks, k=5)

        answer = generate_answer(query, similar_chunks)
        print("\n🤖 Answer:\n")
        print(answer)


if __name__ == "__main__":
    file_path = "./data/Python_Tutorial_EDIT.pdf"
    rag_pipeline(file_path)