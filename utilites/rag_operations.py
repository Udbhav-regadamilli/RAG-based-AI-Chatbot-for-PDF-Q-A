import nltk
from sentence_transformers import SentenceTransformer, CrossEncoder
import faiss
import numpy as np

nltk.download('punkt_tab')
# Load once
model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def chunk_text_sentences(text, chunk_size=5, overlap=1):
    """
    Chunk text based on sentences.

    Args:
        text (str): Input text
        chunk_size (int): Number of sentences per chunk
        overlap (int): Overlapping sentences

    Returns:
        list[str]: List of chunks
    """

    sentences = nltk.sent_tokenize(text)
    chunks = []
    start = 0

    while start < len(sentences):
        end = start + chunk_size
        chunk_sentences = sentences[start:end]
        chunk = " ".join(chunk_sentences)
        chunks.append(chunk)

        if end >= len(sentences):
            break

        start += chunk_size - overlap

    return chunks

def generate_embeddings(chunks):
    embeddings = model.encode(chunks)
    return embeddings


def create_faiss_index(embeddings):
    """
    Create FAISS index from embeddings
    """

    # Convert to numpy array
    embeddings_np = np.array(embeddings).astype("float32")

    dimension = embeddings_np.shape[1]

    # Create index
    index = faiss.IndexFlatL2(dimension)

    # Add embeddings
    index.add(embeddings_np)

    return index, embeddings_np


def ranking_chunks(query, chunks, top_k=2):
    """
    Rerank retrieved chunks based on relevance to query
    """

    pairs = [[query, chunk] for chunk in chunks]

    scores = reranker.predict(pairs)

    # Combine chunks with scores
    scored_chunks = list(zip(chunks, scores))

    # Sort by score (descending)
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    # Return top_k chunks
    return [chunk for chunk, _ in scored_chunks[:top_k]]


def search_similar(index, query_embedding, chunks, k=3):
    """
    Search top-k similar chunks
    """

    query_vector = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(query_vector, k)

    results = []
    for idx in indices[0]:
        results.append(chunks[idx])

    return results