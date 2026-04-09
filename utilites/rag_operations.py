import nltk
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import faiss
import numpy as np

from utilites.file_operations import load_index

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


def ranking_chunks(query, file_path, top_k=2):
    """
    Rerank retrieved chunks based on relevance to query
    """

    chunks = search_similar(query, file_path, top_k=10)

    pairs = [[query, chunk] for chunk in chunks]

    scores = reranker.predict(pairs)

    # Combine chunks with scores
    scored_chunks = list(zip(chunks, scores))

    # Sort by score (descending)
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    # Return top_k chunks
    return [chunk for chunk, _ in scored_chunks[:top_k]]


def search_similar(query, file_path, top_k=3):
    sentences = []

    index, chunks = load_index(file_path)

    # Collect all sentences
    for chunk in chunks:
        sentences.extend(chunk.split("."))

    # Remove empty
    sentences = [s.strip() for s in sentences if s.strip()]

    # Encode
    query_embedding = model.encode(query, convert_to_tensor=True)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    # Compute similarity
    scores = util.cos_sim(query_embedding, sentence_embeddings)[0]

    # Get top sentences
    top_indices = scores.argsort(descending=True)[:top_k]

    return [sentences[i] for i in top_indices]