import nltk
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

nltk.download('punkt_tab')

from dotenv import load_dotenv

load_dotenv()

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
    model = SentenceTransformer("all-MiniLM-L6-v2")
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