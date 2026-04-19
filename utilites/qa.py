import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"

OUTPUT_FORMAT = """
- Do NOT use any Markdown, formatting symbols, or special characters for emphasis.
- This includes but is not limited to:
  **bold**, *italics*, __underline__, backticks, hashtags, or bullet symbols.

- Use ONLY plain natural language.
- If emphasis is needed, achieve it through wording, not symbols.

- NEVER include characters like: *, _, `, # in your response unless absolutely necessary for meaning.

- All responses must sound natural.
"""


def generate_answer(query, retrieved_chunks):
    """
    Generate answer using Gemini
    """

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a strict assistant.

Answer ONLY using the exact information in the context.

Rules:
- Do NOT interpret similarities (e.g., "behaves like" ≠ "is").
- Do NOT convert analogies into facts.
- Only return facts explicitly stated.

If the answer is not explicitly stated, say:
"I could not find the answer in the document."

{OUTPUT_FORMAT}

Context:
{context}

Question:
{query}

Answer:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model":MODEL,
            "prompt":prompt,
            "stream":False
        }
    )

    return response.json()["response"]