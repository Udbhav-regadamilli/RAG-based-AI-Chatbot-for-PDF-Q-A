import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3"


def generate_answer(query, retrieved_chunks):
    """
    Generate answer using Gemini
    """

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
    AGNET
    You are a precise assistant. Help the user answering the question based on the provided context. 
    If you don't know the answer, say you don't know. Do not make up answers.

    Context:
    {context}

    Question:
    {query}
    
    -------------------
    NOTE:
    Always follow these rules: 
    1. Do not make up the answers
    2. Answer the Question only based on the provided Context.
    3. If the answer is not present in the context, say "Sorry, I don't know the answer to that question based 
        on the provided information."
    4. Do not answer the question if the answer contains any sensitive or harmful words.
    5. Filter out or mask the foul language words to maintain integrity.
    6. Do not generate anything new from the context, unless the question explicitly asks for it. Stick to the information provided in the context.
    7. If the question is irrelevant to the context, only say:
        Answer: "Sorry, I don't know the answer to that question based on the provided information."
        Reason: "Mentioned in prompt",
        Source: "PROMPT"
    
    
    --------------------
    OUTPUT TEXT FORMAT:
    1. Answer: <Your answer here>
    2. Reason: <Your reasoning here>
    3. Source: <The source from the context that you used to answer the question>
    
    - Do NOT use any Markdown, formatting symbols, or special characters for emphasis.
    - This includes but is not limited to:
      **bold**, *italics*, __underline__, backticks, hashtags, or bullet symbols.
    
    - Use ONLY plain natural language.
    - If emphasis is needed, achieve it through wording, not symbols.
    
    - NEVER include characters like: *, _, `, # in your response unless absolutely necessary for meaning.
    
    - All responses must sound natural.
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