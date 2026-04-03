import time

from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


def generate_answer(query, retrieved_chunks):
    """
    Generate answer using Gemini
    """

    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
    AGNET
    You are a precise assistant. Help the user answering the question based on the provided context. 
    If you don't know the answer, say you don't know. Do not make up answers.
    Provide the answer with detailed reasoning and citing the sources from the context.

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
    
    --------------------
    OUTPUT TEXT FORMAT:
    1. Answer: <Your answer here>
    2. Reason: <Your reasoning here>
    3. Source: <The source from the context that you used to answer the question>
    
    - Do NOT use any Markdown, formatting symbols, or special characters for emphasis.
    - This includes but is not limited to:
      **bold**, *italics*, __underline__, backticks, hashtags, or bullet symbols.
    
    - Use ONLY plain natural language suitable for speech.
    - If emphasis is needed, achieve it through wording, not symbols.
    
    - NEVER include characters like: *, _, `, # in your response unless absolutely necessary for meaning.
    
    - All responses must sound natural when read aloud by a text-to-speech system.
    """

    models = ["gemini-flash-latest", "gemini-2.0-flash-lite"]

    for model_name in models:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            print(f"Error generating answer with model {model_name}: {e}")
            time.sleep(10)
            continue

    return "Sorry, I'm having trouble generating an answer right now. Please try again later."