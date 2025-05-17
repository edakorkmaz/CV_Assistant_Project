import os
from dotenv import load_dotenv
import cohere

def load_cohere_api() -> cohere.Client:
    """Loads the Cohere API key and returns the client."""
    load_dotenv()
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY not found in .env file.")
    return cohere.Client(api_key)

def format_chunks(chunks):
    """Prints the list of chunks in a readable format."""
    for i, chunk in enumerate(chunks, 1):
        print(f"[{i}] {chunk[:120]}...\n")
