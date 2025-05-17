import pdfplumber
from typing import List
import numpy as np
import faiss
import cohere

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from a PDF file."""
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text

def split_text(text: str, group_size=5) -> List[str]:
    """Split text line by line and combine lines into chunks of group_size."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    chunks = []
    for i in range(0, len(lines), group_size):
        chunk = " ".join(lines[i:i+group_size])
        chunks.append(chunk)
    return chunks

def create_embeddings(text_chunks: List[str], co: cohere.Client) -> np.ndarray:
    """Create embedding vectors for text chunks using Cohere."""
    embeddings = []
    batch_size = 20
    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i+batch_size]
        response = co.embed(texts=batch, model="large")
        embeddings.extend(response.embeddings)
    return np.array(embeddings).astype("float32")

def build_faiss_index(embeddings: np.ndarray):
    """Create a FAISS L2 index and add embeddings to it."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search_index(query: str, co: cohere.Client, index, text_chunks: List[str], top_k=3) -> List[str]:
    """Generate embedding for the query and return top_k closest matching text chunks.."""
    query_embedding = co.embed(texts=[query], model="large").embeddings[0]
    query_vector = np.array(query_embedding).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_vector, top_k)
    results = [text_chunks[idx] for idx in indices[0]]
    return results

def run_agent_flexible(query: str, co: cohere.Client, cv_index, cv_chunks, job_index, job_chunks):
    """Search the CV and job description based on the query and generate an answer using LLM."""
    cv_results = search_index(query, co, cv_index, cv_chunks)
    job_results = search_index(query, co, job_index, job_chunks)

    cv_context = "\n".join(cv_results)
    job_context = "\n".join(job_results)

    prompt = f"""
    You are an AI assistant helping a job candidate evaluate their CV against a job description.

    Candidate's CV:
    {cv_context}

    Job Description:
    {job_context}

    User question: {query}

    Please provide a helpful and detailed answer based on the information above.
    """

    response = co.generate(
        model="command",
        prompt=prompt,
        max_tokens=600
    )

    return response.generations[0].text.strip()
