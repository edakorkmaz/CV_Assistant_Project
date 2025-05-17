from app.rag import extract_text_from_pdf, split_text, create_embeddings, build_faiss_index, run_agent_flexible
from app.utils import load_cohere_api
import os

# Cohere client'ı hazırla
co = load_cohere_api()

# PDF'lerden metni çıkar
cv_text = extract_text_from_pdf("cv.pdf")
job_text = extract_text_from_pdf("job_description.pdf")

# Metinleri böl
cv_chunks = split_text(cv_text, group_size=7)
job_chunks = split_text(job_text, group_size=5)

# İş ilanındaki tekrar eden parçaları temizle
job_chunks = list(dict.fromkeys(job_chunks))

# Embedding oluştur
cv_embeddings = create_embeddings(cv_chunks, co)
job_embeddings = create_embeddings(job_chunks, co)

# FAISS index oluştur
cv_index = build_faiss_index(cv_embeddings)
job_index = build_faiss_index(job_embeddings)

# Kullanıcıdan soru al
query = input("Ask a question (e.g., Are my projects sufficient?): ")

# Agent'i çağır ve sonucu yazdır
answer = run_agent_flexible(query, co, cv_index, cv_chunks, job_index, job_chunks)
print("\n Agent's response:\n")
print(answer)
