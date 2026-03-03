import anthropic
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()

# Load embedding model
print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Anthropic client
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def build_vector_store(chunks):
    print("Building vector store...")
    embeddings = embedder.encode(chunks)
    embeddings = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def get_relevant_chunks(question, chunks, index, top_k=3):
    question_embedding = embedder.encode([question])
    question_embedding = np.array(question_embedding).astype('float32')
    distances, indices = index.search(question_embedding, top_k)
    return [chunks[i] for i in indices[0]]

def ask_question(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"Based on the following context, answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"
        }]
    )
    return message.content[0].text

# MAIN
pdf_path = input("Enter PDF file path: ")
print("Processing PDF...")
text = extract_text_from_pdf(pdf_path)
chunks = chunk_text(text)
index, embeddings = build_vector_store(chunks)
print(f"PDF processed! {len(chunks)} chunks created.")
print("\nYou can now ask questions! Type 'quit' to exit.\n")

while True:
    question = input("Your question: ")
    if question.lower() == 'quit':
        break
    relevant_chunks = get_relevant_chunks(question, chunks, index)
    answer = ask_question(question, relevant_chunks)
    print(f"\nAnswer: {answer}\n")