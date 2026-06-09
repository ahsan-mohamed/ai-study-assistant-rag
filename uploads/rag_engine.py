import os
import fitz  # PyMuPDF
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Load models once when app starts ──
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.Client()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_text_from_pdf(pdf_path):
    """Read PDF and return all text as a string."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=500, overlap=50):
    """Break text into overlapping chunks of ~500 characters."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def build_vector_store(pdf_path, collection_name="documents"):
    """Extract PDF text, chunk it, embed it, store in ChromaDB."""
    # Delete old collection if exists
    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass

    collection = chroma_client.create_collection(collection_name)

    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text)

    # Convert chunks to embeddings (numbers)
    embeddings = embedding_model.encode(chunks).tolist()

    # Store in ChromaDB
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )
    return len(chunks)

def answer_question(question, collection_name="documents"):
    """Find relevant chunks and ask Groq LLM to answer."""
    collection = chroma_client.get_collection(collection_name)

    # Convert question to embedding and search
    question_embedding = embedding_model.encode([question]).tolist()
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=3  # Get top 3 most relevant chunks
    )

    # Build context from retrieved chunks
    context = "\n\n".join(results["documents"][0])

    # Ask Groq LLM with context
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful study assistant. Answer questions based ONLY on the provided context. If the answer isn't in the context, say 'I couldn't find that in the document.'"
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )
    return response.choices[0].message.content