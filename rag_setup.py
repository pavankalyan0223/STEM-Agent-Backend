import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader

# --- CONFIG ---
PDF_DIR = "data/"       # Folder where you keep your textbook PDFs
DB_DIR = "data/chroma_db"   # Where to store the local Chroma index
EMBED_MODEL = "all-MiniLM-L6-v2"

# --- SETUP ---
embedder = SentenceTransformer(EMBED_MODEL)
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(name="textbook_chunks")

# --- FUNCTIONS ---

def extract_text_from_pdf(pdf_path):
    """Extracts text from all pages of a PDF."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def chunk_text(text, source_name):
    """Split long text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return [
        {"text": chunk, "metadata": {"source": source_name, "length": len(chunk)}}
        for chunk in chunks
    ]


def index_pdfs():
    """Index all PDFs in the PDF_DIR into ChromaDB."""
    pdf_paths = glob.glob(os.path.join(PDF_DIR, "*.pdf"))
    if not pdf_paths:
        print(f"No PDFs found in {PDF_DIR}")
        return

    doc_id = 0
    for path in pdf_paths:
        print(f"Processing: {path}")
        text = extract_text_from_pdf(path)
        chunks = chunk_text(text, os.path.basename(path))
        print(f" -> {len(chunks)} chunks created")

        for ch in chunks:
            emb = embedder.encode(ch["text"]).tolist()
            collection.add(
                documents=[ch["text"]],
                metadatas=[ch["metadata"]],
                ids=[f"{os.path.basename(path)}_{doc_id}"]
            )
            doc_id += 1

    print(f"Indexed {doc_id} chunks from {len(pdf_paths)} PDFs.")
    print(f"Database stored at: {DB_DIR}")


def query_rag(question, top_k=3):
    """Search for relevant text chunks with improved retrieval."""
    q_emb = embedder.encode(question).tolist()
    results = collection.query(query_embeddings=[q_emb], n_results=top_k)

    print("\n🔍 Retrieved context:")
    formatted_contexts = []
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0]), 1):
        print(f" - Source: {meta['source']}")
        print(f"   Snippet: {doc[:150]}...\n")
        # Format context with source information for better traceability
        formatted_contexts.append(f"[Source: {meta['source']}]\n{doc}")

    # Return formatted context with source attribution
    return "\n\n---\n\n".join(formatted_contexts)


if __name__ == "__main__":
    index_pdfs()
