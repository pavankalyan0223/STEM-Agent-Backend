from fastapi import FastAPI
from pydantic import BaseModel, Field
import requests
from config import OLLAMA_URL, MODEL_NAME, HTTP_TIMEOUT
from rag_setup import query_rag
from fastapi.middleware.cors import CORSMiddleware
from session_manager import load_sessions, save_sessions, list_sessions, get_session, delete_session

from summarizer import summarize_all_pdfs, summarize_pdf, SUMMARY_DIR
import glob, json, os
from fastapi import UploadFile, File
from typing import List
from rag_setup import index_pdfs
from pydantic import BaseModel
import re

app = FastAPI(title="STEM Tutor — Math & Physics (Hybrid Chat)")

session_store = load_sessions()

origins = [
    "http://localhost:5173",  
    "http://localhost:5174",  
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,         
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskPayload(BaseModel):
    mode: str = Field(pattern="^(math|physics)$")
    question: str
    session_id: str = "default"
    use_rag: bool = True

class AskDocPayload(BaseModel):
    pdf_name: str
    question: str

def is_technical_question(question: str) -> bool:
    """Detect if a question is technical and needs RAG context."""
    question_lower = question.lower().strip()
    
    # Casual greetings and non-technical phrases
    casual_phrases = [
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "thanks", "thank you", "bye", "goodbye", "see you", "how are you",
        "what's up", "how's it going"
    ]
    
    # Check if it's a casual greeting
    if any(question_lower.startswith(phrase) or question_lower == phrase for phrase in casual_phrases):
        return False
    
    # Check for technical keywords (math/physics terms)
    technical_keywords = [
        "solve", "calculate", "find", "derive", "prove", "equation", "formula",
        "theorem", "law", "force", "energy", "velocity", "acceleration", "integral",
        "derivative", "vector", "function", "graph", "plot", "physics",
        "mathematics", "math", "algebra", "calculus", "geometry", "trigonometry"
    ]
    
    return any(keyword in question_lower for keyword in technical_keywords)


def ask_mistral(session_id: str, mode: str, question: str, use_rag: bool):
    """Ask Mistral with optional RAG textbook context."""
    # Store the original question before any modifications
    original_question = question
    
    if session_id not in session_store:
        system_prompt = (
            f"You are a friendly and highly knowledgeable {mode} tutor. "
            "You can solve numerical or conceptual problems, explain theories, "
            "and also chat casually when the question is not technical. "
            "IMPORTANT: If the user greets you or asks a casual question, respond naturally and conversationally. "
            f"Only provide technical explanations when asked about {mode} topics. "
            "Keep answers accurate, patient, and conversational when appropriate."
        )
        session_store[session_id] = [{"role": "system", "content": system_prompt}]
        save_sessions(session_store)

    # Ask the question directly to the model
    session_store[session_id].append({"role": "user", "content": original_question})

    payload = {
        "model": MODEL_NAME,
        "messages": session_store[session_id],
        "stream": False
    }

    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=HTTP_TIMEOUT)
        res.raise_for_status()
        reply = res.json()["message"]["content"]
    except Exception as e:
        reply = f"Error contacting model: {e}"

    session_store[session_id].append({"role": "assistant", "content": reply})
    return reply


@app.post("/ask")
def ask(payload: AskPayload):
    """Main endpoint for hybrid math/physics + general chat."""
    answer = ask_mistral(
        session_id=payload.session_id,
        mode=payload.mode,
        question=payload.question,
        use_rag=payload.use_rag,
    )
    return {
        "session_id": payload.session_id,
        "mode": payload.mode,
        "question": payload.question,
        "tutor_reply": answer
    }

@app.get("/sessions")
def get_all_sessions():
    return {"sessions": list_sessions(session_store)}


@app.get("/session/{session_id}")
def get_single_session(session_id: str):
    return {"messages": get_session(session_store, session_id)}


@app.delete("/session/{session_id}")
def remove_session(session_id: str):
    if delete_session(session_store, session_id):
        return {"message": f"Session '{session_id}' deleted."}
    return {"error": f"Session '{session_id}' not found."}


@app.post("/reset_session")
def reset_session(session_id: str):
    if session_id in session_store:
        del session_store[session_id]
    return {"message": f"Session '{session_id}' cleared."}

@app.post("/summarize_all")
def summarize_all():
    summarize_all_pdfs("data/")
    return {"message": "All PDFs summarized successfully."}

@app.get("/summaries")
def list_summaries():
    files = glob.glob(os.path.join(SUMMARY_DIR, "*.json"))
    return {"summaries": [os.path.basename(f) for f in files]}

@app.get("/summary/{name}")
def get_summary(name: str):
    path = os.path.join(SUMMARY_DIR, name)
    if not os.path.exists(path):
        return {"error": "Summary not found."}
    with open(path, "r") as f:
        data = json.load(f)
    return {"summary": data}

@app.post("/upload_pdfs")
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """
    Upload one or more PDFs. Each will be saved in data/, then automatically
    indexed into the RAG database and summarized.
    """
    upload_dir = "data/"
    os.makedirs(upload_dir, exist_ok=True)

    saved_files = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue
        file_path = os.path.join(upload_dir, file.filename)
        with open(file_path, "wb") as f:
            f.write(await file.read())
        saved_files.append(file.filename)

    if not saved_files:
        return {"error": "No valid PDF files uploaded."}

    try:
        index_pdfs()
        summarize_all_pdfs(upload_dir)
    except Exception as e:
        return {"error": f"Processing failed: {e}"}

    return {
        "message": "PDFs uploaded, indexed, and summarized successfully.",
        "files": saved_files
    }

@app.post("/ask_doc")
def ask_doc(payload: AskDocPayload):
    """Chat with a specific uploaded document only."""
    from rag_setup import collection, embedder
    q_emb = embedder.encode(payload.question).tolist()

    results = collection.query(
        query_embeddings=[q_emb],
        n_results=3,
        where={"source": payload.pdf_name}  # restrict to chosen file
    )

    context = "\n\n".join(results["documents"][0])
    prompt = (
        f"You are a tutor who must answer only using '{payload.pdf_name}'.\n\n"
        f"Relevant excerpts:\n{context}\n\n"
        f"Question: {payload.question}"
    )

    payload_llm = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "Answer only using the provided document."},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    try:
        res = requests.post(OLLAMA_URL, json=payload_llm, timeout=HTTP_TIMEOUT)
        res.raise_for_status()
        answer = res.json()["message"]["content"]
    except Exception as e:
        answer = f"Error contacting model: {e}"

    return {"document": payload.pdf_name, "question": payload.question, "answer": answer}

@app.post("/hybrid_search")
def hybrid_search(payload: dict):
    """Search both local PDFs (RAG) and the web."""
    query = payload.get("query", "")
    if not query.strip():
        return {"error": "Empty query."}

    from rag_setup import query_rag
    local_context = query_rag(query, top_k=3)

    try:
        import requests
        web_results = []
        res = requests.get(f"https://api.duckduckgo.com/?q={query}&format=json")
        data = res.json()
        for topic in data.get("RelatedTopics", [])[:3]:
            if "Text" in topic and "FirstURL" in topic:
                web_results.append({
                    "title": topic["Text"],
                    "url": topic["FirstURL"]
                })
    except Exception as e:
        web_results = [{"title": f"Web search failed: {e}", "url": ""}]

    return {
        "query": query,
        "local_context": local_context,
        "web_results": web_results
    }

@app.get("/")
def root():
    return {"message": "Hybrid Math/Physics Tutor — Ask /ask with mode='math' or 'physics'."}
