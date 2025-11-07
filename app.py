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
    allow_origins=origins,          # or ["*"] for all origins (useful during dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskPayload(BaseModel):
    mode: str = Field(pattern="^(math|physics)$")
    question: str
    session_id: str = "default"
    use_rag: bool = True


def ask_mistral(session_id: str, mode: str, question: str, use_rag: bool):
    """Ask Mistral with optional RAG textbook context."""
    if session_id not in session_store:
        system_prompt = (
            f"You are a friendly and highly knowledgeable {mode} tutor. "
            "You can solve numerical or conceptual problems, explain theories, "
            "and also chat casually when the question is not technical. "
            "Keep answers accurate, patient, and conversational when appropriate."
        )
        session_store[session_id] = [{"role": "system", "content": system_prompt}]
        save_sessions(session_store)


    # Retrieve context from textbooks if enabled
    context = ""
    if use_rag:
        context = query_rag(question, top_k=3)
        if context:
            question = (
                f"Here are some textbook excerpts for reference:\n{context}\n\n"
                f"Now respond naturally and helpfully to this query:\n{question}"
            )

    session_store[session_id].append({"role": "user", "content": question})

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


@app.get("/")
def root():
    return {"message": "Hybrid Math/Physics Tutor — Ask /ask with mode='math' or 'physics'."}
