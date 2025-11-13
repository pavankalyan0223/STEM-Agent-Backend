from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import requests
import glob
import json
import os
from typing import List, Optional, Dict

from config import OLLAMA_URL, MODEL_NAME, HTTP_TIMEOUT
from session_manager import load_sessions, save_sessions, list_sessions, get_session, delete_session
from summarizer import summarize_all_pdfs, SUMMARY_DIR
from rag_setup import index_pdfs
from mcp_client import initialize_mcp_client, get_mcp_client

# Agentic architecture imports
from agentic_core import CentralAgent, PlanningMode
from specialized_agents import AgentOrchestrator
from generation import Generator

# Global orchestrator and generator
orchestrator = AgentOrchestrator()
generator = Generator()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application lifespan events."""
    # Startup
    print("🤖 Agentic RAG System Initialized")
    print("   - Central Agent with Memory & Planning")
    print("   - Local Data Agent (RAG + MCP)")
    print("   - Search Engine Agent (Web Search + MCP)")
    print("   - Generation Component")
    yield
    # Shutdown (cleanup if needed)
    pass

app = FastAPI(
    title="Agentic STEM Tutor — Math & Physics (RAG with MCP)",
    lifespan=lifespan
)

session_store = load_sessions()

# Agent store: session_id -> CentralAgent
agent_store: Dict[str, CentralAgent] = {}

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
    planning_mode: Optional[str] = Field(default="react", pattern="^(react|cot)$")  # ReACT or CoT

class AskDocPayload(BaseModel):
    pdf_name: str
    question: str

def is_casual_greeting(question: str) -> bool:
    """Detect if a question is a casual greeting that doesn't need agent processing."""
    question_lower = question.lower().strip()
    
    casual_phrases = [
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "thanks", "thank you", "bye", "goodbye", "see you", "how are you",
        "what's up", "how's it going", "greetings", "hi there", "hello there"
    ]
    
    # Check if it's exactly a casual greeting or starts with one
    return any(
        question_lower == phrase or 
        question_lower.startswith(phrase + " ") or 
        question_lower.startswith(phrase + "!")
        for phrase in casual_phrases
    )

@app.post("/ask")
async def ask(payload: AskPayload):
    """
    Main agentic endpoint: Processes query through agentic system.
    Flow: Query → Central Agent (Planning) → Specialized Agents → Generation → Answer
    """
    session_id = payload.session_id
    
    # Check if it's a casual greeting - skip agent processing
    if is_casual_greeting(payload.question):
        # Simple friendly response for greetings
        greeting_responses = {
            "math": "Hello! I'm your friendly math tutor. I'm here to help you with mathematics - whether it's algebra, calculus, geometry, or any other math topic. What would you like to learn today?",
            "physics": "Hello! I'm your friendly physics tutor. I'm here to help you understand physics concepts - from mechanics and thermodynamics to electromagnetism and quantum physics. What would you like to explore?"
        }
        answer = greeting_responses.get(payload.mode, greeting_responses["math"])
        
        # Update session store
        if session_id not in session_store:
            system_prompt = (
                f"You are a friendly and highly knowledgeable {payload.mode} tutor. "
                "You use an agentic system with memory, planning, and specialized agents."
            )
            session_store[session_id] = [{"role": "system", "content": system_prompt}]
        
        session_store[session_id].append({"role": "user", "content": payload.question})
        session_store[session_id].append({"role": "assistant", "content": answer})
        save_sessions(session_store)
        
        return {
            "session_id": session_id,
            "mode": payload.mode,
            "question": payload.question,
            "tutor_reply": answer,
            "agentic_metadata": {
                "casual_greeting": True,
                "agents_skipped": True
            }
        }
    
    # Get or create central agent for this session
    if session_id not in agent_store:
        planning_mode = PlanningMode.REACT if payload.planning_mode == "react" else PlanningMode.COT
        agent_store[session_id] = CentralAgent(session_id, planning_mode)
    
    central_agent = agent_store[session_id]
    
    # Step 1: Central Agent processes query and creates plan
    agent_result = central_agent.process_query(payload.question, payload.mode)
    plan = agent_result["plan"]
    memory_context = agent_result["memory_context"]
    
    # Step 2: Execute plan using specialized agents
    execution_results = await orchestrator.execute_plan(plan, payload.question)
    agent_results = execution_results["agent_results"]
    
    # Step 3: Generate final answer using Generation component
    conversation_history = get_session(session_store, session_id)
    generation_result = generator.generate_answer(
        query=payload.question,
        agent_results=agent_results,
        memory_context=memory_context,
        mode=payload.mode,
        conversation_history=conversation_history
    )
    
    # Step 4: Add answer to memory and session
    answer = generation_result.get("answer", "No answer generated")
    central_agent.add_to_memory(
        f"Q: {payload.question}\nA: {answer}",
        is_important=generation_result.get("status") == "success"
    )
    
    # Update session store
    if session_id not in session_store:
        system_prompt = (
            f"You are a friendly and highly knowledgeable {payload.mode} tutor. "
            "You use an agentic system with memory, planning, and specialized agents."
        )
        session_store[session_id] = [{"role": "system", "content": system_prompt}]
    
    session_store[session_id].append({"role": "user", "content": payload.question})
    session_store[session_id].append({"role": "assistant", "content": answer})
    save_sessions(session_store)
    
    return {
        "session_id": session_id,
        "mode": payload.mode,
        "question": payload.question,
        "tutor_reply": answer,
        "agentic_metadata": {
            "plan": plan,
            "execution_summary": execution_results["execution_summary"],
            "generation_status": generation_result.get("status"),
            "context_used": generation_result.get("context_used", {}),
            "sources": generation_result.get("sources", {})
        }
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
    """Reset both session store and agent store for a session."""
    if session_id in session_store:
        del session_store[session_id]
    if session_id in agent_store:
        del agent_store[session_id]
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
async def hybrid_search(payload: dict):
    """
    Agentic hybrid search: Uses Local Data Agent and Search Engine Agent via MCP.
    """
    query = payload.get("query", "")
    if not query.strip():
        return {"error": "Empty query."}

    # Use specialized agents directly
    local_data_result = await orchestrator.local_data_agent.retrieve_local_context(query, top_k=3)
    search_result = await orchestrator.search_engine_agent.search_web(query, max_results=3)

    return {
        "query": query,
        "local_data_agent": local_data_result,
        "search_engine_agent": search_result,
        "mcp_enhanced": local_data_result.get("mcp_enhanced", False) or search_result.get("processed_summary") is not None
    }

@app.post("/configure_mcp")
async def configure_mcp(payload: dict):
    """Configure the MCP server connection."""
    server_command = payload.get("server_command")
    server_args = payload.get("server_args", [])
    
    if not server_command:
        return {"error": "server_command is required"}
    
    initialize_mcp_client(server_command, server_args)
    client = get_mcp_client()
    
    if client:
        connected = await client.connect()
        return {
            "status": "configured",
            "connected": connected,
            "server_command": server_command,
            "server_args": server_args
        }
    else:
        return {"error": "Failed to initialize MCP client"}

@app.get("/mcp_status")
async def mcp_status():
    """Get the current MCP client status."""
    client = get_mcp_client()
    if not client:
        return {
            "available": False,
            "status": "not_configured",
            "message": "MCP client not initialized. Use /configure_mcp to set it up."
        }
    
    return {
        "available": client.available,
        "connected": client.session is not None,
        "server_command": client.server_command,
        "server_args": client.server_args
    }

@app.get("/agent/{session_id}/memory")
async def get_agent_memory(session_id: str):
    """Get memory state for a specific agent session."""
    if session_id not in agent_store:
        return {"error": f"Agent for session {session_id} not found"}
    
    agent = agent_store[session_id]
    memory_summary = agent.get_memory_summary()
    
    return {
        "session_id": session_id,
        "memory": memory_summary
    }

@app.get("/agent/{session_id}/status")
async def get_agent_status(session_id: str):
    """Get status of agent for a session."""
    if session_id not in agent_store:
        return {
            "session_id": session_id,
            "status": "not_found",
            "message": "Agent not initialized for this session"
        }
    
    agent = agent_store[session_id]
    return {
        "session_id": session_id,
        "status": "active",
        "planning_mode": agent.planning.mode.value,
        "available_agents": agent.available_agents,
        "memory": agent.get_memory_summary()
    }

@app.post("/agent/{session_id}/clear_memory")
async def clear_agent_memory(session_id: str, memory_type: str = "short_term"):
    """Clear agent memory (short_term or long_term)."""
    if session_id not in agent_store:
        return {"error": f"Agent for session {session_id} not found"}
    
    agent = agent_store[session_id]
    if memory_type == "short_term":
        agent.memory.clear_short_term()
        return {"message": "Short-term memory cleared"}
    elif memory_type == "long_term":
        agent.memory.long_term = []
        agent.memory._save_long_term()
        return {"message": "Long-term memory cleared"}
    else:
        return {"error": "memory_type must be 'short_term' or 'long_term'"}

@app.get("/agents")
async def list_all_agents():
    """List all active agents."""
    return {
        "active_agents": list(agent_store.keys()),
        "total_count": len(agent_store),
        "orchestrator_agents": list(orchestrator.agents.keys())
    }

@app.get("/")
def root():
    return {
        "message": "Agentic STEM Tutor — Math & Physics (RAG with MCP)",
        "architecture": {
            "central_agent": "Memory (Short/Long-term) + Planning (ReACT/CoT)",
            "specialized_agents": [
                "Local Data Agent (RAG + MCP)",
                "Search Engine Agent (Web Search + MCP)"
            ],
            "generation": "Synthesis of all agent results",
            "endpoints": {
                "/ask": "Main agentic query endpoint (always uses both local documents and MCP for web search)",
                "/hybrid_search": "Agentic search using specialized agents",
                "/agent/{session_id}/memory": "View agent memory",
                "/agent/{session_id}/status": "View agent status",
                "/agents": "List all active agents"
            },
            "search_strategy": {
                "local_documents": "Always used via Local Data Agent",
                "web_search": "Handled by MCP server (falls back to DuckDuckGo if MCP not configured)"
            }
        }
    }
