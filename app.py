from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
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
from research_graph import build_research_graph, get_research_graph

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
    print("Application start")
    yield
    # Shutdown (cleanup if needed)
    pass

app = FastAPI(
    title="Agentic STEM Expert — Math & Physics (RAG with MCP)",
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
            "math": "Hello! I'm your friendly math expert. I'm here to help you with mathematics - whether it's algebra, calculus, geometry, or any other math topic. What would you like to learn today?",
            "physics": "Hello! I'm your friendly physics expert. I'm here to help you understand physics concepts - from mechanics and thermodynamics to electromagnetism and quantum physics. What would you like to explore?"
        }
        answer = greeting_responses.get(payload.mode, greeting_responses["math"])
        
        # Update session store
        if session_id not in session_store:
            system_prompt = (
                f"You are a friendly and highly knowledgeable {payload.mode} expert. "
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
            "expert_reply": answer,
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
        f"You are an expert who must answer only using '{payload.pdf_name}'.\n\n"
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

class ResearchQuery(BaseModel):
    query: str
    max_results: Optional[int] = Field(default=20, ge=1, le=100)
    category: Optional[str] = Field(default=None)  # e.g., "math", "physics", "cs", etc.

class BuildGraphRequest(BaseModel):
    pdf_files: Optional[List[str]] = None

@app.post("/research")
async def fetch_research_papers(payload: ResearchQuery):
    """
    Fetch research papers from arXiv API based on query.
    Returns papers with titles, authors, abstracts, and links.
    Note: arXiv API uses HTTP (not HTTPS) and has rate limits.
    """
    try:
        query = payload.query
        max_results = payload.max_results
        category = payload.category
        
        # Build arXiv API query
        # arXiv API documentation: https://arxiv.org/help/api/user-manual
        # IMPORTANT: arXiv API uses HTTP, not HTTPS
        base_url = "http://export.arxiv.org/api/query"
        
        # Map category to arXiv categories
        category_map = {
            "math": "math",
            "physics": "physics",
            "cs": "cs",
            "quantum": "quant-ph",
            "ai": "cs.AI",
            "ml": "cs.LG"
        }
        
        # Build search query - use proper arXiv search syntax
        # For text search, use all: prefix or ti: for title, au: for author, etc.
        search_query = f"all:{query}"
        if category and category.lower() in category_map:
            search_query = f"cat:{category_map[category.lower()]} AND all:{query}"
        
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        
        # Add headers to identify the client (arXiv API best practice)
        headers = {
            "User-Agent": "STEM-Expert-Research/1.0 (contact: your-email@example.com)"
        }
        
        response = requests.get(base_url, params=params, headers=headers, timeout=HTTP_TIMEOUT)
        
        # Handle rate limiting
        if response.status_code == 429:
            return {
                "error": "Rate limit exceeded. Please wait a moment and try again. arXiv API allows 1 request per 3 seconds.",
                "papers": []
            }
        
        response.raise_for_status()
        
        # Parse XML response
        import xml.etree.ElementTree as ET
        root = ET.fromstring(response.content)
        
        # Namespace for arXiv XML
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        papers = []
        for entry in root.findall('atom:entry', ns):
            paper = {
                "id": entry.find('atom:id', ns).text if entry.find('atom:id', ns) is not None else "",
                "title": entry.find('atom:title', ns).text.strip() if entry.find('atom:title', ns) is not None else "No title",
                "summary": entry.find('atom:summary', ns).text.strip() if entry.find('atom:summary', ns) is not None else "No abstract",
                "published": entry.find('atom:published', ns).text if entry.find('atom:published', ns) is not None else "",
                "updated": entry.find('atom:updated', ns).text if entry.find('atom:updated', ns) is not None else "",
                "authors": [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns) if author.find('atom:name', ns) is not None],
                "link": entry.find('atom:id', ns).text if entry.find('atom:id', ns) is not None else "",
                "pdf_link": ""
            }
            
            # Get PDF link
            for link in entry.findall('atom:link', ns):
                if link.get('type') == 'application/pdf':
                    paper["pdf_link"] = link.get('href', '')
                    break
            
            # If no PDF link found, construct it from ID
            if not paper["pdf_link"] and paper["id"]:
                arxiv_id = paper["id"].split('/')[-1]
                paper["pdf_link"] = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            
            papers.append(paper)
        
        return {
            "papers": papers,
            "total": len(papers),
            "query": query,
            "category": category
        }
        
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Please try again with a simpler query.", "papers": []}
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            return {"error": "Rate limit exceeded. Please wait a few seconds and try again.", "papers": []}
        return {"error": f"HTTP error: {str(e)}", "papers": []}
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch research papers: {str(e)}", "papers": []}
    except Exception as e:
        return {"error": f"Error processing research papers: {str(e)}", "papers": []}

@app.get("/research-graph/pdfs")
async def list_pdfs():
    """List all available PDF files in the data directory."""
    import os
    pdf_dir = "data/"
    if not os.path.exists(pdf_dir):
        return {"pdfs": []}
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    return {"pdfs": pdf_files}

@app.post("/research-graph/build")
async def build_graph(request: BuildGraphRequest = BuildGraphRequest()):
    """
    Build a knowledge graph from selected research papers.
    If pdf_files is None or empty, uses all PDFs in data folder.
    Creates nodes for papers and topics, and edges showing relationships.
    Saves graph with filename based on paper names separated by '_'.
    """
    try:
        from research_graph import build_research_graph
        import os
        import re
        
        pdf_dir = "data/"
        
        pdf_files = request.pdf_files if request else None
        
        # Generate filename from paper names
        def generate_graph_filename(pdf_list):
            """Generate filename from PDF names, removing .pdf extension and joining with '_'"""
            if not pdf_list:
                return "all_papers"
            # Remove .pdf extension and sanitize filenames
            clean_names = []
            for pdf in pdf_list:
                name = pdf.replace('.pdf', '').replace('.PDF', '')
                # Remove invalid filename characters
                name = re.sub(r'[<>:"/\\|?*]', '_', name)
                clean_names.append(name)
            return '_'.join(clean_names)
        
        # If specific PDFs are provided, filter them
        if pdf_files and len(pdf_files) > 0:
            # Generate filename from selected PDFs
            graph_filename = generate_graph_filename(pdf_files)
            
            # Create a temporary directory with only selected PDFs
            import tempfile
            import shutil
            
            temp_dir = tempfile.mkdtemp()
            try:
                for pdf_file in pdf_files:
                    src_path = os.path.join(pdf_dir, pdf_file)
                    if os.path.exists(src_path):
                        shutil.copy2(src_path, os.path.join(temp_dir, pdf_file))
                
                graph = build_research_graph(temp_dir, graph_filename=graph_filename)
                # Add metadata about which PDFs were used
                if "metadata" in graph:
                    graph["metadata"]["pdf_files_used"] = pdf_files
                return graph
            finally:
                shutil.rmtree(temp_dir)
        else:
            # Use all PDFs
            all_pdfs = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
            graph_filename = generate_graph_filename(all_pdfs)
            graph = build_research_graph(pdf_dir, graph_filename=graph_filename)
            if "metadata" in graph:
                graph["metadata"]["pdf_files_used"] = all_pdfs
            return graph
    except Exception as e:
        return {"error": f"Error building research graph: {str(e)}"}

@app.get("/research-graph")
async def get_graph(graph_filename: Optional[str] = None):
    """
    Get a research graph by filename.
    If graph_filename is not provided, returns the default graph.
    """
    try:
        from research_graph import get_research_graph
        graph = get_research_graph(graph_filename=graph_filename)
        return graph
    except Exception as e:
        return {"error": f"Error loading research graph: {str(e)}"}


@app.delete("/research-graph/{graph_filename}")
async def delete_graph(graph_filename: str):
    """Delete a research graph by filename."""
    try:
        from research_graph import delete_research_graph
        import urllib.parse
        # Decode URL-encoded filename
        decoded_filename = urllib.parse.unquote(graph_filename)
        result = delete_research_graph(decoded_filename)
        return result
    except Exception as e:
        return {"error": f"Error deleting research graph: {str(e)}"}

@app.get("/research-graph/list")
async def list_graphs():
    """List all available research graphs with their associated papers."""
    import os
    import json
    
    graph_dir = "data/research_graphs"
    graphs = []
    
    if os.path.exists(graph_dir):
        graph_files = [f for f in os.listdir(graph_dir) if f.endswith(".json")]
        
        for graph_file in graph_files:
            graph_path = os.path.join(graph_dir, graph_file)
            try:
                with open(graph_path, "r", encoding="utf-8") as f:
                    graph_data = json.load(f)
                    
                    # Extract paper information
                    papers = []
                    if "nodes" in graph_data:
                        for node in graph_data["nodes"]:
                            if node.get("type") == "paper":
                                papers.append({
                                    "id": node.get("id"),
                                    "title": node.get("title") or node.get("label"),
                                    "filename": node.get("filename")
                                })
                    
                    graphs.append({
                        "filename": graph_file,
                        "name": graph_file.replace(".json", ""),
                        "papers": papers,
                        "papers_count": len(papers),
                        "topics_count": graph_data.get("metadata", {}).get("topics_count", 0),
                        "edges_count": graph_data.get("metadata", {}).get("edges_count", 0),
                        "created": os.path.getmtime(graph_path) if os.path.exists(graph_path) else None
                    })
            except Exception as e:
                print(f"Error reading graph {graph_file}: {e}")
                continue
    
    return {"graphs": graphs}

@app.get("/research-graph/pdf/{filename}")
async def get_pdf_file(filename: str):
    """Serve a PDF file directly."""
    import urllib.parse
    
    pdf_dir = "data/"
    decoded_filename = urllib.parse.unquote(filename)
    pdf_path = os.path.join(pdf_dir, decoded_filename)
    
    if not os.path.exists(pdf_path) or not decoded_filename.lower().endswith(".pdf"):
        return {"error": "PDF not found"}
    
    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=decoded_filename,
        headers={"Content-Disposition": f"inline; filename={decoded_filename}"}
    )

@app.get("/")
def root():
    return {
        "message": "Agentic STEM Expert — Math & Physics (RAG with MCP)",
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
