# Agentic STEM Tutor — Math & Physics (RAG with MCP)

A smart tutoring system for math and physics that actually remembers what you talked about and can pull information from your documents and the web. Think of it as a tutor that gets smarter the more you use it.

## What This Is

So you know how regular chatbots forget everything after each conversation? This one doesn't. It's built with an "agentic" architecture - basically, there's a central brain that coordinates specialized helpers to find answers. One helper searches through your uploaded PDFs, another searches the web, and they all work together to give you comprehensive answers.

The cool part? It remembers important stuff across sessions. Ask it about derivatives today, and next week it'll still remember what you struggled with. It's like having a tutor that actually pays attention.

## What It Can Do

### The Agent System

The system has a central agent that acts like a coordinator. When you ask a question, it figures out what needs to happen and delegates to specialized agents:

- **Central Agent**: The brain of the operation. It remembers things, plans what to do, and coordinates everything.
- **Local Data Agent**: This one digs through your uploaded PDFs and documents. Upload your math textbook, and it can answer questions based on what's in there.
- **Search Engine Agent**: When your local documents don't have the answer, this one searches the web to find what you need.
- **Planning**: It can think through problems in two ways - ReACT (quick and efficient) or Chain of Thought (thorough and detailed).

### Memory That Actually Works

Unlike most chatbots, this one has both short-term and long-term memory:

- **Short-term Memory**: Remembers the last 10 things you talked about. So if you're working through a problem step by step, it knows what step you're on.
- **Long-term Memory**: Important facts get saved permanently. If you tell it you're studying calculus, it'll remember that next time.
- **Smart Memory Management**: It automatically figures out what's important enough to remember long-term. You don't have to do anything.

### Document Management

Upload your PDFs and the system takes care of the rest:

- **Automatic Indexing**: Drop in your PDFs and they're automatically searchable
- **Smart Search**: Uses semantic search, so you can ask questions in natural language and it finds the right parts
- **Summaries**: It automatically creates summaries of your documents so you can quickly see what's in them
- **Multi-document**: Search across all your documents at once, or focus on a specific one

### Research Graph Builder

Build interactive knowledge graphs from research papers:

- **PDF Processing**: Uses Grobid for clean text and formula extraction from PDFs
- **Keyword Extraction**: Uses KeyBERT with Sentence-BERT embeddings to extract meaningful keywords
- **Semantic Understanding**: Uses Sentence-BERT (all-mpnet-base-v2) for capturing real meaning, not just words
- **Concept Comparison**: Uses Ollama/Mistral to check if sentences describe the same scientific concept
- **Graph Visualization**: Interactive network graphs showing relationships between papers and topics
- **Multiple Graphs**: Save multiple graph builds with descriptive filenames (paper names separated by underscore)
- **Graph Management**: Delete graphs you no longer need
- **Filtering Options**:
  - **All Nodes**: Shows all papers and topics
  - **Papers Only**: Shows only paper nodes
  - **Topics Only**: Shows all topic nodes
  - **Linked Topics**: Shows only topics that are connected to other topics (topic-to-topic relationships)
  - **Difference**: Shows only topics that are linked to topics from different papers (excludes connections where both topics are from the same paper)

### MCP Integration (Optional)

If you're into Model Context Protocol, you can connect custom MCP servers to enhance how the system processes information. It's optional though - everything works fine without it.

### Session Management

- Handle multiple conversations at once (great if you're tutoring multiple students)
- Full conversation history for each session
- Everything saves automatically so you don't lose your progress

## How It Works (The Big Picture)

Here's what happens when you ask a question:

1. You send a question
2. The Central Agent receives it and adds it to memory
3. The Central Agent creates a plan - which helpers should it call?
4. The helpers do their thing:
   - Local Data Agent searches your documents
   - Search Engine Agent searches the web (if needed)
5. All the results get combined
6. The system generates a comprehensive answer
7. Important stuff gets saved to memory for next time

It's like having a team working together instead of just one person trying to remember everything.

## Getting Started

### What You'll Need

- Python 3.8 or newer
- Ollama installed and running on your computer
- The Mistral 7B model (or something compatible)
- Grobid server (optional, for advanced PDF processing with formulas)

### Step-by-Step Setup

First, make sure you have Ollama installed. If you don't, grab it from ollama.ai. Then pull the Mistral model:

```bash
ollama pull mistral:7b-instruct
```

**Optional: Set up Grobid** (for better PDF extraction with formulas)

If you want to use Grobid for PDF processing (recommended for research papers with formulas):

1. Install Grobid server (see [Grobid documentation](https://grobid.readthedocs.io/))
2. Start Grobid server (default: `http://localhost:8070`)
3. Set `GROBID_URL` environment variable if using a different address

Now let's set up the project:

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Agent
   ```

2. **Create a virtual environment** (trust me, you'll want this)
   ```bash
   python -m venv venv
   ```

3. **Activate it**
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Install everything**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up the database** (this indexes your documents)
   ```bash
   python rag_setup.py
   ```

6. **Start the server**
   ```bash
   uvicorn app:app --reload
   ```

Now you should be able to access the API at `http://localhost:8000`. The application will display "Application start" when it launches.

## Using It

### Asking Questions

The main endpoint is `/ask`. Send it a question like this:

```json
{
  "mode": "math",
  "question": "What is the derivative of x^2?",
  "session_id": "user123",
  "planning_mode": "react"
}
```

The `mode` can be "math" or "physics". The `planning_mode` can be "react" (faster) or "cot" (more thorough). The `session_id` lets you have separate conversations - use the same ID to continue a conversation.

### Uploading Documents

Want to ask questions about your PDFs? Upload them to `/upload_pdfs`. The system will automatically:
- Extract the text (using Grobid if available, otherwise PyPDF2)
- Extract formulas (if using Grobid)
- Index it for searching
- Create summaries
- Make it searchable

Just upload and go - no manual setup needed.

### Querying Specific Documents

If you have multiple PDFs and want to ask about just one:

```json
{
  "pdf_name": "math.pdf",
  "question": "Explain integration by parts"
}
```

### Hybrid Search

Sometimes you want both local documents and web results. Use `/hybrid_search`:

```json
{
  "query": "latest developments in quantum computing"
}
```

This searches both your documents and the web, giving you the best of both worlds.

### Building Research Graphs

Create knowledge graphs from your research papers:

1. **Upload PDFs** via `/upload_pdfs` or use the frontend
2. **Build a graph** via `/research-graph/build`:
   ```json
   {
     "pdf_files": ["paper1.pdf", "paper2.pdf"]
   }
   ```
   Or leave `pdf_files` empty to use all PDFs in the data folder.

3. **Graphs are saved** with filenames based on paper names (e.g., `paper1_paper2.json`)

4. **List all graphs** via `/research-graph/list`

5. **Load a specific graph** via `/research-graph?graph_filename=paper1_paper2.json`

6. **Delete a graph** via `DELETE /research-graph/{graph_filename}`

The graph building process:
- Extracts clean text and formulas using Grobid (or PyPDF2 fallback)
- Extracts keywords using KeyBERT with Sentence-BERT embeddings
- Creates sentence embeddings using Sentence-BERT (all-mpnet-base-v2)
- Compares sentences using Ollama/Mistral to find same concepts
- Builds relationships between papers based on shared topics, embeddings, and concept similarity
- Stores everything in JSON format with nodes (papers and topics) and edges (relationships)

## API Endpoints

### The Main Ones

- `POST /ask` - Ask a question. This is what you'll use most of the time.
- `POST /upload_pdfs` - Upload PDFs to make them searchable
- `POST /ask_doc` - Ask about a specific document
- `POST /hybrid_search` - Search both local docs and web

### Research Graph Endpoints

- `POST /research-graph/build` - Build a knowledge graph from selected PDFs
- `GET /research-graph` - Get a research graph (optionally specify `graph_filename` query parameter)
- `GET /research-graph/list` - List all available graphs
- `DELETE /research-graph/{graph_filename}` - Delete a specific graph
- `GET /research-graph/pdfs` - List all available PDF files

### Managing Sessions

- `GET /sessions` - See all your active sessions
- `GET /session/{session_id}` - Get the conversation history for a session
- `DELETE /session/{session_id}` - Delete a session
- `POST /reset_session` - Clear everything for a session (both conversation and memory)

### Agent Stuff

- `GET /agent/{session_id}/memory` - See what the agent remembers
- `GET /agent/{session_id}/status` - Check on the agent's status
- `POST /agent/{session_id}/clear_memory` - Clear the agent's memory
- `GET /agents` - List all active agents

### Document Summaries

- `POST /summarize_all` - Generate summaries for all PDFs
- `GET /summaries` - List available summaries
- `GET /summary/{name}` - Get a specific summary

### MCP Configuration

- `POST /configure_mcp` - Set up an MCP server connection
- `GET /mcp_status` - Check if MCP is connected

## Project Structure

Here's what's in the project:

```
Agent/
├── app.py                 # The main FastAPI app - all the endpoints live here
├── agentic_core.py        # The Central Agent with memory and planning
├── specialized_agents.py  # The Local Data and Search Engine agents
├── generation.py          # Takes all the results and makes a final answer
├── research_graph.py      # Research graph builder with Grobid, KeyBERT, Sentence-BERT
├── rag_setup.py           # Sets up the document search database
├── mcp_client.py          # Handles MCP server connections
├── session_manager.py     # Manages conversation sessions
├── summarizer.py          # Creates summaries of PDFs
├── config.py              # Configuration settings
├── requirements.txt       # Python packages you need
├── data/                  # Where everything gets stored
│   ├── *.pdf             # Your uploaded PDFs
│   ├── chroma_db/        # The search database
│   ├── summaries/        # Generated summaries
│   ├── memories/         # Long-term memory storage
│   ├── research_graphs/  # Saved research graphs (JSON files)
│   └── sessions.json     # Conversation history
└── README.md             # This file
```

## Technologies Used

**Core Framework**
- FastAPI - Modern Python web framework
- Uvicorn - The server that runs everything

**AI/ML**
- Ollama - Runs the language model locally (Mistral 7B)
- Sentence Transformers (Sentence-BERT) - Converts text to vectors for searching
- ChromaDB - Stores and searches your documents
- KeyBERT - Extracts keywords using embeddings

**Document Processing**
- Grobid - Extracts clean text and formulas from PDFs (optional)
- PyPDF2 - Fallback PDF text extraction
- LangChain - Helps split and process text

**Graph Processing**
- NetworkX - Builds and manages knowledge graphs
- Sentence-BERT (all-mpnet-base-v2) - Creates semantic embeddings
- KeyBERT - Extracts keywords/phrases using embeddings
- Ollama/Mistral - Compares sentences to find same concepts

**Other Useful Stuff**
- SymPy - For doing actual math calculations
- Pint - Handles physics units
- Requests - Makes HTTP calls
- MCP SDK - For Model Context Protocol integration
- NumPy - Numerical computations
- scikit-learn - Machine learning utilities

## Configuration

Want to change something? Edit `config.py`:

```python
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "mistral:7b-instruct"
HTTP_TIMEOUT = 60.0
MAX_HISTORY_TURNS = 4
```

Change the model name if you're using something different, adjust the timeout if you need more time, etc.

**Environment Variables:**
- `GROBID_URL` - Set to your Grobid server URL (default: `http://localhost:8070`)
- `OLLAMA_KEEP_ALIVE` - Keep model loaded in memory (e.g., `5m` for 5 minutes)

## How It Actually Works

Let me walk you through what happens when you ask a question:

1. **You ask a question** - Send it to `/ask` with your question and session ID

2. **Central Agent wakes up** - It adds your question to short-term memory, then thinks about what needs to happen. Does it need to search documents? The web? Both?

3. **Planning happens** - Depending on the planning mode:
   - ReACT: Quick analysis, activates only what's needed
   - CoT: Breaks down the question, creates a detailed plan

4. **Agents do their thing**:
   - Local Data Agent searches your PDFs using semantic search
   - Search Engine Agent hits the web if needed
   - Both can process results through MCP if you have it set up

5. **Everything gets combined** - The Generation module takes all the results, adds memory context, and creates a comprehensive answer

6. **Memory gets updated** - Important stuff gets saved. The answer goes into short-term memory, and if it's really important, it might get promoted to long-term memory

7. **You get your answer** - Along with metadata about what sources were used

## Research Graph Building Process

When building a research graph:

1. **PDF Extraction**:
   - Tries Grobid first (if available) for clean text + formulas
   - Falls back to PyPDF2 if Grobid unavailable
   - Extracts formulas from PDFs (if using Grobid)

2. **Keyword Extraction**:
   - Uses KeyBERT with Sentence-BERT embeddings
   - Extracts meaningful keywords/phrases (1-2 word phrases)
   - Validates keywords appear in source text

3. **Embedding Creation**:
   - Uses Sentence-BERT (all-mpnet-base-v2) for semantic embeddings
   - Creates embeddings for topics/keywords
   - Creates embeddings for paper contexts

4. **Relationship Detection**:
   - Finds similar topics using embeddings
   - Compares papers using:
     - Shared keywords (40% weight)
     - Sentence-BERT embedding similarity (30% weight)
     - Ollama sentence comparison for same concepts (30% weight)
   - Uses Ollama/Mistral to check if sentences describe the same scientific concept

5. **Graph Storage**:
   - Saves as JSON with nodes (papers and topics) and edges (relationships)
   - Filename based on paper names (e.g., `paper1_paper2.json`)
   - Stores formulas, keywords, embeddings, and relationship metadata

## Planning Modes Explained

**ReACT (Reasoning and Acting)**
This is the faster mode. It looks at your question, figures out what it needs, and activates only the necessary agents. Good for straightforward questions.

**CoT (Chain of Thought)**
This one thinks things through more carefully. It breaks down your question into parts, creates a detailed plan, and usually activates multiple agents to be thorough. Use this when you want comprehensive answers.

## The Memory System

The memory system is what makes this different from regular chatbots:

- **Short-term Memory**: Holds the last 10 interactions. Perfect for following a conversation or working through a problem step by step.

- **Long-term Memory**: Important facts get saved here permanently. If you tell it you're studying calculus, that goes in long-term memory.

- **Automatic Promotion**: When short-term memory fills up, really important stuff (importance > 0.7) automatically moves to long-term. You don't have to do anything.

- **Context Retrieval**: Every time you ask a question, the system pulls relevant context from both short-term and long-term memory to give you better answers.

## MCP Integration

MCP (Model Context Protocol) is optional but powerful. If you set it up:

1. Configure your MCP server via `/configure_mcp`
2. The system can enhance context between retrieving information and generating answers
3. Both the Local Data Agent and Search Engine Agent can use MCP processing
4. If you don't set it up, everything still works fine - it just won't use MCP

## Example Questions

Here are some things you might ask:

**Math Questions**
- "What is the integral of sin(x)?"
- "Explain the chain rule in calculus"
- "How do I solve quadratic equations?"
- "What's the difference between a derivative and an integral?"

**Physics Questions**
- "What is Newton's second law?"
- "Explain the photoelectric effect"
- "How does quantum entanglement work?"
- "What's the relationship between force, mass, and acceleration?"

The system is designed for math and physics, but feel free to experiment!

## Frontend

The project includes a React frontend for:
- Chat interface with the tutoring system
- Research graph visualization (interactive network graphs)
- Graph building and management (upload PDFs, build graphs, delete graphs)
- Paper comparison and visualization
- **Graph Filtering**: Multiple filter options to focus on specific aspects:
  - View all nodes, papers only, or topics only
  - **Linked Topics**: Focus on topics that have relationships with other topics
  - **Difference**: Identify topics that connect across different papers (cross-paper relationships)

To run the frontend:
```bash
cd agent_frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173` (or the port shown in the terminal).

## Troubleshooting

**Grobid not working?**
- Make sure Grobid server is running on port 8070 (or set `GROBID_URL` environment variable)
- The system will automatically fall back to PyPDF2 if Grobid is unavailable
- Check Grobid logs for any errors

**Ollama connection issues?**
- Make sure Ollama is running: `ollama serve`
- Verify the model is installed: `ollama list`
- Check `config.py` for correct `OLLAMA_URL` and `MODEL_NAME`

**Graph building is slow?**
- Sentence comparison with Ollama can be slow - it compares up to 5 sentence pairs per paper pair
- Consider reducing the number of papers in a single graph build
- The system processes papers sequentially for better reliability

**Memory issues?**
- Large PDFs can consume memory during processing
- Consider processing fewer papers at once
- The system uses efficient embeddings, but very large documents may need chunking

## Contributing

Found a bug? Have an idea? Want to add a feature? Contributions are welcome! Just submit a pull request and we'll take a look.

## License

[Add your license here]

## Acknowledgments

Built with FastAPI and Ollama for the AI, ChromaDB for document storage, Grobid for PDF processing, Sentence-BERT for embeddings, KeyBERT for keyword extraction, NetworkX for graph building, and integrates with Model Context Protocol for advanced features.

---

**Important Notes**: 
- Make sure Ollama is running before you start the server. The system needs it to generate answers. If Ollama isn't running, you'll get errors when trying to ask questions.
- Grobid is optional but recommended for research papers with formulas. The system works fine without it, using PyPDF2 as a fallback.
- All console output is plain text (no emojis) for better compatibility with various terminals.
