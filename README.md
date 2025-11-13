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

### Step-by-Step Setup

First, make sure you have Ollama installed. If you don't, grab it from ollama.ai. Then pull the Mistral model:

```bash
ollama pull mistral:7b-instruct
```

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

Now you should be able to access the API at `http://localhost:8000`. Nice!

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
- Extract the text
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

## API Endpoints

### The Main Ones

- `POST /ask` - Ask a question. This is what you'll use most of the time.
- `POST /upload_pdfs` - Upload PDFs to make them searchable
- `POST /ask_doc` - Ask about a specific document
- `POST /hybrid_search` - Search both local docs and web

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
│   └── sessions.json     # Conversation history
└── README.md             # This file
```

## Technologies Used

**Core Framework**
- FastAPI - Modern Python web framework
- Uvicorn - The server that runs everything

**AI/ML**
- Ollama - Runs the language model locally
- Sentence Transformers - Converts text to vectors for searching
- ChromaDB - Stores and searches your documents

**Document Processing**
- PyPDF2 - Extracts text from PDFs
- LangChain - Helps split and process text

**Other Useful Stuff**
- SymPy - For doing actual math calculations
- Pint - Handles physics units
- Requests - Makes HTTP calls
- MCP SDK - For Model Context Protocol integration

## Configuration

Want to change something? Edit `config.py`:

```python
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "mistral:7b-instruct"
HTTP_TIMEOUT = 60.0
MAX_HISTORY_TURNS = 8
```

Change the model name if you're using something different, adjust the timeout if you need more time, etc.

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

## Contributing

Found a bug? Have an idea? Want to add a feature? Contributions are welcome! Just submit a pull request and we'll take a look.

## License

[Add your license here]

## Acknowledgments

Built with FastAPI and Ollama for the AI, ChromaDB for document storage, and integrates with Model Context Protocol for advanced features.

---

**Important Note**: Make sure Ollama is running before you start the server. The system needs it to generate answers. If Ollama isn't running, you'll get errors when trying to ask questions.
