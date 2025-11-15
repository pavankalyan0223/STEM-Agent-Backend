# Central config
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "mistral:7b-instruct"

HTTP_TIMEOUT = 60.0

# Session settings
MAX_HISTORY_TURNS = 4  # keep conversation concise (reduced for faster responses)

# Performance optimization: Set OLLAMA_KEEP_ALIVE environment variable to keep model loaded
# Example: export OLLAMA_KEEP_ALIVE=5m (keeps model in memory for 5 minutes)
# This prevents reloading the model on each request, significantly improving response time
