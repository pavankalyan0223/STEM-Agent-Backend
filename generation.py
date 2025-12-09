"""
Generation Module
Handles final answer generation by synthesizing results from all agents.
"""
from typing import Dict, List, Optional, Any, Iterator
import requests
import json
from config import OLLAMA_URL, MODEL_NAME, HTTP_TIMEOUT


class Generator:
    """Generation component that synthesizes agent results into final answer."""
    
    def __init__(self):
        self.model_url = OLLAMA_URL
        self.model_name = MODEL_NAME
    
    def generate_answer(
        self,
        query: str,
        agent_results: Dict[str, Any],
        memory_context: str,
        mode: str = "math",
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate final answer by synthesizing all agent results.
        Always uses both local documents and web search (via Helper Agents).
        
        Args:
            query: Original user query
            agent_results: Results from all specialized agents
            memory_context: Context from memory system
            mode: Subject mode (math/physics)
            conversation_history: Previous conversation messages
            
        Returns:
            Dictionary with generated answer and metadata
        """
        # Build context from agent results (always use both sources)
        context_parts = []
        
        # Get agent results
        local_data = agent_results.get("local_data_agent", {})
        search_results = agent_results.get("search_engine_agent", {})
        
        # Add memory context
        if memory_context and memory_context != "No memory context available.":
            context_parts.append("=== Memory Context ===")
            context_parts.append(memory_context)
            context_parts.append("")
        
        # Add local data agent results
        if local_data.get("status") == "success":
            context_parts.append("=== Local Document Context ===")
            processed = local_data.get("processed_context") or local_data.get("raw_context", "")
            if processed:
                context_parts.append(processed)
                context_parts.append("")
        
        # Add search engine agent results (from Helper Agents)
        if search_results.get("status") == "success":
            context_parts.append("=== Web Search Results (via Helper Agents) ===")
            results = search_results.get("results", [])
            for i, result in enumerate(results[:5], 1):
                context_parts.append(f"{i}. {result.get('title', '')}")
                if result.get('snippet'):
                    context_parts.append(f"   {result.get('snippet', '')[:200]}...")
                if result.get('url'):
                    context_parts.append(f"   Source: {result.get('url', '')}")
            context_parts.append("")
            
            # Add abstract or processed summary if available (from Helper Agents)
            if search_results.get("abstract"):
                context_parts.append("=== Direct Answer from Web ===")
                context_parts.append(search_results.get("abstract"))
                context_parts.append("")
            elif search_results.get("processed_summary"):
                context_parts.append("=== Web Search Summary (via Helper Agents) ===")
                context_parts.append(search_results.get("processed_summary"))
                context_parts.append("")
        
        # Combine all context
        full_context = "\n".join(context_parts)
        
        # Build system prompt
        system_prompt = (
            f"You are a friendly and highly knowledgeable {mode} expert specializing in mathematics and physics. "
            "You ONLY provide answers related to math and physics topics. "
            "If the user asks about unrelated topics (like geography, languages, general knowledge), "
            "politely redirect them to math/physics questions. "
            "You have access to local documents and web search results (via Helper Agents). "
            "Synthesize the information from all sources to provide a comprehensive answer. "
            "If information conflicts, prioritize local documents for domain-specific knowledge "
            "and web search for current or general information. "
            "Be accurate, clear, and conversational. Focus strictly on {mode} topics."
        )
        
        # Build user prompt
        if full_context.strip():
            user_prompt = f"""Based on the following context, answer the user's question about {mode}.

Context:
{full_context}

User Question: {query}

Provide a comprehensive answer that synthesizes information from all available sources (local documents and web search via Helper Agents). 
IMPORTANT: Only answer if this is a {mode} question. If it's not related to {mode}, politely say you specialize in {mode} and ask how you can help with {mode} topics."""
        else:
            user_prompt = f"""Answer the user's question about {mode}.

User Question: {query}

IMPORTANT: Only answer if this is a {mode} question. If it's not related to {mode}, politely say you specialize in {mode} and ask how you can help with {mode} topics."""

        # Prepare messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history if available
        if conversation_history:
            # Filter out system messages and add user/assistant messages
            for msg in conversation_history[-3:]:  # Last 3 messages for context (reduced for faster responses)
                if msg.get("role") in ["user", "assistant"]:
                    messages.append(msg)
        
        messages.append({"role": "user", "content": user_prompt})
        
        # Generate answer
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False
        }
        
        try:
            res = requests.post(self.model_url, json=payload, timeout=HTTP_TIMEOUT)
            res.raise_for_status()
            answer = res.json()["message"]["content"]
            
            return {
                "status": "success",
                "answer": answer,
                "context_used": {
                    "memory": memory_context != "No memory context available.",
                    "local_documents": local_data.get("status") == "success",
                    "web_search": search_results.get("status") == "success"
                },
                "sources": {
                    "local_results_count": local_data.get("results_count", 0),
                    "web_results_count": search_results.get("results_count", 0)
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "answer": f"Error generating answer: {e}"
            }
    
    def generate_answer_stream(
        self,
        query: str,
        agent_results: Dict[str, Any],
        memory_context: str,
        mode: str = "math",
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Iterator[str]:
        """
        Generate final answer by streaming tokens from Ollama.
        Yields chunks of text as they are generated.
        
        Args:
            query: Original user query
            agent_results: Results from all specialized agents
            memory_context: Context from memory system
            mode: Subject mode (math/physics)
            conversation_history: Previous conversation messages
            
        Yields:
            Text chunks as they are generated
        """
        # Build context from agent results (same as non-streaming version)
        context_parts = []
        
        # Get agent results
        local_data = agent_results.get("local_data_agent", {})
        search_results = agent_results.get("search_engine_agent", {})
        
        # Add memory context
        if memory_context and memory_context != "No memory context available.":
            context_parts.append("=== Memory Context ===")
            context_parts.append(memory_context)
            context_parts.append("")
        
        # Add local data agent results
        if local_data.get("status") == "success":
            context_parts.append("=== Local Document Context ===")
            processed = local_data.get("processed_context") or local_data.get("raw_context", "")
            if processed:
                context_parts.append(processed)
                context_parts.append("")
        
        # Add search engine agent results
        if search_results.get("status") == "success":
            context_parts.append("=== Web Search Results (via Helper Agents) ===")
            results = search_results.get("results", [])
            for i, result in enumerate(results[:5], 1):
                context_parts.append(f"{i}. {result.get('title', '')}")
                if result.get('snippet'):
                    context_parts.append(f"   {result.get('snippet', '')[:200]}...")
                if result.get('url'):
                    context_parts.append(f"   Source: {result.get('url', '')}")
            context_parts.append("")
            
            if search_results.get("abstract"):
                context_parts.append("=== Direct Answer from Web ===")
                context_parts.append(search_results.get("abstract"))
                context_parts.append("")
            elif search_results.get("processed_summary"):
                context_parts.append("=== Web Search Summary (via Helper Agents) ===")
                context_parts.append(search_results.get("processed_summary"))
                context_parts.append("")
        
        # Combine all context
        full_context = "\n".join(context_parts)
        
        # Build system prompt
        system_prompt = (
            f"You are a friendly and highly knowledgeable {mode} expert specializing in mathematics and physics. "
            "You ONLY provide answers related to math and physics topics. "
            "If the user asks about unrelated topics (like geography, languages, general knowledge), "
            "politely redirect them to math/physics questions. "
            "You have access to local documents and web search results (via Helper Agents). "
            "Synthesize the information from all sources to provide a comprehensive answer. "
            "If information conflicts, prioritize local documents for domain-specific knowledge "
            "and web search for current or general information. "
            "Be accurate, clear, and conversational. Focus strictly on {mode} topics."
        )
        
        # Build user prompt
        if full_context.strip():
            user_prompt = f"""Based on the following context, answer the user's question about {mode}.

Context:
{full_context}

User Question: {query}

Provide a comprehensive answer that synthesizes information from all available sources (local documents and web search via Helper Agents). 
IMPORTANT: Only answer if this is a {mode} question. If it's not related to {mode}, politely say you specialize in {mode} and ask how you can help with {mode} topics."""
        else:
            user_prompt = f"""Answer the user's question about {mode}.

User Question: {query}

IMPORTANT: Only answer if this is a {mode} question. If it's not related to {mode}, politely say you specialize in {mode} and ask how you can help with {mode} topics."""

        # Prepare messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add conversation history if available
        if conversation_history:
            for msg in conversation_history[-3:]:
                if msg.get("role") in ["user", "assistant"]:
                    messages.append(msg)
        
        messages.append({"role": "user", "content": user_prompt})
        
        # Generate answer with streaming
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": True
        }
        
        try:
            res = requests.post(self.model_url, json=payload, timeout=HTTP_TIMEOUT, stream=True)
            res.raise_for_status()
            
            # Stream response from Ollama
            for line in res.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if "message" in chunk and "content" in chunk["message"]:
                            content = chunk["message"]["content"]
                            if content:
                                yield content
                        # Check if done
                        if chunk.get("done", False):
                            break
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            yield f"Error generating answer: {e}"
    

