"""
Agentic Core Module
Implements the central Agent with Memory and Planning capabilities.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
import os
from dataclasses import dataclass, asdict
from enum import Enum


class PlanningMode(Enum):
    """Planning strategies."""
    REACT = "react"  # Reasoning and Acting
    COT = "cot"  # Chain of Thought


@dataclass
class MemoryEntry:
    """Single memory entry."""
    content: str
    timestamp: str
    importance: float  # 0.0 to 1.0
    context: Dict[str, Any]


class Memory:
    """Memory system with Short-term and Long-term memory."""
    
    def __init__(self, session_id: str, memory_dir: str = "data/memories"):
        self.session_id = session_id
        self.memory_dir = memory_dir
        os.makedirs(memory_dir, exist_ok=True)
        
        # Short-term memory (recent conversation)
        self.short_term: List[MemoryEntry] = []
        self.short_term_max_size = 10
        
        # Long-term memory (persistent, important facts)
        self.long_term_file = os.path.join(memory_dir, f"{session_id}_long_term.json")
        self.long_term: List[MemoryEntry] = self._load_long_term()
    
    def _load_long_term(self) -> List[MemoryEntry]:
        """Load long-term memory from disk."""
        if os.path.exists(self.long_term_file):
            try:
                with open(self.long_term_file, 'r') as f:
                    data = json.load(f)
                    return [MemoryEntry(**entry) for entry in data]
            except Exception as e:
                print(f"Error loading long-term memory: {e}")
        return []
    
    def _save_long_term(self):
        """Save long-term memory to disk."""
        try:
            with open(self.long_term_file, 'w') as f:
                json.dump([asdict(entry) for entry in self.long_term], f, indent=2)
        except Exception as e:
            print(f"Error saving long-term memory: {e}")
    
    def add_short_term(self, content: str, context: Optional[Dict[str, Any]] = None):
        """Add entry to short-term memory."""
        entry = MemoryEntry(
            content=content,
            timestamp=datetime.now().isoformat(),
            importance=0.5,
            context=context or {}
        )
        self.short_term.append(entry)
        
        # Maintain max size
        if len(self.short_term) > self.short_term_max_size:
            # Move oldest to long-term if important enough
            oldest = self.short_term.pop(0)
            if oldest.importance > 0.7:
                self.add_long_term(oldest.content, oldest.context, oldest.importance)
    
    def add_long_term(self, content: str, context: Optional[Dict[str, Any]] = None, importance: float = 0.8):
        """Add entry to long-term memory."""
        entry = MemoryEntry(
            content=content,
            timestamp=datetime.now().isoformat(),
            importance=importance,
            context=context or {}
        )
        self.long_term.append(entry)
        self._save_long_term()
    
    def get_context(self, max_short_term: int = 3, max_long_term: int = 2) -> str:
        """Get formatted context from memory."""
        context_parts = []
        
        # Short-term memory (most recent)
        recent = self.short_term[-max_short_term:] if len(self.short_term) > max_short_term else self.short_term
        if recent:
            context_parts.append("=== Recent Context (Short-term Memory) ===")
            for entry in recent:
                context_parts.append(f"[{entry.timestamp}] {entry.content}")
        
        # Long-term memory (most important)
        important = sorted(self.long_term, key=lambda x: x.importance, reverse=True)[:max_long_term]
        if important:
            context_parts.append("\n=== Important Facts (Long-term Memory) ===")
            for entry in important:
                context_parts.append(f"[{entry.timestamp}] {entry.content}")
        
        return "\n".join(context_parts) if context_parts else "No memory context available."
    
    def clear_short_term(self):
        """Clear short-term memory."""
        self.short_term = []


class Planning:
    """Planning module with ReACT and CoT strategies."""
    
    def __init__(self, mode: PlanningMode = PlanningMode.REACT):
        self.mode = mode
    
    def plan(self, query: str, available_agents: List[str], memory_context: str) -> Dict[str, Any]:
        """
        Generate a plan for executing the query.
        Always uses both local data agent and MCP (which handles web search).
        
        Args:
            query: User query
            available_agents: List of available agents
            memory_context: Context from memory
        
        Returns:
            Dictionary with plan steps and reasoning
        """
        if self.mode == PlanningMode.REACT:
            return self._react_plan(query, available_agents, memory_context)
        elif self.mode == PlanningMode.COT:
            return self._cot_plan(query, available_agents, memory_context)
        else:
            return {"steps": [], "reasoning": "Unknown planning mode"}
    
    def _react_plan(self, query: str, available_agents: List[str], memory_context: str) -> Dict[str, Any]:
        """ReACT planning: Reasoning and Acting. Always uses both local data and MCP (web search)."""
        steps = []
        reasoning = []
        
        # Step 1: Think about what information is needed
        reasoning.append(f"Query: {query}")
        reasoning.append(f"Available agents: {', '.join(available_agents)}")
        reasoning.append("→ MCP handles web search, always using both local data and MCP")
        
        # Step 2: Always activate both agents
        # Local data agent for document retrieval
        steps.append({
            "agent": "local_data_agent",
            "action": "retrieve_local_context",
            "reason": "Retrieve relevant information from local documents"
        })
        reasoning.append("→ Activating Local Data Agent for document retrieval")
        
        # Search engine agent (MCP handles web search)
        steps.append({
            "agent": "search_engine_agent",
            "action": "search_web",
            "reason": "MCP handles web search for comprehensive results"
        })
        reasoning.append("→ Activating Search Engine Agent (via MCP) for web search")
        
        # Step 3: Plan synthesis
        steps.append({
            "agent": "central_agent",
            "action": "synthesize_results",
            "reason": "Combine results from all agents"
        })
        
        return {
            "mode": "react",
            "steps": steps,
            "reasoning": "\n".join(reasoning),
            "estimated_steps": len(steps)
        }
    
    def _cot_plan(self, query: str, available_agents: List[str], memory_context: str) -> Dict[str, Any]:
        """Chain of Thought planning. Always uses both local data and MCP (web search)."""
        reasoning = []
        steps = []
        
        reasoning.append("=== Chain of Thought Planning ===")
        reasoning.append(f"Step 1: Understanding the query: {query}")
        reasoning.append("→ MCP handles web search, always using both local data and MCP")
        
        # Break down the query
        reasoning.append("Step 2: Breaking down the query into components:")
        query_parts = query.split("?")[0].split(".")
        for i, part in enumerate(query_parts, 1):
            if part.strip():
                reasoning.append(f"  {i}. {part.strip()}")
        
        reasoning.append("Step 3: Determining required information sources:")
        reasoning.append("  Required sources: local_documents, web_search (via MCP)")
        
        # Always use both agents
        steps.append({
            "agent": "local_data_agent",
            "action": "retrieve_local_context",
            "reason": "Retrieve relevant information from local documents"
        })
        
        steps.append({
            "agent": "search_engine_agent",
            "action": "search_web",
            "reason": "MCP handles web search for comprehensive results"
        })
        
        reasoning.append("Step 4: Planning execution sequence")
        
        steps.append({
            "agent": "central_agent",
            "action": "synthesize_results",
            "reason": "Final synthesis"
        })
        
        return {
            "mode": "cot",
            "steps": steps,
            "reasoning": "\n".join(reasoning),
            "estimated_steps": len(steps)
        }


class CentralAgent:
    """Central Agent orchestrating specialized agents."""
    
    def __init__(self, session_id: str, planning_mode: PlanningMode = PlanningMode.REACT):
        self.session_id = session_id
        self.memory = Memory(session_id)
        self.planning = Planning(planning_mode)
        self.available_agents = ["local_data_agent", "search_engine_agent"]
    
    def process_query(self, query: str, mode: str = "math") -> Dict[str, Any]:
        """
        Process a query through the agentic system.
        MCP handles web search, so we always use both local data and MCP.
        
        Args:
            query: User query
            mode: Subject mode (math/physics)
        
        Returns:
            Dictionary with plan, execution results, and final answer
        """
        # Add query to short-term memory
        self.memory.add_short_term(f"User query: {query}", {"mode": mode})
        
        # Get memory context
        memory_context = self.memory.get_context()
        
        # Generate plan (always use both local data and MCP for web search)
        plan = self.planning.plan(query, self.available_agents, memory_context)
        
        return {
            "query": query,
            "session_id": self.session_id,
            "plan": plan,
            "memory_context": memory_context,
            "mode": mode
        }
    
    def add_to_memory(self, content: str, is_important: bool = False, context: Optional[Dict[str, Any]] = None):
        """Add information to memory."""
        if is_important:
            self.memory.add_long_term(content, context)
        else:
            self.memory.add_short_term(content, context)
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of memory state."""
        return {
            "short_term_count": len(self.memory.short_term),
            "long_term_count": len(self.memory.long_term),
            "context": self.memory.get_context()
        }

