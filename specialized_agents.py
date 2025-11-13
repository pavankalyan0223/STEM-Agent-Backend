"""
Specialized Agents Module
Implements Agent 1 (Local Data) and Agent 2 (Search Engine) with MCP server integration.
"""
from typing import Dict, List, Optional, Any
import requests
import urllib.parse
import asyncio
from rag_setup import query_rag
from mcp_client import process_with_mcp, get_mcp_client


class LocalDataAgent:
    """Agent 1: Handles local data retrieval (documents, PDFs, RAG)."""
    
    def __init__(self, agent_id: str = "local_data_agent"):
        self.agent_id = agent_id
        self.mcp_client = get_mcp_client()
    
    async def retrieve_local_context(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Retrieve context from local documents using RAG.
        
        Args:
            query: Search query
            top_k: Number of results to retrieve
            
        Returns:
            Dictionary with retrieved context and metadata
        """
        try:
            # Step 1: Query RAG system
            local_context = query_rag(query, top_k=top_k)
            
            # Step 2: Process through MCP server if available
            mcp_result = await process_with_mcp(query, local_context)
            processed_context = mcp_result.get("processed_context", local_context)
            mcp_metadata = mcp_result.get("mcp_metadata", {})
            
            return {
                "agent": self.agent_id,
                "status": "success",
                "raw_context": local_context,
                "processed_context": processed_context,
                "mcp_enhanced": mcp_result.get("enhanced", False),
                "mcp_metadata": mcp_metadata,
                "source": "local_documents",
                "results_count": top_k
            }
        except Exception as e:
            return {
                "agent": self.agent_id,
                "status": "error",
                "error": str(e),
                "source": "local_documents"
            }
    
    async def query_document(self, pdf_name: str, question: str) -> Dict[str, Any]:
        """Query a specific document."""
        try:
            from rag_setup import collection, embedder
            q_emb = embedder.encode(question).tolist()
            
            results = collection.query(
                query_embeddings=[q_emb],
                n_results=3,
                where={"source": pdf_name}
            )
            
            context = "\n\n".join(results["documents"][0])
            
            return {
                "agent": self.agent_id,
                "status": "success",
                "document": pdf_name,
                "context": context,
                "source": "specific_document"
            }
        except Exception as e:
            return {
                "agent": self.agent_id,
                "status": "error",
                "error": str(e),
                "document": pdf_name
            }


class SearchEngineAgent:
    """Agent 2: Handles web search functionality via MCP."""
    
    def __init__(self, agent_id: str = "search_engine_agent"):
        self.agent_id = agent_id
        self.mcp_client = get_mcp_client()
    
    async def search_web(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search the web for information using MCP server.
        MCP handles web search, so we delegate to MCP tools.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Dictionary with search results
        """
        # Use MCP for web search if available
        if self.mcp_client and self.mcp_client.available and self.mcp_client.session:
            try:
                # Get available MCP tools
                tools = await self.mcp_client.session.list_tools()
                
                # Look for web search tool
                web_search_tool = None
                if tools and tools.tools:
                    for tool in tools.tools:
                        tool_name_lower = tool.name.lower()
                        if "search" in tool_name_lower or "web" in tool_name_lower or "duckduckgo" in tool_name_lower:
                            web_search_tool = tool
                            break
                
                if web_search_tool:
                    # Use MCP tool for web search
                    try:
                        result = await self.mcp_client.session.call_tool(
                            web_search_tool.name,
                            arguments={"query": query, "max_results": max_results}
                        )
                        
                        # Process MCP result
                        if result.content:
                            # Extract results from MCP response
                            # MCP might return different formats, so we handle various cases
                            content_text = result.content[0].text if result.content else ""
                            
                            return {
                                "agent": self.agent_id,
                                "status": "success",
                                "query": query,
                                "results": [],  # MCP handles formatting
                                "abstract": content_text,
                                "abstract_url": None,
                                "processed_summary": content_text,
                                "results_count": 1,
                                "source": "web_search",
                                "mcp_tool": web_search_tool.name
                            }
                    except Exception as e:
                        print(f"MCP web search tool error: {e}")
                        # Fall through to fallback
                
            except Exception as e:
                print(f"MCP web search error: {e}")
                # Fall through to fallback
        
        # Fallback: Use direct DuckDuckGo search if MCP not available or fails
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
            
            def do_search():
                try:
                    ddgs = DDGS()
                    search_results = list(ddgs.text(query, max_results=max_results))
                    
                    web_results = []
                    for result in search_results:
                        web_results.append({
                            "title": result.get("title", ""),
                            "url": result.get("href", ""),
                            "snippet": result.get("body", "")[:300]
                        })
                    return web_results
                except Exception as e:
                    print(f"DuckDuckGo search error: {e}")
                    return []
            
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            web_results = await loop.run_in_executor(None, do_search)
            
            if not web_results:
                return {
                    "agent": self.agent_id,
                    "status": "no_results",
                    "query": query,
                    "results": [],
                    "source": "web_search",
                    "error": "No search results found"
                }
            
            # Process through MCP if available (even if MCP doesn't have web search tool)
            processed_summary = None
            if web_results and self.mcp_client:
                try:
                    results_summary = "\n".join([f"{r['title']}: {r['snippet']}" for r in web_results])
                    mcp_result = await process_with_mcp(query, results_summary)
                    processed_summary = mcp_result.get("processed_context", results_summary)
                except Exception as e:
                    print(f"MCP processing error: {e}")
            
            return {
                "agent": self.agent_id,
                "status": "success",
                "query": query,
                "results": web_results,
                "abstract": None,
                "abstract_url": None,
                "processed_summary": processed_summary,
                "results_count": len(web_results),
                "source": "web_search"
            }
            
        except ImportError:
            return {
                "agent": self.agent_id,
                "status": "error",
                "error": "Web search not available. Configure MCP server or install ddgs package.",
                "query": query,
                "source": "web_search"
            }
        except Exception as e:
            return {
                "agent": self.agent_id,
                "status": "error",
                "error": str(e),
                "query": query,
                "source": "web_search"
            }
    
    async def search_with_context(self, query: str, context: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search web with additional context (e.g., from local documents).
        
        Args:
            query: Search query
            context: Additional context to refine search
            max_results: Maximum number of results
        """
        # Enhance query with context
        enhanced_query = f"{query} {context[:100]}"  # Add context snippet
        
        return await self.search_web(enhanced_query, max_results)


class AgentOrchestrator:
    """Orchestrates multiple specialized agents."""
    
    def __init__(self):
        self.local_data_agent = LocalDataAgent()
        self.search_engine_agent = SearchEngineAgent()
        self.agents = {
            "local_data_agent": self.local_data_agent,
            "search_engine_agent": self.search_engine_agent
        }
    
    async def execute_plan(self, plan: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Execute a plan using specialized agents.
        
        Args:
            plan: Plan dictionary from Planning module
            query: Original query
            
        Returns:
            Dictionary with execution results from all agents
        """
        steps = plan.get("steps", [])
        results = {
            "query": query,
            "plan_mode": plan.get("mode", "unknown"),
            "agent_results": {},
            "execution_summary": []
        }
        
        for step in steps:
            agent_name = step.get("agent")
            action = step.get("action")
            
            if agent_name == "central_agent":
                # Skip synthesis step for now (handled separately)
                continue
            
            agent = self.agents.get(agent_name)
            if not agent:
                results["agent_results"][agent_name] = {
                    "status": "error",
                    "error": f"Agent {agent_name} not found"
                }
                continue
            
            # Execute agent action
            try:
                if agent_name == "central_agent" and action == "direct_response":
                    # Skip agent processing for non-technical questions
                    results["agent_results"][agent_name] = {
                        "status": "skipped",
                        "reason": "Non-technical question - direct response mode"
                    }
                    results["execution_summary"].append({
                        "agent": agent_name,
                        "action": action,
                        "status": "skipped"
                    })
                
                elif agent_name == "local_data_agent" and action == "retrieve_local_context":
                    agent_result = await agent.retrieve_local_context(query)
                    results["agent_results"][agent_name] = agent_result
                    results["execution_summary"].append({
                        "agent": agent_name,
                        "action": action,
                        "status": agent_result.get("status", "unknown")
                    })
                
                elif agent_name == "search_engine_agent" and action == "search_web":
                    agent_result = await agent.search_web(query)
                    results["agent_results"][agent_name] = agent_result
                    results["execution_summary"].append({
                        "agent": agent_name,
                        "action": action,
                        "status": agent_result.get("status", "unknown")
                    })
                else:
                    results["agent_results"][agent_name] = {
                        "status": "error",
                        "error": f"Unknown action: {action}"
                    }
            except Exception as e:
                results["agent_results"][agent_name] = {
                    "status": "error",
                    "error": str(e)
                }
                results["execution_summary"].append({
                    "agent": agent_name,
                    "action": action,
                    "status": "error",
                    "error": str(e)
                })
        
        return results

