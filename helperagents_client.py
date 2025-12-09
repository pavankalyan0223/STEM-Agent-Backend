"""
Helper Agents Client Module
Handles communication with helper agent servers to enhance context between text retrieval and web search.
"""
from typing import Optional, Dict, Any

try:
    # Try different possible import paths for MCP SDK (used for helper agents)
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        # Alternative import path
        from mcp_sdk import ClientSession, StdioServerParameters
        from mcp_sdk.client.stdio import stdio_client
    HELPERAGENTS_AVAILABLE = True
except ImportError:
    HELPERAGENTS_AVAILABLE = False
    print("Warning: Helper Agents SDK not installed. Install with: pip install mcp")
    print("Note: Helper Agents functionality will be disabled. The system will work without helper agents processing.")


class HelperAgentsClient:
    """Client for interacting with helper agent servers."""
    
    def __init__(self, server_command: Optional[str] = None, server_args: Optional[list] = None):
        """
        Initialize Helper Agents client.
        
        Args:
            server_command: Command to run the helper agent server (e.g., "python", "node")
            server_args: Arguments for the server command (e.g., ["-m", "helper_agent_server"])
        """
        self.server_command = server_command
        self.server_args = server_args or []
        self.session: Optional[ClientSession] = None
        self.available = HELPERAGENTS_AVAILABLE
    
    async def connect(self):
        """Connect to the helper agent server."""
        if not self.available:
            return False
        
        try:
            if self.server_command:
                server_params = StdioServerParameters(
                    command=self.server_command,
                    args=self.server_args
                )
                self.session = await stdio_client(server_params)
                await self.session.initialize()
                return True
            else:
                # If no server command specified, return False
                # User can configure their own helper agent server connection
                return False
        except Exception as e:
            print(f"Helper Agents connection error: {e}")
            return False
    
    async def process_context(self, query: str, retrieved_context: str) -> Dict[str, Any]:
        """
        Process retrieved context through helper agent server before web search.
        
        Args:
            query: The original search query
            retrieved_context: Context retrieved from RAG/text retrieval
            
        Returns:
            Dictionary with processed context and metadata
        """
        if not self.available or not self.session:
            # If helper agents are not available, return original context
            return {
                "processed_context": retrieved_context,
                "enhanced": False,
                "helperagents_metadata": {}
            }
        
        try:
            # Call helper agent tools/resources to enhance context
            # This is a placeholder - adjust based on your helper agent server's available tools
            tools = await self.session.list_tools()
            
            # Example: Use a context enhancement tool if available
            enhanced_context = retrieved_context
            helperagents_metadata = {
                "tools_available": len(tools.tools) if tools else 0,
                "query": query
            }
            
            # If there's a context processing tool, use it
            if tools and tools.tools:
                for tool in tools.tools:
                    if "context" in tool.name.lower() or "enhance" in tool.name.lower():
                        try:
                            result = await self.session.call_tool(
                                tool.name,
                                arguments={
                                    "query": query,
                                    "context": retrieved_context
                                }
                            )
                            if result.content:
                                enhanced_context = result.content[0].text if result.content else retrieved_context
                                helperagents_metadata["tool_used"] = tool.name
                                helperagents_metadata["enhanced"] = True
                                break
                        except Exception as e:
                            print(f"Error calling helper agent tool {tool.name}: {e}")
            
            return {
                "processed_context": enhanced_context,
                "enhanced": helperagents_metadata.get("enhanced", False),
                "helperagents_metadata": helperagents_metadata
            }
            
        except Exception as e:
            print(f"Helper Agents processing error: {e}")
            return {
                "processed_context": retrieved_context,
                "enhanced": False,
                "helperagents_metadata": {"error": str(e)}
            }
    
    async def close(self):
        """Close the helper agent session."""
        if self.session:
            try:
                await self.session.close()
            except Exception as e:
                print(f"Error closing helper agent session: {e}")


# Global Helper Agents client instance (can be configured)
_helperagents_client: Optional[HelperAgentsClient] = None


def get_helperagents_client() -> Optional[HelperAgentsClient]:
    """Get the global Helper Agents client instance."""
    return _helperagents_client


def initialize_helperagents_client(server_command: Optional[str] = None, server_args: Optional[list] = None):
    """
    Initialize the global Helper Agents client.
    
    Args:
        server_command: Command to run the helper agent server
        server_args: Arguments for the server command
    """
    global _helperagents_client
    _helperagents_client = HelperAgentsClient(server_command, server_args)


async def process_with_helperagents(query: str, retrieved_context: str) -> Dict[str, Any]:
    """
    Convenience function to process context through helper agents.
    
    Args:
        query: The search query
        retrieved_context: Context from text retrieval
        
    Returns:
        Processed context dictionary
    """
    client = get_helperagents_client()
    if not client:
        return {
            "processed_context": retrieved_context,
            "enhanced": False,
            "helperagents_metadata": {"status": "no_client"}
        }
    
    # Ensure client is connected
    if not client.session:
        await client.connect()
    
    return await client.process_context(query, retrieved_context)

