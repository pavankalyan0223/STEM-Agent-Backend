"""
MCP (Model Context Protocol) Client Module
Handles communication with MCP servers to enhance context between text retrieval and web search.
"""
from typing import Optional, Dict, Any

try:
    # Try different possible import paths for MCP SDK
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        # Alternative import path
        from mcp_sdk import ClientSession, StdioServerParameters
        from mcp_sdk.client.stdio import stdio_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("Warning: MCP SDK not installed. Install with: pip install mcp")
    print("Note: MCP functionality will be disabled. The system will work without MCP processing.")


class MCPClient:
    """Client for interacting with MCP servers."""
    
    def __init__(self, server_command: Optional[str] = None, server_args: Optional[list] = None):
        """
        Initialize MCP client.
        
        Args:
            server_command: Command to run the MCP server (e.g., "python", "node")
            server_args: Arguments for the server command (e.g., ["-m", "mcp_server"])
        """
        self.server_command = server_command
        self.server_args = server_args or []
        self.session: Optional[ClientSession] = None
        self.available = MCP_AVAILABLE
    
    async def connect(self):
        """Connect to the MCP server."""
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
                # User can configure their own MCP server connection
                return False
        except Exception as e:
            print(f"MCP connection error: {e}")
            return False
    
    async def process_context(self, query: str, retrieved_context: str) -> Dict[str, Any]:
        """
        Process retrieved context through MCP server before web search.
        
        Args:
            query: The original search query
            retrieved_context: Context retrieved from RAG/text retrieval
            
        Returns:
            Dictionary with processed context and metadata
        """
        if not self.available or not self.session:
            # If MCP is not available, return original context
            return {
                "processed_context": retrieved_context,
                "enhanced": False,
                "mcp_metadata": {}
            }
        
        try:
            # Call MCP tools/resources to enhance context
            # This is a placeholder - adjust based on your MCP server's available tools
            tools = await self.session.list_tools()
            
            # Example: Use a context enhancement tool if available
            enhanced_context = retrieved_context
            mcp_metadata = {
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
                                mcp_metadata["tool_used"] = tool.name
                                mcp_metadata["enhanced"] = True
                                break
                        except Exception as e:
                            print(f"Error calling MCP tool {tool.name}: {e}")
            
            return {
                "processed_context": enhanced_context,
                "enhanced": mcp_metadata.get("enhanced", False),
                "mcp_metadata": mcp_metadata
            }
            
        except Exception as e:
            print(f"MCP processing error: {e}")
            return {
                "processed_context": retrieved_context,
                "enhanced": False,
                "mcp_metadata": {"error": str(e)}
            }
    
    async def close(self):
        """Close the MCP session."""
        if self.session:
            try:
                await self.session.close()
            except Exception as e:
                print(f"Error closing MCP session: {e}")


# Global MCP client instance (can be configured)
_mcp_client: Optional[MCPClient] = None


def get_mcp_client() -> Optional[MCPClient]:
    """Get the global MCP client instance."""
    return _mcp_client


def initialize_mcp_client(server_command: Optional[str] = None, server_args: Optional[list] = None):
    """
    Initialize the global MCP client.
    
    Args:
        server_command: Command to run the MCP server
        server_args: Arguments for the server command
    """
    global _mcp_client
    _mcp_client = MCPClient(server_command, server_args)


async def process_with_mcp(query: str, retrieved_context: str) -> Dict[str, Any]:
    """
    Convenience function to process context through MCP.
    
    Args:
        query: The search query
        retrieved_context: Context from text retrieval
        
    Returns:
        Processed context dictionary
    """
    client = get_mcp_client()
    if not client:
        return {
            "processed_context": retrieved_context,
            "enhanced": False,
            "mcp_metadata": {"status": "no_client"}
        }
    
    # Ensure client is connected
    if not client.session:
        await client.connect()
    
    return await client.process_context(query, retrieved_context)

