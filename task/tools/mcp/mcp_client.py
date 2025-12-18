from typing import Optional, Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult, TextContent, ReadResourceResult, TextResourceContents, BlobResourceContents
from pydantic import AnyUrl

from task.tools.mcp.mcp_tool_model import MCPToolModel


class MCPClient:
    """Handles MCP server connection and tool execution"""

    def __init__(self, mcp_server_url: str) -> None:
        self.server_url = mcp_server_url
        self.session: Optional[ClientSession] = None
        self._streams_context = None
        self._session_context = None

    @classmethod
    async def create(cls, mcp_server_url: str) -> 'MCPClient':
        """Async factory method to create and connect MCPClient"""
        # 1. Create instance of MCPClient
        instance = cls(mcp_server_url)
        # 2. Connect to MCP server
        await instance.connect()
        # 3. Return created instance
        return instance

    async def connect(self):
        """Connect to MCP server"""
        # 1. Check if session is present, if yes just return
        if self.session is not None:
            return
        
        # 2. Call streamablehttp_client method with server_url
        self._streams_context = streamablehttp_client(self.server_url)
        
        # 3. Enter _streams_context
        read_stream, write_stream, _ = await self._streams_context.__aenter__()
        
        # 4. Create ClientSession with streams
        self._session_context = ClientSession(read_stream, write_stream)
        
        # 5. Enter _session_context and set as session
        self.session = await self._session_context.__aenter__()
        
        # 6. Initialize session and print result
        init_result = await self.session.initialize()
        print(f"[MCPClient] Connected to {self.server_url}: {init_result}")

    async def get_tools(self) -> list[MCPToolModel]:
        """Get available tools from MCP server"""
        if not self.session:
            await self.connect()
        
        tools_result = await self.session.list_tools()
        
        mcp_tools = []
        for tool in tools_result.tools:
            mcp_tools.append(MCPToolModel(
                name=tool.name,
                description=tool.description or "",
                parameters=tool.inputSchema if tool.inputSchema else {}
            ))
        
        return mcp_tools

    async def call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> Any:
        """Call a tool on the MCP server"""
        if not self.session:
            await self.connect()
        
        result: CallToolResult = await self.session.call_tool(tool_name, tool_args)
        
        # Handle the result content array
        content_parts = []
        for content in result.content:
            if isinstance(content, TextContent):
                content_parts.append(content.text)
            else:
                # Handle other content types if needed
                content_parts.append(str(content))
        
        return "\n".join(content_parts)

    async def get_resource(self, uri: AnyUrl) -> str | bytes:
        """Get specific resource content"""
        if not self.session:
            await self.connect()
        
        result: ReadResourceResult = await self.session.read_resource(uri)
        
        # Resources can be TextResourceContents or BlobResourceContents
        for content in result.contents:
            if isinstance(content, TextResourceContents):
                return content.text
            elif isinstance(content, BlobResourceContents):
                return content.blob
        
        return ""

    async def close(self):
        """Close connection to MCP server"""
        # 1. Close _session_context
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        
        # 2. Close _streams_context
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)
        
        # 3. Set session, _session_context and _streams_context as None
        self.session = None
        self._session_context = None
        self._streams_context = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
        return False
