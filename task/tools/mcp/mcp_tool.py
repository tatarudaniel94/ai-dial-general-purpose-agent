import json
from typing import Any

from aidial_sdk.chat_completion import Message

from task.tools.base import BaseTool
from task.tools.mcp.mcp_client import MCPClient
from task.tools.mcp.mcp_tool_model import MCPToolModel
from task.tools.models import ToolCallParams


class MCPTool(BaseTool):

    def __init__(self, client: MCPClient, mcp_tool_model: MCPToolModel):
        # 1. Set client
        self.client = client
        # 2. Set mcp_tool_model
        self.mcp_tool_model = mcp_tool_model

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        # 1. Load arguments with json
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        
        # 2. Get content with mcp client tool call
        content = await self.client.call_tool(self.name, arguments)
        
        # 3. Append retrieved content to stage
        tool_call_params.stage.append_content(str(content))
        
        # 4. Return content
        return str(content)

    @property
    def name(self) -> str:
        # Provide name from mcp_tool_model
        return self.mcp_tool_model.name

    @property
    def description(self) -> str:
        # Provide description from mcp_tool_model
        return self.mcp_tool_model.description

    @property
    def parameters(self) -> dict[str, Any]:
        # Provide parameters from mcp_tool_model
        return self.mcp_tool_model.parameters
