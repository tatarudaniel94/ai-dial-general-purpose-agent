import json
from typing import Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Role
from pydantic import StrictStr

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams


class WebSearchTool(BaseTool):
    """
    Tool for WEB searching using Gemini with Google Search grounding.
    """

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.deployment_name = "gemini-2.5-pro"

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Searches the web for real-time information using Google Search. "
            "Use this tool when you need current information, news, weather, "
            "or any data that might not be in your training data. "
            "Provides up-to-date answers based on live web search results."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The search query or question to search for on the web"
                }
            },
            "required": ["request"]
        }

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        # Load arguments
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        request = arguments.get("request", "")
        
        stage = tool_call_params.stage
        
        # Create AsyncDial client
        client = AsyncDial(
            base_url=self.endpoint,
            api_key=tool_call_params.api_key,
        )
        
        # Call chat completions with Google Search grounding tool
        chunks = await client.chat.completions.create(
            messages=[
                {"role": "user", "content": request}
            ],
            stream=True,
            deployment_name=self.deployment_name,
            temperature=0,
            extra_body={
                "tools": [
                    {
                        "type": "static_function",
                        "static_function": {
                            "name": "google_search",
                            "description": "Grounding with Google Search",
                            "configuration": {}
                        }
                    }
                ]
            }
        )
        
        # Collect and stream content
        content = ""
        async for chunk in chunks:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    content += delta.content
                    stage.append_content(delta.content)
        
        if not content:
            content = "No results found for the search query."
        
        return content

