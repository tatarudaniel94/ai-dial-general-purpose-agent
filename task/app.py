import os

import uvicorn
from aidial_sdk import DIALApp
from aidial_sdk.chat_completion import ChatCompletion, Request, Response

from task.agent import GeneralPurposeAgent
from task.prompts import SYSTEM_PROMPT
from task.tools.base import BaseTool
from task.tools.deployment.image_generation_tool import ImageGenerationTool
from task.tools.deployment.web_search_tool import WebSearchTool
from task.tools.files.file_content_extraction_tool import FileContentExtractionTool
from task.tools.py_interpreter.python_code_interpreter_tool import PythonCodeInterpreterTool
from task.tools.mcp.mcp_client import MCPClient
from task.tools.mcp.mcp_tool import MCPTool
from task.tools.rag.document_cache import DocumentCache
from task.tools.rag.rag_tool import RagTool

DIAL_ENDPOINT = os.getenv('DIAL_ENDPOINT', "http://localhost:8080")
# DEPLOYMENT_NAME = os.getenv('DEPLOYMENT_NAME', 'gpt-4o')
DEPLOYMENT_NAME = os.getenv('DEPLOYMENT_NAME', 'claude-sonnet-3-7')


class GeneralPurposeAgentApplication(ChatCompletion):

    def __init__(self):
        self.tools: list[BaseTool] = []

    async def _get_mcp_tools(self, url: str) -> list[BaseTool]:
        # 1. Create list of BaseTool
        tools: list[BaseTool] = []
        
        # 2. Create MCPClient
        mcp_client = MCPClient(url)
        
        # 3. Get tools and add them to the list as MCPTool
        mcp_tools = await mcp_client.get_tools()
        for mcp_tool_model in mcp_tools:
            tools.append(MCPTool(client=mcp_client, mcp_tool_model=mcp_tool_model))
        
        # 4. Return created tool list
        return tools

    async def _create_tools(self) -> list[BaseTool]:
        # 1. Create list of BaseTool
        tools: list[BaseTool] = []
        
        # Tools initialization can be skipped for now (Step 1)
        # We will add tools in later steps as they are implemented
        
        # 2. Add ImageGenerationTool with DIAL_ENDPOINT
        tools.append(ImageGenerationTool(endpoint=DIAL_ENDPOINT))
        
        # 2b. Add WebSearchTool with DIAL_ENDPOINT (uses Gemini with Google Search grounding)
        tools.append(WebSearchTool(endpoint=DIAL_ENDPOINT))
        
        # 3. Add FileContentExtractionTool with DIAL_ENDPOINT
        tools.append(FileContentExtractionTool(endpoint=DIAL_ENDPOINT))
        
        # 4. Add RagTool with DIAL_ENDPOINT, DEPLOYMENT_NAME, and DocumentCache
        document_cache = DocumentCache.create()
        tools.append(RagTool(
            endpoint=DIAL_ENDPOINT,
            deployment_name=DEPLOYMENT_NAME,
            document_cache=document_cache
        ))
        
        # 5. Add PythonCodeInterpreterTool
        py_interpreter = await PythonCodeInterpreterTool.create(
            mcp_url="http://localhost:8050/mcp",
            tool_name="execute_code",
            dial_endpoint=DIAL_ENDPOINT,
        )
        tools.append(py_interpreter)
        
        # 6. Extend tools with MCP tools from DDG search server
        mcp_tools = await self._get_mcp_tools("http://localhost:8051/mcp")
        tools.extend(mcp_tools)
        
        return tools

    async def chat_completion(self, request: Request, response: Response) -> None:
        # 1. If tools are absent, create them
        if not self.tools:
            self.tools = await self._create_tools()
        
        # 2. Create choice and handle request
        with response.create_single_choice() as choice:
            agent = GeneralPurposeAgent(
                endpoint=DIAL_ENDPOINT,
                system_prompt=SYSTEM_PROMPT,
                tools=self.tools,
            )
            await agent.handle_request(
                choice=choice,
                deployment_name=DEPLOYMENT_NAME,
                request=request,
                response=response,
            )


# 1. Create DIALApp
app = DIALApp()

# 2. Create GeneralPurposeAgentApplication
agent_app = GeneralPurposeAgentApplication()

# 3. Add chat_completion to DIALApp
app.add_chat_completion(
    deployment_name="general-purpose-agent",
    impl=agent_app,
)

# 4. Run with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, port=5030, host="0.0.0.0")
