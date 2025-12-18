import base64
import json
from typing import Any, Optional

from aidial_client import Dial
from aidial_sdk.chat_completion import Message, Attachment
from pydantic import StrictStr, AnyUrl

from task.tools.base import BaseTool
from task.tools.py_interpreter._response import _ExecutionResult
from task.tools.mcp.mcp_client import MCPClient
from task.tools.mcp.mcp_tool_model import MCPToolModel
from task.tools.models import ToolCallParams


class PythonCodeInterpreterTool(BaseTool):
    """
    Uses https://github.com/khshanovskyi/mcp-python-code-interpreter PyInterpreter MCP Server.

    ⚠️ Pay attention that this tool will wrap all the work with PyInterpreter MCP Server.
    """

    def __init__(
            self,
            mcp_client: MCPClient,
            mcp_tool_models: list[MCPToolModel],
            tool_name: str,
            dial_endpoint: str,
    ):
        """
        :param tool_name: it must be actual name of tool that executes code. It is 'execute_code'.
            https://github.com/khshanovskyi/mcp-python-code-interpreter/blob/main/interpreter/server.py#L303
        """
        # 1. Set dial_endpoint
        self.dial_endpoint = dial_endpoint
        
        # 2. Set mcp_client
        self.mcp_client = mcp_client
        
        # 3. Set _code_execute_tool: find the tool matching tool_name
        self._code_execute_tool: Optional[MCPToolModel] = None
        for tool_model in mcp_tool_models:
            if tool_model.name == tool_name:
                self._code_execute_tool = tool_model
                break
        
        # 4. If _code_execute_tool is null then raise error
        if self._code_execute_tool is None:
            raise ValueError(f"Tool '{tool_name}' not found in MCP tools. Cannot set up PythonCodeInterpreterTool.")

    @classmethod
    async def create(
            cls,
            mcp_url: str,
            tool_name: str,
            dial_endpoint: str,
    ) -> 'PythonCodeInterpreterTool':
        """Async factory method to create PythonCodeInterpreterTool"""
        # 1. Create MCPClient
        mcp_client = await MCPClient.create(mcp_url)
        
        # 2. Get tools
        mcp_tool_models = await mcp_client.get_tools()
        
        # 3. Create PythonCodeInterpreterTool instance and return it
        return cls(
            mcp_client=mcp_client,
            mcp_tool_models=mcp_tool_models,
            tool_name=tool_name,
            dial_endpoint=dial_endpoint,
        )

    @property
    def show_in_stage(self) -> bool:
        # Set as False since we will have custom variant of representation in Stage
        return False

    @property
    def name(self) -> str:
        # Provide _code_execute_tool name
        return self._code_execute_tool.name

    @property
    def description(self) -> str:
        # Provide _code_execute_tool description
        return self._code_execute_tool.description

    @property
    def parameters(self) -> dict[str, Any]:
        # Provide _code_execute_tool parameters
        return self._code_execute_tool.parameters

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        # 1. Load arguments with json
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        
        # 2. Get code from arguments
        code = arguments.get("code", "")
        
        # 3. Get session_id from arguments (optional)
        session_id = arguments.get("session_id")
        
        # 4. Get stage from tool_call_params
        stage = tool_call_params.stage
        
        # 5. Append content to stage: request header
        stage.append_content("## Request arguments: \n")
        
        # 6. Append code to stage as python markdown
        stage.append_content(f"```python\n\r{code}\n\r```\n\r")
        
        # 7. Append session info to stage
        if session_id and session_id != 0:
            stage.append_content(f"**session_id**: {session_id}\n\r")
        else:
            stage.append_content("New session will be created\n\r")
        
        # 8. Make tool call
        result = await self.mcp_client.call_tool(self.name, arguments)
        
        # 9. Load retrieved response as json
        result_json = json.loads(result)
        
        # 10. Validate result with _ExecutionResult
        execution_result = _ExecutionResult.model_validate(result_json)
        
        # 11. If execution_result contains files, process them
        if execution_result.files:
            # Create Dial client
            dial_client = Dial(
                base_url=self.dial_endpoint,
                api_key=tool_call_params.api_key,
            )
            
            # Get my_appdata_home path as files_home
            files_home = dial_client.my_appdata_home()
            
            # Iterate through files
            for file_ref in execution_result.files:
                file_name = file_ref.name
                mime_type = file_ref.mime_type
                
                # Get resource with mcp client by URL from file
                resource_content = await self.mcp_client.get_resource(AnyUrl(file_ref.uri))
                
                # Check if text or binary content
                text_types = ['text/', 'application/json', 'application/xml']
                is_text = any(mime_type.startswith(t) for t in text_types)
                
                if is_text:
                    # Text content - encode as bytes
                    if isinstance(resource_content, str):
                        file_bytes = resource_content.encode('utf-8')
                    else:
                        file_bytes = resource_content
                else:
                    # Binary content - decode from base64
                    if isinstance(resource_content, str):
                        file_bytes = base64.b64decode(resource_content)
                    else:
                        file_bytes = base64.b64decode(resource_content)
                
                # Prepare URL to upload downloaded file
                upload_url = f"files/{(files_home / file_name).as_posix()}"
                
                # Upload file with DIAL client
                dial_client.files.upload(upload_url, (file_name, file_bytes, mime_type))
                
                # Prepare Attachment
                attachment = Attachment(
                    url=upload_url,
                    type=mime_type,
                    title=file_name,
                )
                
                # Add attachment to stage and choice
                stage.add_attachment(
                    url=upload_url,
                    type=mime_type,
                    title=file_name,
                )
                tool_call_params.choice.add_attachment(
                    url=upload_url,
                    type=mime_type,
                    title=file_name,
                )
        
        # 12. Truncate output to avoid high costs and context window overload
        if execution_result.output:
            truncated_output = []
            for output_item in execution_result.output:
                if len(output_item) > 1000:
                    truncated_output.append(output_item[:1000] + "... [truncated]")
                else:
                    truncated_output.append(output_item)
            execution_result.output = truncated_output
        
        # 13. Append result to stage as json
        stage.append_content(f"## Response:\n```json\n\r{execution_result.model_dump_json(indent=2)}\n\r```\n\r")
        
        # 14. Return execution result as string
        return execution_result.model_dump_json()
