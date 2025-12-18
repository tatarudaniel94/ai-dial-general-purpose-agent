import asyncio
import json
from typing import Any

from aidial_client import AsyncDial
from aidial_client.types.chat.legacy.chat_completion import CustomContent, ToolCall
from aidial_sdk.chat_completion import Message, Role, Choice, Request, Response

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.utils.constants import TOOL_CALL_HISTORY_KEY
from task.utils.history import unpack_messages
from task.utils.stage import StageProcessor


class GeneralPurposeAgent:

    def __init__(
            self,
            endpoint: str,
            system_prompt: str,
            tools: list[BaseTool],
    ):
        self.endpoint = endpoint
        self.system_prompt = system_prompt
        self.tools = tools
        
        # Prepare tools_dict for faster lookup by tool name
        self._tools_dict: dict[str, BaseTool] = {tool.name: tool for tool in tools}
        
        # Create state dict with tool call history
        self.state: dict[str, Any] = {TOOL_CALL_HISTORY_KEY: []}

    async def handle_request(self, deployment_name: str, choice: Choice, request: Request, response: Response) -> Message:
        # 1. Create AsyncDial client
        api_key = request.api_key or ""
        api_version = getattr(request, 'api_version', None)
        
        client = AsyncDial(
            base_url=self.endpoint,
            api_key=api_key,
        )
        
        # 2. Create chunks with chat completions
        messages = self._prepare_messages(request.messages)
        tool_schemas = [tool.schema for tool in self.tools] if self.tools else None
        
        chunks = await client.chat.completions.create(
            messages=messages,
            tools=tool_schemas,
            deployment_name=deployment_name,
            stream=True,
        )
        
        # 3. Create tool_call_index_map and content collector
        tool_call_index_map: dict[int, Any] = {}
        content = ""
        
        # 4. Async loop through chunks
        async for chunk in chunks:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta:
                    # Handle content streaming
                    if delta.content:
                        choice.append_content(delta.content)
                        content += delta.content
                    
                    # Handle tool calls streaming
                    if delta.tool_calls:
                        for tool_call_delta in delta.tool_calls:
                            if tool_call_delta.id:
                                # First chunk of tool call - add to map
                                tool_call_index_map[tool_call_delta.index] = tool_call_delta
                            else:
                                # Subsequent chunks - append arguments
                                existing_tool_call = tool_call_index_map.get(tool_call_delta.index)
                                if existing_tool_call and tool_call_delta.function:
                                    argument_chunk = tool_call_delta.function.arguments or ""
                                    if existing_tool_call.function:
                                        existing_tool_call.function.arguments = (
                                            (existing_tool_call.function.arguments or "") + argument_chunk
                                        )
        
        # 5. Create assistant_message
        tool_calls_list = None
        if tool_call_index_map:
            tool_calls_list = [
                ToolCall.validate(tc) for tc in tool_call_index_map.values()
            ]
        
        assistant_message = Message(
            role=Role.ASSISTANT,
            content=content if content else None,
            tool_calls=tool_calls_list,
        )
        
        # 6. Check if we need to process tool calls
        if assistant_message.tool_calls:
            conversation_id = request.headers.get("x-conversation-id", "")
            
            # Create tasks for parallel tool execution
            tasks = [
                self._process_tool_call(tool_call, choice, api_key, conversation_id)
                for tool_call in assistant_message.tool_calls
            ]
            
            # Execute tasks in parallel
            tool_messages = await asyncio.gather(*tasks)
            
            # Update state with assistant message and tool responses
            self.state[TOOL_CALL_HISTORY_KEY].append(
                assistant_message.dict(exclude_none=True)
            )
            self.state[TOOL_CALL_HISTORY_KEY].extend(tool_messages)
            
            # Recursive call to continue conversation
            return await self.handle_request(deployment_name, choice, request, response)
        
        # 7. No tool calls - set state and return final message
        choice.set_state(self.state)
        return assistant_message

    def _prepare_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        # 1. Unpack messages
        unpacked = unpack_messages(messages, self.state.get(TOOL_CALL_HISTORY_KEY, []))
        
        # 2. Insert system prompt as first message
        unpacked.insert(0, {"role": "system", "content": self.system_prompt})
        
        # 3. Print history for debugging
        for msg in unpacked:
            print(json.dumps(msg, default=str))
        
        # 4. Return unpacked messages
        return unpacked

    async def _process_tool_call(self, tool_call: ToolCall, choice: Choice, api_key: str, conversation_id: str) -> dict[str, Any]:
        # 1. Get tool name
        tool_name = tool_call.function.name
        
        # 2. Open stage
        stage = StageProcessor.open_stage(choice, tool_name)
        
        # 3. Get tool from tools dict
        tool = self._tools_dict.get(tool_name)
        
        if tool:
            # 4. Show request in stage if enabled
            if tool.show_in_stage:
                stage.append_content("## Request arguments: \n")
                stage.append_content(
                    f"```json\n\r{json.dumps(json.loads(tool_call.function.arguments), indent=2)}\n\r```\n\r"
                )
                stage.append_content("## Response: \n")
            
            # 5. Execute tool
            tool_call_params = ToolCallParams(
                tool_call=tool_call,
                stage=stage,
                choice=choice,
                api_key=api_key,
                conversation_id=conversation_id,
            )
            tool_message = await tool.execute(tool_call_params)
        else:
            # Tool not found
            tool_message = Message(
                role=Role.TOOL,
                content=f"Error: Tool '{tool_name}' not found",
                tool_call_id=tool_call.id,
            )
        
        # 6. Close stage
        StageProcessor.close_stage_safely(stage)
        
        # 7. Return tool message as dict
        return tool_message.dict(exclude_none=True)
