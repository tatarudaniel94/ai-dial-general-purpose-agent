from abc import ABC, abstractmethod
from typing import Any

from aidial_client.types.chat import ToolParam, FunctionParam
from aidial_client.types.chat.legacy.chat_completion import Role
from aidial_sdk.chat_completion import Message
from pydantic import StrictStr

from task.tools.models import ToolCallParams


class BaseTool(ABC):

    async def execute(self, tool_call_params: ToolCallParams) -> Message:
        # 1. Create Message obj with role, name, and tool_call_id
        message = Message(
            role=Role.TOOL,
            name=StrictStr(tool_call_params.tool_call.function.name),
            tool_call_id=StrictStr(tool_call_params.tool_call.id),
        )
        
        # 2. Template method pattern with try-except
        try:
            result = await self._execute(tool_call_params)
            if isinstance(result, Message):
                message = result
                # Ensure required fields are set
                message.role = Role.TOOL
                message.name = StrictStr(tool_call_params.tool_call.function.name)
                message.tool_call_id = StrictStr(tool_call_params.tool_call.id)
            else:
                message.content = StrictStr(result)
        except Exception as e:
            message.content = StrictStr(f"Error: {str(e)}")
        
        # 3. Return created message
        return message

    @abstractmethod
    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        pass

    @property
    def show_in_stage(self) -> bool:
        return True

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        pass

    @property
    def schema(self) -> ToolParam:
        """Provides tool schema according to DIAL specification."""
        return ToolParam(
            type="function",
            function=FunctionParam(
                name=self.name,
                description=self.description,
                parameters=self.parameters
            )
        )
