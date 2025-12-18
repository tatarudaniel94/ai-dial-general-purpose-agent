import json
from abc import ABC, abstractmethod
from typing import Any

from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Role, CustomContent, Attachment
from pydantic import StrictStr

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams


class DeploymentTool(BaseTool, ABC):

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    @property
    @abstractmethod
    def deployment_name(self) -> str:
        pass

    @property
    def tool_parameters(self) -> dict[str, Any]:
        return {}

    @property
    def system_prompt(self) -> str | None:
        """Optional system prompt for the deployment. Override in subclasses if needed."""
        return None

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        # 1. Load arguments with json
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        
        # 2. Get prompt from arguments
        prompt = arguments.get("prompt", "")
        
        # 3. Delete prompt from arguments (remaining are custom_fields)
        if "prompt" in arguments:
            del arguments["prompt"]
        
        # 4. Create AsyncDial client
        client = AsyncDial(
            base_url=self.endpoint,
            api_key=tool_call_params.api_key,
        )
        
        # 5. Build messages list
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Call chat completions
        extra_body = {"custom_fields": arguments} if arguments else None
        
        chunks = await client.chat.completions.create(
            messages=messages,
            stream=True,
            deployment_name=self.deployment_name,
            extra_body=extra_body,
            **self.tool_parameters
        )
        
        # 6. Collect content and attachments
        content = ""
        attachments = []
        stage = tool_call_params.stage
        
        async for chunk in chunks:
            if chunk.choices:
                choice = chunk.choices[0]
                delta = choice.delta
                
                if delta:
                    # Collect content
                    if delta.content:
                        content += delta.content
                        stage.append_content(delta.content)
                    
                    # Collect attachments from custom_content
                    if hasattr(delta, 'custom_content') and delta.custom_content:
                        if hasattr(delta.custom_content, 'attachments') and delta.custom_content.attachments:
                            for attachment in delta.custom_content.attachments:
                                # Only add attachments that have url or data
                                att_url = getattr(attachment, 'url', None)
                                att_data = getattr(attachment, 'data', None)
                                if att_url or att_data:
                                    attachments.append(attachment)
                                    # Add attachment to stage only if it has url
                                    if att_url:
                                        stage.add_attachment(
                                            type=getattr(attachment, 'type', None),
                                            title=getattr(attachment, 'title', None),
                                            url=att_url,
                                            reference_url=getattr(attachment, 'reference_url', None),
                                        )
        
        # 7. Return Message with tool role, content, custom_content and tool_call_id
        custom_content = None
        if attachments:
            # Convert attachments to Attachment objects with valid url/data
            valid_attachments = []
            for att in attachments:
                att_url = getattr(att, 'url', None)
                att_data = getattr(att, 'data', None)
                if att_url or att_data:
                    valid_attachments.append(Attachment(
                        type=getattr(att, 'type', None),
                        title=getattr(att, 'title', None),
                        url=att_url,
                        data=att_data,
                        reference_url=getattr(att, 'reference_url', None),
                    ))
            if valid_attachments:
                custom_content = CustomContent(attachments=valid_attachments)
        
        return Message(
            role=Role.TOOL,
            content=StrictStr(content) if content else None,
            custom_content=custom_content,
            tool_call_id=StrictStr(tool_call_params.tool_call.id),
        )
