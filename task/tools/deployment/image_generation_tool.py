from typing import Any

from aidial_sdk.chat_completion import Message
from pydantic import StrictStr

from task.tools.deployment.base import DeploymentTool
from task.tools.models import ToolCallParams


class ImageGenerationTool(DeploymentTool):

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        # 1. Call parent function _execute and get result
        result = await super()._execute(tool_call_params)
        
        # 2. If attachments are present, filter only images and add to choice
        if result.custom_content and result.custom_content.attachments:
            image_attachments = [
                att for att in result.custom_content.attachments
                if att.type in ["image/png", "image/jpeg"]
            ]
            
            # 3. Append images as content to choice
            for attachment in image_attachments:
                if attachment.url:
                    tool_call_params.choice.append_content(f"\n\r![image]({attachment.url})\n\r")
        
        # 4. If message content is absent, add instruction
        if not result.content:
            result.content = StrictStr("The image has been successfully generated according to request and shown to user!")
        
        return result

    @property
    def deployment_name(self) -> str:
        return "dall-e-3"

    @property
    def name(self) -> str:
        return "image_generation"

    @property
    def description(self) -> str:
        return (
            "Generates images based on text descriptions using DALL-E-3. "
            "Provide a detailed prompt describing the image you want to create. "
            "The more specific the description, the better the result. "
            "Supports different sizes and quality settings."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Extensive description of the image that should be generated."
                },
                "size": {
                    "type": "string",
                    "enum": ["1024x1024", "1792x1024", "1024x1792"],
                    "description": "The size of the generated image. Default is 1024x1024."
                },
                "quality": {
                    "type": "string",
                    "enum": ["standard", "hd"],
                    "description": "The quality of the generated image. 'hd' creates higher quality images. Default is 'standard'."
                },
                "style": {
                    "type": "string",
                    "enum": ["vivid", "natural"],
                    "description": "The style of the generated image. 'vivid' generates hyper-real images, 'natural' produces more natural-looking images. Default is 'vivid'."
                }
            },
            "required": ["prompt"]
        }
