import json
from typing import Any

import faiss
import numpy as np
from aidial_client import AsyncDial
from aidial_sdk.chat_completion import Message, Role
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.tools.rag.document_cache import DocumentCache
from task.utils.dial_file_conent_extractor import DialFileContentExtractor

# System prompt for Generation step
_SYSTEM_PROMPT = """
You are a helpful assistant that answers questions based on the provided context.
Use ONLY the information from the context to answer the question.
If the answer cannot be found in the context, say "I couldn't find this information in the document."
Be concise and accurate in your responses.
"""


class RagTool(BaseTool):
    """
    Performs semantic search on documents to find and answer questions based on relevant content.
    Supports: PDF, TXT, CSV, HTML.
    """

    def __init__(self, endpoint: str, deployment_name: str, document_cache: DocumentCache):
        # 1. Set endpoint
        self.endpoint = endpoint
        # 2. Set deployment_name
        self.deployment_name = deployment_name
        # 3. Set document_cache
        self.document_cache = document_cache
        # 4. Create SentenceTransformer model
        self.model = SentenceTransformer(
            model_name_or_path='all-MiniLM-L6-v2',
            device='cpu'
        )
        # 5. Create RecursiveCharacterTextSplitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    @property
    def show_in_stage(self) -> bool:
        # Set as False since we will have custom variant of representation in Stage
        return False

    @property
    def name(self) -> str:
        return "rag_search"

    @property
    def description(self) -> str:
        return (
            "Performs semantic search on documents to find relevant content and answer questions. "
            "Use this tool when you need to find specific information in large documents. "
            "It indexes the document, finds the most relevant sections, and generates an answer based on them. "
            "Supports PDF, TXT, CSV, and HTML files. Best for questions about document content."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "request": {
                    "type": "string",
                    "description": "The search query or question to search for in the document"
                },
                "file_url": {
                    "type": "string",
                    "description": "The URL of the file to search in"
                }
            },
            "required": ["request", "file_url"]
        }

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        # 1. Load arguments
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        
        # 2. Get request from arguments
        request = arguments.get("request")
        
        # 3. Get file_url from arguments
        file_url = arguments.get("file_url")
        
        # 4. Get stage from tool_call_params
        stage = tool_call_params.stage
        
        # 5. Append content to stage: request header
        stage.append_content("## Request arguments: \n")
        
        # 6. Append request to stage
        stage.append_content(f"**Request**: {request}\n\r")
        
        # 7. Append file URL to stage
        stage.append_content(f"**File URL**: {file_url}\n\r")
        
        # 8. Create cache_document_key
        cache_document_key = f"{tool_call_params.conversation_id}:{file_url}"
        
        # 9. Get from document_cache by cache_document_key
        cached_data = self.document_cache.get(cache_document_key)
        
        # 10. Use cache or create new index
        if cached_data:
            index, chunks = cached_data
        else:
            # Create DialFileContentExtractor and extract text
            extractor = DialFileContentExtractor(
                endpoint=self.endpoint,
                api_key=tool_call_params.api_key
            )
            text_content = extractor.extract_text(file_url)
            
            # If no text_content
            if not text_content:
                stage.append_content("**Error**: File content not found.\n\r")
                return "Error: File content not found."
            
            # Create chunks with text_splitter
            chunks = self.text_splitter.split_text(text_content)
            
            # Create embeddings with model
            embeddings = self.model.encode(chunks)
            
            # Create IndexFlatL2 with 384 dimensions
            index = faiss.IndexFlatL2(384)
            
            # Add to index
            index.add(np.array(embeddings).astype('float32'))
            
            # Add to document_cache
            self.document_cache.set(cache_document_key, index, chunks)
        
        # 11. Prepare query_embedding
        query_embedding = self.model.encode([request]).astype('float32')
        
        # 12. Search through index
        distances, indices = index.search(query_embedding, k=3)
        
        # 13. Get retrieved chunks
        retrieved_chunks = [chunks[idx] for idx in indices[0] if idx < len(chunks)]
        
        # 14. Make augmentation
        augmented_prompt = self.__augmentation(request, retrieved_chunks)
        
        # 15. Append content to stage
        stage.append_content("## RAG Request: \n")
        
        # 16. Append augmented prompt to stage
        stage.append_content(f"```text\n\r{augmented_prompt}\n\r```\n\r")
        
        # 17. Append response header to stage
        stage.append_content("## Response: \n")
        
        # 18. Make Generation with AsyncDial
        client = AsyncDial(
            base_url=self.endpoint,
            api_key=tool_call_params.api_key,
        )
        
        chunks_stream = await client.chat.completions.create(
            deployment_name=self.deployment_name,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": augmented_prompt}
            ],
            stream=True,
        )
        
        # Stream response to stage and collect content
        collected_content = ""
        async for chunk in chunks_stream:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    stage.append_content(delta.content)
                    collected_content += delta.content
        
        # 19. Return collected content
        return collected_content

    def __augmentation(self, request: str, chunks: list[str]) -> str:
        # Make prompt augmentation
        context = "\n\n---\n\n".join(chunks)
        return f"""Based on the following context, answer the question.

Context:
{context}

Question: {request}

Answer:"""
