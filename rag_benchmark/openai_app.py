"""
OpenAI API streaming RAG application for benchmark testing.
"""

import asyncio
import time
import numpy as np
from openai import AsyncOpenAI
from langchain_openai import OpenAIEmbeddings
from knowledge_base import get_knowledge_base, get_test_queries


class OpenAIRAGApp:
    def __init__(self):
        self.client = AsyncOpenAI()
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.knowledge_base = get_knowledge_base()
        self.document_embeddings = None
        
    async def initialize(self):
        """Initialize embeddings."""
        # Create embeddings for knowledge base
        self.document_embeddings = await self.embeddings.aembed_documents(self.knowledge_base)
        self.document_embeddings = np.array(self.document_embeddings)
        
    async def _retrieve_documents(self, query: str) -> str:
        """Retrieve relevant documents based on the query."""
        # Get query embedding
        query_embedding = await self.embeddings.aembed_query(query)
        query_embedding = np.array(query_embedding)
        
        # Calculate similarities
        similarities = np.dot(self.document_embeddings, query_embedding)
        
        # Get top 3 most relevant documents
        top_indices = np.argsort(similarities)[-3:][::-1]
        relevant_docs = [self.knowledge_base[i] for i in top_indices]
        
        return "\n\n".join(relevant_docs)
    
    async def stream_response(self, query: str):
        """Stream response for a given query."""
        # Retrieve relevant context
        context = await self._retrieve_documents(query)
        
        # Create enhanced prompt
        enhanced_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nPlease provide a helpful answer based on the context provided."
        
        # Stream response from OpenAI
        stream = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use the provided context to answer questions accurately."},
                {"role": "user", "content": enhanced_prompt}
            ],
            stream=True,
            temperature=0
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    
    async def get_response(self, query: str) -> tuple[str, float]:
        """Get complete response and measure latency."""
        start_time = time.time()
        
        response_parts = []
        async for chunk in self.stream_response(query):
            response_parts.append(chunk)
        
        end_time = time.time()
        latency = end_time - start_time
        
        return "".join(response_parts), latency


