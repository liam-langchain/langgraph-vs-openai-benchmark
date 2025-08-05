"""
LangGraph streaming RAG application for benchmark testing.
"""

import asyncio
import time
from typing import TypedDict, List, Union
import numpy as np
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from knowledge_base import get_knowledge_base, get_test_queries


class RAGState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    documents: List[str]
    query: str
    context: str


class LangGraphRAGApp:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.knowledge_base = get_knowledge_base()
        self.document_embeddings = None
        self.graph = None
        
    async def initialize(self):
        """Initialize embeddings and build graph."""
        # Create embeddings for knowledge base
        self.document_embeddings = await self.embeddings.aembed_documents(self.knowledge_base)
        self.document_embeddings = np.array(self.document_embeddings)
        
        # Build the graph
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(MessagesState)
        
        # Add node
        workflow.add_node("generate", self._generate_response)
        
        # Add edges
        workflow.add_edge(START, "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
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
    
    async def _generate_response(self, state: MessagesState) -> MessagesState:
        """Generate streaming response using the LLM."""
        response = await self.llm.ainvoke(state["messages"])
        return {"messages": state["messages"] + [response]}
    
    async def stream_response(self, query: str):
        """Stream response for a given query."""
        # Retrieve relevant context 
        context = await self._retrieve_documents(query)
        
        # Create enhanced prompt 
        enhanced_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nPlease provide a helpful answer based on the context provided."
        
        # Create messages with system message 
        initial_state = {"messages": [
            HumanMessage(content="You are a helpful assistant. Use the provided context to answer questions accurately."),
            HumanMessage(content=enhanced_prompt)
        ]}
        
        # Use stream_mode="messages" 
        async for msg, metadata in self.graph.astream(initial_state, stream_mode="messages"):
            # Stream tokens from the generate node 
            if hasattr(msg, 'content') and msg.content and metadata.get("langgraph_node") == "generate":
                yield msg.content
    
    async def get_response(self, query: str) -> tuple[str, float]:
        """Get complete response and measure latency."""
        start_time = time.time()
        
        response_parts = []
        async for chunk in self.stream_response(query):
            response_parts.append(chunk)
        
        end_time = time.time()
        latency = end_time - start_time
        
        return "".join(response_parts), latency


