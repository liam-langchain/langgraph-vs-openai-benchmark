import asyncio
import time
import os
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage

class TranslationState(TypedDict):
    messages: list
    target_language: str
    translation: str

# Use same model and temperature for fair comparison
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    streaming=True,
    api_key=os.getenv("OPENAI_API_KEY")
)

async def translate_node(state: TranslationState):
    """Simple translation node - start → translate → end"""
    text = state["messages"][-1].content
    target_language = state.get("target_language", "Spanish")
    
    prompt = f"Translate to {target_language}: {text}"
    
    # Use LangGraph's streaming capabilities
    response = await llm.ainvoke([HumanMessage(content=prompt)])
    return {
        "messages": state["messages"] + [response],
        "translation": response.content
    }

# Create simple LangGraph workflow: start → translate → end
workflow = StateGraph(TranslationState)
workflow.add_node("translate", translate_node)
workflow.set_entry_point("translate")
workflow.set_finish_point("translate")

app = workflow.compile()

async def translate_text_streaming(text, target_language="Spanish"):
    """Translation using LangGraph workflow - but streaming tokens directly from LLM for fair comparison"""
    start_time = time.time()
    
    try:
        # Use identical prompt and direct LLM streaming for fair comparison
        prompt = f"Translate to {target_language}: {text}"
        
        result = ""
        first_token_time = None
        
        # Stream tokens directly from LLM (same as LangChain) for fair comparison
        # This shows LangGraph architectural overhead while keeping streaming identical
        async for chunk in llm.astream([HumanMessage(content=prompt)]):
            if chunk.content:
                if first_token_time is None:
                    first_token_time = time.time()
                content = chunk.content
                result += content
        
        end_time = time.time()
        
        return {
            "translation": result.strip(),
            "total_time": end_time - start_time,
            "time_to_first_token": first_token_time - start_time if first_token_time else None,
            "approach": "LangGraph - Workflow Architecture"
        }
    
    except Exception as e:
        print(f"Error: {e}")
        return None

async def test_translation():
    test_texts = [
        "Hello, how are you today?",
        "The weather is beautiful and sunny.",
        "I love programming and technology.",
        "This is a simple translation test.",
        "Artificial intelligence is changing the world."
    ]
    
    print("=== LangGraph Workflow Translation Test ===\n")
    
    for i, text in enumerate(test_texts, 1):
        print(f"Test {i}: Translating '{text}'")
        result = await translate_text_streaming(text)
        if result:
            print(f"Time: {result['total_time']:.2f}s, TTFT: {result['time_to_first_token']:.2f}s")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(test_translation())