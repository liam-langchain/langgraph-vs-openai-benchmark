import asyncio
import time
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# LangChain setup - identical model configuration
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    streaming=True,
    api_key=os.getenv("OPENAI_API_KEY")
)

async def translate_text_streaming(text, target_language="Spanish"):
    """Translation using LangChain with identical prompt to others"""
    start_time = time.time()
    
    try:
        # Use identical simple prompt as OpenAI Raw and LangGraph
        prompt = f"Translate to {target_language}: {text}"
        messages = [HumanMessage(content=prompt)]
        
        result = ""
        first_token_time = None
        
        # Stream with LangChain abstractions
        async for chunk in llm.astream(messages):
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
            "approach": "LangChain - Wrapper Overhead"
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
    
    print("=== LangGraph Translation Test ===\n")
    
    for i, text in enumerate(test_texts, 1):
        print(f"Test {i}: Translating '{text}'")
        result = await translate_text_streaming(text)
        if result:
            print(f"Time: {result['total_time']:.2f}s, TTFT: {result['time_to_first_token']:.2f}s")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(test_translation())