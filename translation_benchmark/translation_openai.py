import asyncio
import time
import os
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def translate_text_streaming(text, target_language="Spanish"):
    """Simple translation using raw OpenAI API - minimal overhead, basic functionality"""
    start_time = time.time()
    
    # Simple, direct prompt
    prompt = f"Translate to {target_language}: {text}"
    
    try:
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            temperature=0.3
        )
        
        result = ""
        first_token_time = None
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                if first_token_time is None:
                    first_token_time = time.time()
                content = chunk.choices[0].delta.content
                result += content
        
        end_time = time.time()
        
        return {
            "translation": result.strip(),
            "total_time": end_time - start_time,
            "time_to_first_token": first_token_time - start_time if first_token_time else None,
            "approach": "Raw OpenAI API - Simple Translation"
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
    
    print("=== OpenAI Direct API Translation Test ===\n")
    
    for i, text in enumerate(test_texts, 1):
        print(f"Test {i}: Translating '{text}'")
        result = await translate_text_streaming(text)
        if result:
            print(f"Time: {result['total_time']:.2f}s, TTFT: {result['time_to_first_token']:.2f}s")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(test_translation())