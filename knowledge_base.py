"""
Simple knowledge base for RAG benchmark testing.
Contains sample documents about AI and technology topics.
"""

SAMPLE_DOCUMENTS = [
    """
    Artificial Intelligence (AI) is the simulation of human intelligence in machines that are programmed 
    to think and learn. AI systems can perform tasks that typically require human intelligence, such as 
    visual perception, speech recognition, decision-making, and language translation. Machine learning, 
    a subset of AI, enables systems to automatically learn and improve from experience without being 
    explicitly programmed for every scenario.
    """,
    
    """
    Large Language Models (LLMs) are AI systems trained on vast amounts of text data to understand and 
    generate human-like text. These models, such as GPT-4, use transformer architecture to process and 
    generate text by predicting the next word in a sequence. LLMs have revolutionized natural language 
    processing and enabled applications like chatbots, content generation, and code assistance.
    """,
    
    """
    Retrieval-Augmented Generation (RAG) is a technique that combines the power of large language models 
    with external knowledge retrieval. Instead of relying solely on the model's training data, RAG 
    systems retrieve relevant information from external databases or documents and use this context to 
    generate more accurate and up-to-date responses. This approach is particularly useful for 
    domain-specific applications and reducing hallucinations.
    """,
    
    """
    Vector databases are specialized storage systems designed to store and query high-dimensional vectors 
    efficiently. In AI applications, these vectors often represent embeddings of text, images, or other 
    data types. Vector databases enable semantic search by finding vectors that are similar in the 
    high-dimensional space, making them essential for RAG systems, recommendation engines, and similarity 
    search applications.
    """,
    
    """
    Streaming in AI applications refers to the real-time delivery of responses as they are generated, 
    rather than waiting for the complete response before displaying it to users. This approach 
    significantly improves user experience by reducing perceived latency and providing immediate feedback. 
    Streaming is particularly important for conversational AI and long-form content generation where 
    users benefit from seeing partial results.
    """,
    
    """
    LangGraph is a framework for building stateful, multi-actor applications with language models. 
    It extends LangChain with the ability to create complex workflows that can maintain state across 
    multiple steps and handle conditional logic. LangGraph is particularly useful for building agents 
    that need to make decisions, call tools, and maintain conversation context over multiple interactions.
    """,
    
    """
    Performance optimization in AI systems involves various techniques to reduce latency and improve 
    throughput. Key strategies include model quantization, caching frequently used embeddings, 
    parallel processing, and efficient memory management. For streaming applications, optimizing 
    time-to-first-token and maintaining consistent token generation rates are critical metrics.
    """,
    
    """
    The OpenAI API provides access to powerful language models through RESTful endpoints. It supports 
    various modes including streaming responses, function calling, and fine-tuning. The API is designed 
    for scalability and includes features like rate limiting, usage monitoring, and different model 
    options optimized for various use cases from chat completion to code generation.
    """,
    
    """
    Embeddings are numerical representations of text that capture semantic meaning in high-dimensional 
    vector space. Words or phrases with similar meanings have similar embeddings, enabling machines to 
    understand context and relationships. Modern embedding models like OpenAI's text-embedding-3-small 
    are trained on diverse datasets to capture nuanced meanings across different domains and languages.
    """,
    
    """
    Benchmark testing in AI systems involves systematic measurement of performance across different 
    implementations or configurations. Key metrics include latency percentiles (p50, p90, p99), 
    throughput, accuracy, and resource utilization. Proper benchmarking requires consistent test 
    conditions, sufficient sample sizes, and statistical analysis to draw meaningful conclusions 
    about system performance.
    """
]

def get_knowledge_base() -> list[str]:
    """Return the sample documents for RAG testing."""
    return SAMPLE_DOCUMENTS

def get_test_queries() -> list[str]:
    """Return test queries for benchmark evaluation."""
    return [
        "What is artificial intelligence and how does it work?",
        "Explain how large language models generate text",
        "What are the benefits of using RAG systems?",
        "How do vector databases enable semantic search?",
        "Why is streaming important for AI applications?",
        "What makes LangGraph different from other frameworks?",
        "What techniques are used for AI performance optimization?",
        "How does the OpenAI API handle streaming responses?",
        "What are embeddings and how are they used?",
        "What metrics are important for AI benchmark testing?"
    ]