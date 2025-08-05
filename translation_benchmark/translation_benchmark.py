import asyncio
import time
import numpy as np
import matplotlib.pyplot as plt
from translation_openai import translate_text_streaming as openai_translate
from translation_langchain import translate_text_streaming as langchain_translate
from translation_langgraph import translate_text_streaming as langgraph_translate

async def benchmark_translation():
    """Fair benchmark: Identical functionality, measuring pure architectural overhead"""
    
    test_texts = [
        "Hello, how are you today?",
        
        "The weather is beautiful and sunny.",
        
        "I love programming and technology.",
        
        "Modern software development practices emphasize collaboration, continuous integration, and agile methodologies that enable teams to deliver high-quality products more efficiently. DevOps culture has bridged the gap between development and operations, fostering better communication and reducing the time between code creation and deployment to production environments.",
        
        "Quantum computing promises to solve complex computational problems that are currently intractable for classical computers, potentially revolutionizing fields such as cryptography, drug discovery, and financial modeling. As quantum hardware continues to improve and become more accessible, researchers are exploring new algorithms and applications that could unlock unprecedented computational capabilities.",
        
        "The rise of remote work has fundamentally altered traditional workplace dynamics, forcing organizations to reimagine how they collaborate, communicate, and maintain company culture across distributed teams. Digital collaboration tools, cloud-based infrastructure, and flexible work arrangements have become essential components of modern business operations in the post-pandemic world.",
        
        "Cybersecurity has evolved from a niche technical concern to a critical business imperative as organizations increasingly rely on digital infrastructure and face sophisticated threats from malicious actors. Companies must now implement comprehensive security strategies that encompass not only technical defenses but also employee training, incident response planning, and regulatory compliance measures.",
        
        "The emergence of blockchain technology has introduced new paradigms for decentralized systems, digital currencies, and smart contracts that operate without traditional intermediaries. While still evolving, blockchain applications are being explored across various industries including finance, supply chain management, healthcare, and digital identity verification.",
        
        "Data privacy and ethical AI have become paramount concerns as artificial intelligence systems process increasingly sensitive personal information and make decisions that significantly impact individuals' lives. Regulatory frameworks like GDPR and emerging AI governance standards are shaping how organizations collect, process, and protect user data while ensuring algorithmic fairness and transparency.",
        
        "The Internet of Things (IoT) continues to expand the boundaries of connected devices, creating smart environments where everyday objects can communicate, collect data, and respond to changing conditions autonomously. From smart homes and cities to industrial automation and healthcare monitoring, IoT technologies are creating new possibilities for efficiency, convenience, and data-driven insights."
    ]
    
    openai_times = []
    langchain_times = []
    langgraph_times = []
    openai_ttft = []
    langchain_ttft = []
    langgraph_ttft = []
    
    print("=== Fair Translation Performance Benchmark ===\n")
    print("OpenAI Raw: Direct AsyncOpenAI client (baseline)")
    print("LangChain: ChatOpenAI wrapper overhead") 
    print("LangGraph: Workflow architecture overhead")
    print("\nIDENTICAL: Model, temperature, prompt, streaming, processing")
    print("MEASURING: Pure architectural overhead differences")
    print("\nRunning 1,500 tests total (500 per implementation - 50 runs per message)...\n")
    
    # Create 50 runs of each test text for statistical significance
    all_test_texts = test_texts * 50
    
    # Run tests concurrently in batches for speed
    print("Testing OpenAI Direct API (batch processing - 500 tests)...")
    openai_tasks = [openai_translate(text) for text in all_test_texts]
    openai_results = await asyncio.gather(*openai_tasks)
    
    for i, result in enumerate(openai_results, 1):
        if result:
            openai_times.append(result['total_time'])
            if result['time_to_first_token']:
                openai_ttft.append(result['time_to_first_token'])
        if i % 50 == 0:
            print(f"OpenAI Raw: Completed {i}/500 tests")
    
    print("\nTesting LangChain Wrapper (batch processing - 500 tests)...")
    langchain_tasks = [langchain_translate(text) for text in all_test_texts]
    langchain_results = await asyncio.gather(*langchain_tasks)
    
    for i, result in enumerate(langchain_results, 1):
        if result:
            langchain_times.append(result['total_time'])
            if result['time_to_first_token']:
                langchain_ttft.append(result['time_to_first_token'])
        if i % 50 == 0:
            print(f"LangChain: Completed {i}/500 tests")
    
    print("\nTesting LangGraph Workflow (batch processing - 500 tests)...")
    langgraph_tasks = [langgraph_translate(text) for text in all_test_texts]
    langgraph_results = await asyncio.gather(*langgraph_tasks)
    
    for i, result in enumerate(langgraph_results, 1):
        if result:
            langgraph_times.append(result['total_time'])
            if result['time_to_first_token']:
                langgraph_ttft.append(result['time_to_first_token'])
        if i % 50 == 0:
            print(f"LangGraph: Completed {i}/500 tests")
    
    # Calculate comprehensive statistics
    print("\n=== COMPREHENSIVE BENCHMARK RESULTS ===")
    
    # Calculate all percentiles and statistics for all three implementations
    openai_stats = {
        'mean': np.mean(openai_times),
        'median': np.median(openai_times),
        'p90': np.percentile(openai_times, 90),
        'p95': np.percentile(openai_times, 95),
        'p99': np.percentile(openai_times, 99),
        'min': np.min(openai_times),
        'max': np.max(openai_times),
        'std': np.std(openai_times),
        'ttft_mean': np.mean(openai_ttft) if openai_ttft else 0
    }
    
    langchain_stats = {
        'mean': np.mean(langchain_times),
        'median': np.median(langchain_times),
        'p90': np.percentile(langchain_times, 90),
        'p95': np.percentile(langchain_times, 95),
        'p99': np.percentile(langchain_times, 99),
        'min': np.min(langchain_times),
        'max': np.max(langchain_times),
        'std': np.std(langchain_times),
        'ttft_mean': np.mean(langchain_ttft) if langchain_ttft else 0
    }
    
    langgraph_stats = {
        'mean': np.mean(langgraph_times),
        'median': np.median(langgraph_times),
        'p90': np.percentile(langgraph_times, 90),
        'p95': np.percentile(langgraph_times, 95),
        'p99': np.percentile(langgraph_times, 99),
        'min': np.min(langgraph_times),
        'max': np.max(langgraph_times),
        'std': np.std(langgraph_times),
        'ttft_mean': np.mean(langgraph_ttft) if langgraph_ttft else 0
    }
    
    # Print detailed comparison table
    print("\nSTATISTICAL SUMMARY\n")
    print(f"{'Metric':<20} {'OpenAI Raw':<12} {'LangChain':<12} {'LangGraph':<12} {'Winner':<15}")
    print("-" * 85)
    
    metrics = [
        ('Mean Latency', 'mean'),
        ('Median Latency', 'median'),
        ('P90 Latency', 'p90'),
        ('P95 Latency', 'p95'),
        ('P99 Latency', 'p99'),
        ('Min Latency', 'min'),
        ('Max Latency', 'max'),
        ('Std Deviation', 'std'),
        ('Mean TTFT', 'ttft_mean')
    ]
    
    for metric_name, metric_key in metrics:
        oa_val = openai_stats[metric_key]
        lc_val = langchain_stats[metric_key]
        lg_val = langgraph_stats[metric_key]
        
        # Determine winner (lower is better for all these metrics)
        values = [oa_val, lc_val, lg_val]
        min_val = min(values)
        
        if oa_val == min_val:
            winner = "**OpenAI Raw**"
        elif lc_val == min_val:
            winner = "**LangChain**"
        else:
            winner = "**LangGraph**"
            
        print(f"{metric_name:<20} {oa_val:<12.3f} {lc_val:<12.3f} {lg_val:<12.3f} {winner:<15}")
    
    # Performance insights with realistic context
    print("\nKEY PERFORMANCE INSIGHTS")
    print(f"• OpenAI Raw (Simple): {openai_stats['mean']:.3f}s - Baseline direct API")
    print(f"• LangChain (Processing): {langchain_stats['mean']:.3f}s ({((langchain_stats['mean'] - openai_stats['mean']) / openai_stats['mean'] * 100):+.1f}% vs Raw) - Professional features cost")
    print(f"• LangGraph (Workflow): {langgraph_stats['mean']:.3f}s ({((langgraph_stats['mean'] - openai_stats['mean']) / openai_stats['mean'] * 100):+.1f}% vs Raw) - Workflow architecture cost")
    print(f"• Total Tests: 1,500 ({len(openai_times)} per implementation)")
    print(f"• Consistency (std): Raw={openai_stats['std']:.3f}s, LangChain={langchain_stats['std']:.3f}s, LangGraph={langgraph_stats['std']:.3f}s")
    
    print("\nINTERPRETATION:")
    print("• Raw OpenAI: Baseline performance - direct API calls")
    print("• LangChain: Shows cost of wrapper abstractions and ChatOpenAI overhead") 
    print("• LangGraph: Shows cost of workflow architecture even for simple tasks")
    
    # Create visualization
    # Create comprehensive visualization for three implementations
    plt.figure(figsize=(18, 12))
    
    # 1. Box plots for total time
    plt.subplot(2, 3, 1)
    plt.boxplot([openai_times, langchain_times, langgraph_times], 
                tick_labels=['OpenAI Raw', 'LangChain', 'LangGraph'])
    plt.title('Total Response Time Distribution')
    plt.ylabel('Time (seconds)')
    plt.grid(True, alpha=0.3)
    
    # 2. Box plots for TTFT
    plt.subplot(2, 3, 2)
    plt.boxplot([openai_ttft, langchain_ttft, langgraph_ttft], 
                tick_labels=['OpenAI Raw', 'LangChain', 'LangGraph'])
    plt.title('Time to First Token Distribution')
    plt.ylabel('Time (seconds)')
    plt.grid(True, alpha=0.3)
    
    # 3. Percentile comparison
    plt.subplot(2, 3, 3)
    percentiles = [50, 90, 95, 99]
    openai_percs = [np.percentile(openai_times, p) for p in percentiles]
    langchain_percs = [np.percentile(langchain_times, p) for p in percentiles]
    langgraph_percs = [np.percentile(langgraph_times, p) for p in percentiles]
    
    x = np.arange(len(percentiles))
    width = 0.25
    plt.bar(x - width, openai_percs, width, label='OpenAI Raw', alpha=0.8)
    plt.bar(x, langchain_percs, width, label='LangChain', alpha=0.8)
    plt.bar(x + width, langgraph_percs, width, label='LangGraph', alpha=0.8)
    plt.xlabel('Percentiles')
    plt.ylabel('Time (seconds)')
    plt.title('Percentile Comparison')
    plt.xticks(x, [f'P{p}' for p in percentiles])
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Running average
    plt.subplot(2, 3, 4)
    window = 50
    openai_rolling = np.convolve(openai_times, np.ones(window)/window, mode='valid')
    langchain_rolling = np.convolve(langchain_times, np.ones(window)/window, mode='valid')
    langgraph_rolling = np.convolve(langgraph_times, np.ones(window)/window, mode='valid')
    
    plt.plot(range(window, len(openai_times) + 1), openai_rolling, label='OpenAI Raw', alpha=0.8)
    plt.plot(range(window, len(langchain_times) + 1), langchain_rolling, label='LangChain', alpha=0.8)
    plt.plot(range(window, len(langgraph_times) + 1), langgraph_rolling, label='LangGraph', alpha=0.8)
    plt.title(f'Running Average (window={window})')
    plt.xlabel('Test Number')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Histogram comparison
    plt.subplot(2, 3, 5)
    all_times = openai_times + langchain_times + langgraph_times
    bins = np.linspace(min(all_times), max(all_times), 30)
    plt.hist(openai_times, bins=bins, alpha=0.6, label='OpenAI Raw', density=True)
    plt.hist(langchain_times, bins=bins, alpha=0.6, label='LangChain', density=True)
    plt.hist(langgraph_times, bins=bins, alpha=0.6, label='LangGraph', density=True)
    plt.title('Response Time Distribution (Density)')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Mean comparison bar chart
    plt.subplot(2, 3, 6)
    implementations = ['OpenAI Raw', 'LangChain', 'LangGraph']
    means = [openai_stats['mean'], langchain_stats['mean'], langgraph_stats['mean']]
    colors = ['blue', 'orange', 'green']
    bars = plt.bar(implementations, means, color=colors, alpha=0.7)
    plt.title('Mean Response Time Comparison')
    plt.ylabel('Time (seconds)')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{mean_val:.3f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('translation_performance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nPerformance chart saved as 'translation_performance_comparison.png'")
    
    return {
        'openai_times': openai_times,
        'langchain_times': langchain_times,
        'langgraph_times': langgraph_times,
        'openai_ttft': openai_ttft,
        'langchain_ttft': langchain_ttft,
        'langgraph_ttft': langgraph_ttft,
        'openai_stats': openai_stats,
        'langchain_stats': langchain_stats,
        'langgraph_stats': langgraph_stats
    }

if __name__ == "__main__":
    asyncio.run(benchmark_translation())