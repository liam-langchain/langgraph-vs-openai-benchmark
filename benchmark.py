"""
Benchmark runner for comparing LangGraph vs OpenAI API streaming RAG performance.
Runs 500 iterations of each app and calculates p99 latency.
"""

import asyncio
import time
import json
import statistics
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from langgraph_app import LangGraphRAGApp
from openai_app import OpenAIRAGApp
from knowledge_base import get_test_queries


class BenchmarkRunner:
    def __init__(self):
        self.langgraph_app = LangGraphRAGApp()
        self.openai_app = OpenAIRAGApp()
        self.results = {
            "langgraph": {"latencies": [], "responses": []},
            "openai": {"latencies": [], "responses": []}
        }
        
    async def initialize(self):
        """Initialize both applications."""
        print("Initializing LangGraph app...")
        await self.langgraph_app.initialize()
        
        print("Initializing OpenAI app...")
        await self.openai_app.initialize()
        
        print("Both apps initialized successfully!")
    
    async def run_single_test(self, app, query: str) -> tuple[str, float]:
        """Run a single test with the given app and query."""
        try:
            response, latency = await app.get_response(query)
            return response, latency
        except Exception as e:
            print(f"Error in test: {e}")
            return "", float('inf')
    
    async def run_benchmark_batch(self, app_name: str, app, queries: List[str], runs_per_query: int = 50):
        """Run benchmark for a single app."""
        print(f"\nRunning {app_name} benchmark...")
        print(f"Testing {len(queries)} queries with {runs_per_query} runs each ({len(queries) * runs_per_query} total)")
        
        total_runs = len(queries) * runs_per_query
        completed_runs = 0
        
        for query_idx, query in enumerate(queries):
            print(f"\nQuery {query_idx + 1}/{len(queries)}: {query[:50]}...")
            
            for run_idx in range(runs_per_query):
                response, latency = await self.run_single_test(app, query)
                
                self.results[app_name]["latencies"].append(latency)
                self.results[app_name]["responses"].append({
                    "query": query,
                    "response": response[:100] + "..." if len(response) > 100 else response,
                    "latency": latency,
                    "run": run_idx + 1
                })
                
                completed_runs += 1
                if completed_runs % 50 == 0:
                    avg_latency = statistics.mean(self.results[app_name]["latencies"][-50:])
                    print(f"  Progress: {completed_runs}/{total_runs} runs completed. Avg latency (last 50): {avg_latency:.3f}s")
        
        # Calculate statistics
        latencies = self.results[app_name]["latencies"]
        stats = {
            "total_runs": len(latencies),
            "mean_latency": statistics.mean(latencies),
            "median_latency": statistics.median(latencies),
            "p90_latency": np.percentile(latencies, 90),
            "p95_latency": np.percentile(latencies, 95),
            "p99_latency": np.percentile(latencies, 99),
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "std_dev": statistics.stdev(latencies) if len(latencies) > 1 else 0
        }
        
        self.results[app_name]["stats"] = stats
        
        print(f"\n{app_name.upper()} RESULTS:")
        print(f"  Total runs: {stats['total_runs']}")
        print(f"  Mean latency: {stats['mean_latency']:.3f}s")
        print(f"  Median latency: {stats['median_latency']:.3f}s")
        print(f"  P90 latency: {stats['p90_latency']:.3f}s")
        print(f"  P95 latency: {stats['p95_latency']:.3f}s")
        print(f"  P99 latency: {stats['p99_latency']:.3f}s")
        print(f"  Min latency: {stats['min_latency']:.3f}s")
        print(f"  Max latency: {stats['max_latency']:.3f}s")
        print(f"  Std deviation: {stats['std_dev']:.3f}s")
    
    async def run_full_benchmark(self, runs_per_query: int = 50):
        """Run the complete benchmark for both apps."""
        start_time = time.time()
        queries = get_test_queries()
        
        print(f"Starting benchmark with {len(queries)} queries and {runs_per_query} runs per query")
        print(f"Total runs per app: {len(queries) * runs_per_query}")
        
        # Run LangGraph benchmark
        await self.run_benchmark_batch("langgraph", self.langgraph_app, queries, runs_per_query)
        
        # Run OpenAI benchmark
        await self.run_benchmark_batch("openai", self.openai_app, queries, runs_per_query)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n" + "="*80)
        print("BENCHMARK COMPARISON")
        print("="*80)
        
        lg_stats = self.results["langgraph"]["stats"]
        oa_stats = self.results["openai"]["stats"]
        
        print(f"{'Metric':<20} {'LangGraph':<12} {'OpenAI':<12} {'Difference':<15}")
        print("-" * 60)
        print(f"{'Mean Latency':<20} {lg_stats['mean_latency']:<12.3f} {oa_stats['mean_latency']:<12.3f} {lg_stats['mean_latency'] - oa_stats['mean_latency']:+.3f}s")
        print(f"{'Median Latency':<20} {lg_stats['median_latency']:<12.3f} {oa_stats['median_latency']:<12.3f} {lg_stats['median_latency'] - oa_stats['median_latency']:+.3f}s")
        print(f"{'P90 Latency':<20} {lg_stats['p90_latency']:<12.3f} {oa_stats['p90_latency']:<12.3f} {lg_stats['p90_latency'] - oa_stats['p90_latency']:+.3f}s")
        print(f"{'P95 Latency':<20} {lg_stats['p95_latency']:<12.3f} {oa_stats['p95_latency']:<12.3f} {oa_stats['p95_latency'] - oa_stats['p95_latency']:+.3f}s")
        print(f"{'P99 Latency':<20} {lg_stats['p99_latency']:<12.3f} {oa_stats['p99_latency']:<12.3f} {lg_stats['p99_latency'] - oa_stats['p99_latency']:+.3f}s")
        
        print(f"\nTotal benchmark time: {total_time:.2f} seconds")
        
        # Determine winner
        if lg_stats['p99_latency'] < oa_stats['p99_latency']:
            winner = "LangGraph"
            difference = oa_stats['p99_latency'] - lg_stats['p99_latency']
        else:
            winner = "OpenAI API"
            difference = lg_stats['p99_latency'] - oa_stats['p99_latency']
        
        print(f"\nP99 LATENCY WINNER: {winner}")
        print(f"Performance advantage: {difference:.3f}s ({(difference/min(lg_stats['p99_latency'], oa_stats['p99_latency'])*100):.1f}% faster)")
    
    def save_results(self, filename: str = "benchmark_results.json"):
        """Save benchmark results to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to {filename}")
    
    def create_visualization(self, filename: str = "performance_comparison.png"):
        """Create performance comparison chart."""
        plt.figure(figsize=(12, 8))
        
        # Prepare data
        lg_latencies = self.results["langgraph"]["latencies"]
        oa_latencies = self.results["openai"]["latencies"]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Histogram comparison
        ax1.hist(lg_latencies, bins=50, alpha=0.7, label='LangGraph', color='blue')
        ax1.hist(oa_latencies, bins=50, alpha=0.7, label='OpenAI API', color='red')
        ax1.set_xlabel('Latency (seconds)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Latency Distribution Comparison')
        ax1.legend()
        
        # Box plot comparison
        ax2.boxplot([lg_latencies, oa_latencies], labels=['LangGraph', 'OpenAI API'])
        ax2.set_ylabel('Latency (seconds)')
        ax2.set_title('Latency Box Plot Comparison')
        
        # Percentile comparison
        percentiles = [50, 90, 95, 99]
        lg_percentiles = [np.percentile(lg_latencies, p) for p in percentiles]
        oa_percentiles = [np.percentile(oa_latencies, p) for p in percentiles]
        
        x = np.arange(len(percentiles))
        width = 0.35
        
        ax3.bar(x - width/2, lg_percentiles, width, label='LangGraph', color='blue')
        ax3.bar(x + width/2, oa_percentiles, width, label='OpenAI API', color='red')
        ax3.set_xlabel('Percentile')
        ax3.set_ylabel('Latency (seconds)')
        ax3.set_title('Percentile Latency Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'P{p}' for p in percentiles])
        ax3.legend()
        
        # Running average comparison
        window_size = 50
        lg_running_avg = [statistics.mean(lg_latencies[max(0, i-window_size):i+1]) for i in range(len(lg_latencies))]
        oa_running_avg = [statistics.mean(oa_latencies[max(0, i-window_size):i+1]) for i in range(len(oa_latencies))]
        
        ax4.plot(lg_running_avg, label='LangGraph', color='blue')
        ax4.plot(oa_running_avg, label='OpenAI API', color='red')
        ax4.set_xlabel('Run Number')
        ax4.set_ylabel('Running Average Latency (seconds)')
        ax4.set_title(f'Running Average Latency (Window: {window_size})')
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {filename}")
        plt.show()


async def main():
    """Main benchmark execution."""
    benchmark = BenchmarkRunner()
    
    try:
        await benchmark.initialize()
        await benchmark.run_full_benchmark(runs_per_query=50)  # 500 total runs per app
        benchmark.save_results()
        benchmark.create_visualization()
        
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
    except Exception as e:
        print(f"Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())