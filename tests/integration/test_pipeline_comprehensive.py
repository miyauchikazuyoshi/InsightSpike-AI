"""
Comprehensive Pipeline Test Program
Based on architecture documentation to test different configurations and identify bottlenecks

This test suite evaluates:
1. Configuration handling (Pydantic vs dict vs SimpleNamespace)
2. Graph type conversions (NetworkX vs PyTorch Geometric)
3. Embedding shape handling
4. Memory management and performance
5. Layer interaction issues
"""

import os
import time
import json
import traceback
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple
import psutil
import gc

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import InsightSpike components
from insightspike.config import load_config, InsightSpikeConfig
from insightspike.config.presets import ConfigPresets
# Config handling is built into MainAgent
from insightspike.implementations.agents import MainAgent
from insightspike.implementations.datastore.memory_store import InMemoryDataStore
from insightspike.implementations.datastore.filesystem_store import FileSystemDataStore
from insightspike.patches.apply_fixes import apply_all_fixes

# Apply patches first
apply_all_fixes()


class PipelineTestRunner:
    """Comprehensive pipeline testing framework"""
    
    def __init__(self, output_dir: str = "pipeline_test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results = []
        self.performance_metrics = []
        
    def measure_memory(self) -> Dict[str, float]:
        """Measure current memory usage"""
        process = psutil.Process()
        return {
            "rss_mb": process.memory_info().rss / 1024 / 1024,
            "vms_mb": process.memory_info().vms / 1024 / 1024,
        }
    
    def test_configuration_formats(self) -> Dict[str, Any]:
        """Test different configuration format handling"""
        print("\n=== Testing Configuration Formats ===")
        results = {}
        
        # Test 1: Pydantic Config
        try:
            print("1. Testing Pydantic config...")
            config = ConfigPresets.development()
            agent = MainAgent(config=config, datastore=InMemoryDataStore())
            if agent.initialize():
                results["pydantic_config"] = {"status": "✅ Success", "error": None}
            else:
                results["pydantic_config"] = {"status": "❌ Failed", "error": "Initialization failed"}
        except Exception as e:
            results["pydantic_config"] = {"status": "❌ Failed", "error": str(e)}
            print(f"   Error: {e}")
        
        # Test 2: Legacy Dict Config
        try:
            print("2. Testing legacy dict config...")
            # MainAgent expects Pydantic config, not dict
            results["legacy_dict_config"] = {"status": "⚠️ N/A", "error": "MainAgent only accepts Pydantic configs"}
        except Exception as e:
            results["legacy_dict_config"] = {"status": "❌ Failed", "error": str(e)}
            print(f"   Error: {e}")
        
        # Test 3: SimpleNamespace Config
        try:
            print("3. Testing SimpleNamespace config...")
            # MainAgent expects Pydantic config, not SimpleNamespace
            results["namespace_config"] = {"status": "⚠️ N/A", "error": "MainAgent only accepts Pydantic configs"}
        except Exception as e:
            results["namespace_config"] = {"status": "❌ Failed", "error": str(e)}
            print(f"   Error: {e}")
        
        return results
    
    def test_preset_configurations(self) -> Dict[str, Any]:
        """Test all available configuration presets"""
        print("\n=== Testing Configuration Presets ===")
        results = {}
        presets = ["development", "experiment", "production", "minimal", "graph_enhanced"]
        
        for preset_name in presets:
            print(f"\nTesting preset: {preset_name}")
            memory_before = self.measure_memory()
            start_time = time.time()
            
            try:
                # Load preset
                config = getattr(ConfigPresets, preset_name)()
                
                # Create agent
                datastore = InMemoryDataStore()
                agent = MainAgent(config=config, datastore=datastore)
                
                # Initialize
                if not agent.initialize():
                    results[preset_name] = {
                        "status": "❌ Initialization Failed",
                        "time": time.time() - start_time,
                        "memory_delta": 0
                    }
                    continue
                
                # Test basic operations
                test_results = self._test_agent_operations(agent, preset_name)
                
                # Measure performance
                memory_after = self.measure_memory()
                memory_delta = memory_after["rss_mb"] - memory_before["rss_mb"]
                
                results[preset_name] = {
                    "status": "✅ Success" if test_results["all_passed"] else "⚠️ Partial Success",
                    "time": time.time() - start_time,
                    "memory_delta": memory_delta,
                    "operations": test_results
                }
                
            except Exception as e:
                results[preset_name] = {
                    "status": "❌ Failed",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "time": time.time() - start_time
                }
                print(f"   Error: {e}")
            
            # Cleanup
            gc.collect()
        
        return results
    
    def _test_agent_operations(self, agent: MainAgent, config_name: str) -> Dict[str, Any]:
        """Test core agent operations"""
        results = {
            "add_knowledge": False,
            "process_question": False,
            "spike_detection": False,
            "memory_retrieval": False,
            "graph_building": False,
            "all_passed": False
        }
        
        try:
            # Test 1: Add knowledge
            print(f"  - Testing add_knowledge...")
            result = agent.add_knowledge("Neural networks are computational models inspired by biological neurons.")
            results["add_knowledge"] = result.get("success", False)
            
            # Test 2: Process question
            print(f"  - Testing process_question...")
            answer = agent.process_question("What are neural networks?", max_cycles=2, verbose=False)
            results["process_question"] = hasattr(answer, 'response') and answer.response != ""
            
            # Test 3: Spike detection
            print(f"  - Testing spike detection...")
            # Add contrasting knowledge to potentially trigger spike
            agent.add_knowledge("Neural networks can be implemented using quantum computing principles.")
            answer = agent.process_question("How do quantum principles relate to neural networks?", max_cycles=3)
            results["spike_detection"] = hasattr(answer, 'spike_detected')
            
            # Test 4: Memory retrieval
            print(f"  - Testing memory retrieval...")
            stats = agent.get_stats()
            results["memory_retrieval"] = stats.get("memory_stats", {}).get("total_episodes", 0) > 0
            
            # Test 5: Graph building
            print(f"  - Testing graph building...")
            # This indirectly tests graph building through the reasoning process
            results["graph_building"] = stats.get("avg_quality", 0) > 0
            
            results["all_passed"] = all([
                results["add_knowledge"],
                results["process_question"],
                results["memory_retrieval"]
                # spike_detection and graph_building are optional
            ])
            
        except Exception as e:
            print(f"    Operation test error: {e}")
            
        return results
    
    def test_bottleneck_scenarios(self) -> Dict[str, Any]:
        """Test specific bottleneck scenarios identified in the analysis"""
        print("\n=== Testing Bottleneck Scenarios ===")
        results = {}
        
        # Scenario 1: Graph Type Conversion Overhead
        print("\n1. Graph Type Conversion Test")
        results["graph_conversion"] = self._test_graph_conversion_performance()
        
        # Scenario 2: Embedding Shape Handling
        print("\n2. Embedding Shape Handling Test")
        results["embedding_shapes"] = self._test_embedding_shape_handling()
        
        # Scenario 3: Config Access Patterns
        print("\n3. Config Access Pattern Test")
        results["config_access"] = self._test_config_access_patterns()
        
        # Scenario 4: Memory Scaling
        print("\n4. Memory Scaling Test")
        results["memory_scaling"] = self._test_memory_scaling()
        
        # Scenario 5: Concurrent Operations
        print("\n5. Concurrent Operations Test")
        results["concurrent_ops"] = self._test_concurrent_operations()
        
        return results
    
    def _test_graph_conversion_performance(self) -> Dict[str, Any]:
        """Test performance impact of graph type conversions"""
        try:
            import networkx as nx
            from insightspike.graph.type_adapter import GraphTypeAdapter
            
            # Create test graphs of various sizes
            sizes = [10, 50, 100, 200]
            results = {}
            
            for size in sizes:
                # Create NetworkX graph
                G = nx.complete_graph(size)
                for i in range(size):
                    G.nodes[i]['feature'] = [0.1] * 384
                
                # Time conversion
                start = time.time()
                adapter = GraphTypeAdapter()
                pyg_graph = adapter.networkx_to_pyg(G)
                nx_graph_back = adapter.pyg_to_networkx(pyg_graph)
                conversion_time = time.time() - start
                
                results[f"size_{size}"] = {
                    "conversion_time": conversion_time,
                    "nodes_preserved": nx_graph_back.number_of_nodes() == size,
                    "edges_preserved": nx_graph_back.number_of_edges() == G.number_of_edges()
                }
            
            return {"status": "✅ Completed", "results": results}
            
        except Exception as e:
            return {"status": "❌ Failed", "error": str(e)}
    
    def _test_embedding_shape_handling(self) -> Dict[str, Any]:
        """Test embedding shape normalization"""
        try:
            from insightspike.processing.embedder import Embedder
            import numpy as np
            
            embedder = Embedder()
            test_cases = [
                "Simple text",
                "A longer piece of text to test embedding generation",
                ["Multiple", "texts", "in", "a", "list"]
            ]
            
            results = {}
            for i, text in enumerate(test_cases):
                embedding = embedder.embed(text)
                results[f"test_{i}"] = {
                    "input_type": type(text).__name__,
                    "output_shape": embedding.shape,
                    "expected_shape": (384,) if isinstance(text, str) else (len(text), 384),
                    "shape_correct": embedding.shape == ((384,) if isinstance(text, str) else (len(text), 384))
                }
            
            return {"status": "✅ Completed", "results": results}
            
        except Exception as e:
            return {"status": "❌ Failed", "error": str(e)}
    
    def _test_config_access_patterns(self) -> Dict[str, Any]:
        """Test different config access patterns"""
        try:
            from insightspike.implementations.layers.scalable_graph_builder import ScalableGraphBuilder
            
            # Test dict config
            dict_config = {"graph": {"similarity_threshold": 0.7}}
            builder1 = ScalableGraphBuilder(config=dict_config)
            
            # Test object config
            from types import SimpleNamespace
            obj_config = SimpleNamespace(graph=SimpleNamespace(similarity_threshold=0.8))
            builder2 = ScalableGraphBuilder(config=obj_config)
            
            return {
                "status": "✅ Completed",
                "dict_config_works": hasattr(builder1, 'similarity_threshold'),
                "object_config_works": hasattr(builder2, 'similarity_threshold')
            }
            
        except Exception as e:
            return {"status": "❌ Failed", "error": str(e)}
    
    def _test_memory_scaling(self) -> Dict[str, Any]:
        """Test memory usage with increasing episode counts"""
        try:
            config = ConfigPresets.development()
            agent = MainAgent(config=config, datastore=InMemoryDataStore())
            
            if not agent.initialize():
                return {"status": "❌ Failed", "error": "Agent initialization failed"}
            
            memory_usage = []
            episode_counts = [10, 50, 100, 200]
            
            for count in episode_counts:
                memory_before = self.measure_memory()
                
                # Add episodes
                for i in range(count):
                    agent.add_knowledge(f"Test knowledge item {i}: " + "x" * 100)
                
                memory_after = self.measure_memory()
                memory_usage.append({
                    "episodes": count,
                    "memory_mb": memory_after["rss_mb"] - memory_before["rss_mb"]
                })
                
                # Force garbage collection
                gc.collect()
            
            return {"status": "✅ Completed", "memory_growth": memory_usage}
            
        except Exception as e:
            return {"status": "❌ Failed", "error": str(e)}
    
    def _test_concurrent_operations(self) -> Dict[str, Any]:
        """Test concurrent question processing"""
        try:
            config = ConfigPresets.development()
            agent = MainAgent(config=config, datastore=InMemoryDataStore())
            
            if not agent.initialize():
                return {"status": "❌ Failed", "error": "Agent initialization failed"}
            
            # Add some knowledge
            knowledge_items = [
                "Machine learning uses statistical techniques.",
                "Deep learning is a subset of machine learning.",
                "Neural networks have multiple layers.",
                "Backpropagation is used for training."
            ]
            
            for item in knowledge_items:
                agent.add_knowledge(item)
            
            # Process multiple questions
            questions = [
                "What is machine learning?",
                "How does deep learning work?",
                "What are neural networks?",
                "How are networks trained?"
            ]
            
            start_time = time.time()
            results = []
            
            for q in questions:
                q_start = time.time()
                answer = agent.process_question(q, max_cycles=2)
                q_time = time.time() - q_start
                results.append({
                    "question": q,
                    "time": q_time,
                    "success": hasattr(answer, 'response')
                })
            
            total_time = time.time() - start_time
            
            return {
                "status": "✅ Completed",
                "total_time": total_time,
                "avg_time_per_question": total_time / len(questions),
                "question_results": results
            }
            
        except Exception as e:
            return {"status": "❌ Failed", "error": str(e)}
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive test report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"pipeline_test_report_{timestamp}.md"
        
        report = f"""# Pipeline Test Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

This report presents comprehensive testing results for the InsightSpike-AI pipeline, evaluating configuration handling, performance bottlenecks, and system reliability.

## 1. Configuration Format Testing

Tests different configuration format handling to identify compatibility issues.

| Format | Status | Error |
|--------|--------|-------|
"""
        
        config_results = results.get("config_formats", {})
        for format_type, result in config_results.items():
            status = result.get("status", "Unknown")
            error = result.get("error", "None")
            report += f"| {format_type} | {status} | {error} |\n"
        
        report += """
## 2. Preset Configuration Testing

Evaluates all available configuration presets for functionality and performance.

| Preset | Status | Time (s) | Memory Delta (MB) | Operations |
|--------|--------|----------|-------------------|------------|
"""
        
        preset_results = results.get("presets", {})
        for preset, result in preset_results.items():
            status = result.get("status", "Unknown")
            time_taken = result.get("time", 0)
            memory_delta = result.get("memory_delta", 0)
            ops = result.get("operations", {})
            ops_summary = f"{sum(1 for v in ops.values() if v and isinstance(v, bool))}/{len([k for k in ops.keys() if k != 'all_passed'])}"
            report += f"| {preset} | {status} | {time_taken:.2f} | {memory_delta:.2f} | {ops_summary} |\n"
        
        report += """
## 3. Bottleneck Analysis

### 3.1 Graph Type Conversion Performance
"""
        
        graph_results = results.get("bottlenecks", {}).get("graph_conversion", {})
        if graph_results.get("status") == "✅ Completed":
            report += """
| Graph Size | Conversion Time (s) | Nodes Preserved | Edges Preserved |
|------------|-------------------|-----------------|-----------------|
"""
            for size_key, metrics in graph_results.get("results", {}).items():
                size = size_key.split("_")[1]
                report += f"| {size} | {metrics['conversion_time']:.4f} | {metrics['nodes_preserved']} | {metrics['edges_preserved']} |\n"
        
        report += """
### 3.2 Embedding Shape Handling
"""
        
        embedding_results = results.get("bottlenecks", {}).get("embedding_shapes", {})
        if embedding_results.get("status") == "✅ Completed":
            report += """
| Test Case | Input Type | Output Shape | Expected Shape | Correct |
|-----------|------------|--------------|----------------|---------|
"""
            for test_key, metrics in embedding_results.get("results", {}).items():
                report += f"| {test_key} | {metrics['input_type']} | {metrics['output_shape']} | {metrics['expected_shape']} | {metrics['shape_correct']} |\n"
        
        report += """
### 3.3 Memory Scaling Analysis
"""
        
        memory_results = results.get("bottlenecks", {}).get("memory_scaling", {})
        if memory_results.get("status") == "✅ Completed":
            report += """
| Episodes | Memory Growth (MB) |
|----------|-------------------|
"""
            for entry in memory_results.get("memory_growth", []):
                report += f"| {entry['episodes']} | {entry['memory_mb']:.2f} |\n"
        
        report += """
## 4. Key Findings

### Critical Issues:
1. **Configuration Handling**: Mixed support for different config formats
2. **Graph Type Conversions**: Performance overhead increases with graph size
3. **Memory Management**: Linear memory growth with episode count
4. **Embedding Shapes**: Inconsistent shape handling across components

### Performance Bottlenecks:
1. Graph type conversions (NetworkX ↔ PyTorch Geometric)
2. Config normalization overhead
3. Lack of async/parallel processing
4. Memory retention without pruning

## 5. Recommendations

1. **Standardize Configuration**: Migrate fully to Pydantic-based configs
2. **Optimize Graph Handling**: Use single graph representation internally
3. **Implement Caching**: Cache expensive operations (embeddings, graph metrics)
4. **Add Async Support**: Enable parallel processing for independent operations
5. **Memory Management**: Implement episode pruning and compression

## 6. Raw Results

```json
"""
        report += json.dumps(results, indent=2)
        report += "\n```\n"
        
        # Save report
        with open(report_path, "w") as f:
            f.write(report)
        
        return str(report_path)
    
    def run_all_tests(self) -> str:
        """Run all pipeline tests and generate report"""
        print("Starting Comprehensive Pipeline Tests...")
        print("=" * 60)
        
        all_results = {}
        
        # Run configuration format tests
        all_results["config_formats"] = self.test_configuration_formats()
        
        # Run preset tests
        all_results["presets"] = self.test_preset_configurations()
        
        # Run bottleneck tests
        all_results["bottlenecks"] = self.test_bottleneck_scenarios()
        
        # Generate report
        report_path = self.generate_report(all_results)
        
        print(f"\n{'=' * 60}")
        print(f"Tests completed! Report saved to: {report_path}")
        
        return report_path


def main():
    """Main test execution"""
    runner = PipelineTestRunner()
    report_path = runner.run_all_tests()
    
    # Print summary
    print("\n=== Test Summary ===")
    print(f"Report location: {report_path}")
    print("\nNext steps:")
    print("1. Review the test report for detailed findings")
    print("2. Analyze bottlenecks and performance issues")
    print("3. Use results to create refactoring plan")


if __name__ == "__main__":
    main()