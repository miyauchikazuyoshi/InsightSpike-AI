#!/usr/bin/env python3
"""
Regenerate Data Structure with Proper Embeddings
===============================================

Uses the fixed MainAgent with SentenceTransformer embeddings
to create a properly embedded data structure.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'src'))

from insightspike.core.agents.main_agent import MainAgent
from insightspike.algorithms.information_gain import InformationGain, EntropyMethod
from insightspike.algorithms.graph_edit_distance import GraphEditDistance, OptimizationLevel

import numpy as np
import time
import json
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProperEmbeddingDataGenerator:
    """Generate data structure with proper semantic embeddings"""
    
    def __init__(self):
        # Initialize InsightSpike components
        self.agent = MainAgent()
        self.ig_calculator = InformationGain(method=EntropyMethod.SHANNON)
        self.ged_calculator = GraphEditDistance(optimization_level=OptimizationLevel.STANDARD)
        
        # Initialize agent
        if not self.agent.initialize():
            logger.warning("Failed to fully initialize MainAgent, continuing anyway")
        
        self.processing_times = []
        
    def check_data_state(self, label="Current"):
        """Check the status of data files"""
        data_dir = Path("data")
        files_to_check = ["episodes.json", "index.faiss", "graph_pyg.pt"]
        
        logger.info(f"\n{label} data state:")
        for filename in files_to_check:
            filepath = data_dir / filename
            if filepath.exists():
                size = filepath.stat().st_size
                mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
                logger.info(f"  {filename}: {size:,} bytes, modified {mtime}")
                
                # Special handling for episodes.json
                if filename == "episodes.json":
                    try:
                        with open(filepath, 'r') as f:
                            episodes = json.load(f)
                        logger.info(f"    Episodes count: {len(episodes)}")
                    except:
                        pass
            else:
                logger.info(f"  {filename}: NOT FOUND")
    
    def create_diverse_documents(self, n_docs=100):
        """Create diverse documents for comprehensive testing"""
        topics = [
            "machine learning", "deep learning", "neural networks",
            "natural language processing", "computer vision",
            "reinforcement learning", "data science", "algorithms",
            "artificial intelligence", "robotics", "quantum computing",
            "blockchain technology", "cybersecurity", "cloud computing",
            "edge computing", "IoT devices", "5G networks",
            "augmented reality", "virtual reality", "mixed reality"
        ]
        
        templates = [
            "{0} is a field that involves {verb} {object}.",
            "The principles of {0} include {aspect1} and {aspect2}.",
            "Applications of {0} range from {app1} to {app2}.",
            "{0} has revolutionized how we approach {problem}.",
            "Recent advances in {0} have enabled {capability}.",
            "The future of {0} lies in {direction}.",
            "{0} combines {field1} with {field2} to achieve {goal}.",
            "Key challenges in {0} include {challenge1} and {challenge2}.",
            "{0} requires understanding of {concept1} and {concept2}.",
            "Innovation in {0} is driven by {factor1} and {factor2}."
        ]
        
        # Varied content components
        verbs = ["analyzing", "processing", "understanding", "optimizing", "learning from", "transforming"]
        objects = ["data patterns", "complex systems", "information", "algorithms", "models", "structures"]
        aspects = ["mathematical foundations", "computational efficiency", "scalability", "accuracy", "robustness"]
        applications = ["healthcare", "finance", "transportation", "education", "entertainment", "manufacturing"]
        problems = ["complex problems", "real-world challenges", "computational tasks", "optimization problems"]
        capabilities = ["real-time processing", "automated decision-making", "pattern recognition", "predictive analytics"]
        
        documents = []
        for i in range(n_docs):
            topic = topics[i % len(topics)]
            template = templates[i % len(templates)]
            
            # Fill in template with varied content
            doc = template.format(topic)
            doc = doc.replace("{verb}", np.random.choice(verbs))
            doc = doc.replace("{object}", np.random.choice(objects))
            doc = doc.replace("{aspect1}", np.random.choice(aspects))
            doc = doc.replace("{aspect2}", np.random.choice(aspects))
            doc = doc.replace("{app1}", np.random.choice(applications))
            doc = doc.replace("{app2}", np.random.choice(applications))
            doc = doc.replace("{problem}", np.random.choice(problems))
            doc = doc.replace("{capability}", np.random.choice(capabilities))
            doc = doc.replace("{field1}", np.random.choice(topics))
            doc = doc.replace("{field2}", np.random.choice(topics))
            doc = doc.replace("{goal}", "better " + np.random.choice(objects))
            doc = doc.replace("{challenge1}", "computational " + np.random.choice(["complexity", "cost", "requirements"]))
            doc = doc.replace("{challenge2}", "data " + np.random.choice(["availability", "quality", "privacy"]))
            doc = doc.replace("{concept1}", np.random.choice(["statistics", "linear algebra", "calculus", "probability"]))
            doc = doc.replace("{concept2}", np.random.choice(["optimization", "algorithms", "data structures", "complexity theory"]))
            doc = doc.replace("{factor1}", "research " + np.random.choice(["breakthroughs", "investments", "collaborations"]))
            doc = doc.replace("{factor2}", "industry " + np.random.choice(["demands", "adoption", "standards"]))
            doc = doc.replace("{direction}", np.random.choice(["automation", "intelligence augmentation", "democratization", "integration"]))
            
            documents.append(doc)
        
        return documents
    
    def process_documents(self, documents):
        """Process documents through InsightSpike-AI with proper embeddings"""
        logger.info(f"Processing {len(documents)} documents with proper embeddings...")
        
        save_frequency = 20  # Save state every 20 documents
        
        for i, doc in enumerate(documents):
            start_time = time.time()
            
            # Process through agent - this now uses proper embeddings
            result = self.agent.process_question(
                f"Learn and remember this information: {doc}",
                max_cycles=2
            )
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            # Check for insights
            if 'reasoning_quality' in result:
                quality = result['reasoning_quality']
                if quality > 0.7:  # Higher threshold for proper embeddings
                    logger.info(f"High quality insight at document {i}: quality={quality:.3f}")
            
            # Save state periodically
            if (i + 1) % save_frequency == 0:
                logger.info(f"Progress: {i+1}/{len(documents)} - Saving state...")
                if self.agent.save_state():
                    logger.info("State saved successfully")
                    # Check embedding quality
                    self.verify_embedding_quality()
                else:
                    logger.warning("Failed to save state")
                    
            # Log progress
            if (i + 1) % 10 == 0:
                avg_time = np.mean(self.processing_times[-10:])
                logger.info(f"Processed {i+1}/{len(documents)} documents (avg time: {avg_time:.3f}s)")
        
        # Final save after all documents
        logger.info("Processing complete - Final state save...")
        if self.agent.save_state():
            logger.info("Final state saved successfully")
        else:
            logger.warning("Failed to save final state")
        
        return len(documents)
    
    def verify_embedding_quality(self):
        """Verify that embeddings are properly semantic"""
        if len(self.agent.l2_memory.episodes) < 2:
            return
        
        # Get last two episodes
        ep1 = self.agent.l2_memory.episodes[-2]
        ep2 = self.agent.l2_memory.episodes[-1]
        
        # Check vector properties
        vec1_norm = np.linalg.norm(ep1.vec)
        vec2_norm = np.linalg.norm(ep2.vec)
        
        # Vectors should be normalized
        if abs(vec1_norm - 1.0) > 0.01 or abs(vec2_norm - 1.0) > 0.01:
            logger.warning(f"Vectors not properly normalized: {vec1_norm:.3f}, {vec2_norm:.3f}")
        
        # Check dimensionality
        if ep1.vec.shape[0] != 384 or ep2.vec.shape[0] != 384:
            logger.warning(f"Incorrect vector dimensions: {ep1.vec.shape}, {ep2.vec.shape}")
    
    def test_retrieval_quality(self, n_queries=10):
        """Test retrieval quality with proper embeddings"""
        logger.info(f"\nTesting retrieval quality with {n_queries} queries...")
        
        test_queries = [
            "What is machine learning?",
            "How does deep learning work?",
            "Explain neural networks",
            "What are the applications of computer vision?",
            "How does natural language processing help?",
            "What is reinforcement learning?",
            "Describe data science methodologies",
            "What are quantum computing principles?",
            "How does blockchain technology work?",
            "What is edge computing?"
        ]
        
        results = []
        for i, query in enumerate(test_queries[:n_queries]):
            result = self.agent.process_question(query, max_cycles=2)
            
            # Extract retrieval metrics
            if 'documents' in result:
                retrieved_count = len(result['documents'])
                avg_similarity = np.mean([doc['similarity'] for doc in result['documents']]) if result['documents'] else 0
            else:
                retrieved_count = 0
                avg_similarity = 0
            
            quality = result.get('reasoning_quality', 0)
            
            results.append({
                'query': query,
                'retrieved_count': retrieved_count,
                'avg_similarity': avg_similarity,
                'quality': quality,
                'response_length': len(result.get('response', ''))
            })
            
            logger.info(f"Query {i+1}: Retrieved {retrieved_count} docs, "
                       f"avg similarity {avg_similarity:.3f}, quality {quality:.3f}")
        
        # Summary statistics
        avg_retrieved = np.mean([r['retrieved_count'] for r in results])
        avg_similarity = np.mean([r['avg_similarity'] for r in results])
        avg_quality = np.mean([r['quality'] for r in results])
        
        logger.info(f"\nRetrieval Summary:")
        logger.info(f"  Average documents retrieved: {avg_retrieved:.1f}")
        logger.info(f"  Average similarity score: {avg_similarity:.3f}")
        logger.info(f"  Average response quality: {avg_quality:.3f}")
        
        return results

def main():
    """Run the data regeneration with proper embeddings"""
    print("="*60)
    print("Regenerating Data Structure with Proper Embeddings")
    print("="*60)
    
    # Create generator instance
    generator = ProperEmbeddingDataGenerator()
    
    # Check initial state
    generator.check_data_state("Initial")
    
    # Create diverse documents
    print("\nCreating diverse documents...")
    documents = generator.create_diverse_documents(100)
    print(f"Created {len(documents)} documents")
    
    # Process documents with proper embeddings
    print("\nProcessing documents...")
    processed_count = generator.process_documents(documents)
    
    # Check final state
    generator.check_data_state("\nFinal")
    
    # Test retrieval quality
    print("\nTesting retrieval quality...")
    retrieval_results = generator.test_retrieval_quality(10)
    
    # Save generation report
    report = {
        'timestamp': datetime.now().isoformat(),
        'documents_processed': processed_count,
        'episodes_created': len(generator.agent.l2_memory.episodes),
        'avg_processing_time': np.mean(generator.processing_times),
        'retrieval_test_results': retrieval_results,
        'embedding_type': 'SentenceTransformer (all-MiniLM-L6-v2)'
    }
    
    output_dir = Path("results_proper_embeddings")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'generation_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=float)
    
    print(f"\nGeneration complete! Report saved to {output_dir}")
    print(f"Episodes created: {report['episodes_created']}")
    print(f"Average processing time: {report['avg_processing_time']:.3f}s")
    
    # Final check on graph state
    graph_state = generator.agent.get_memory_graph_state()
    print(f"\nGraph state: {graph_state['graph']}")

if __name__ == "__main__":
    main()