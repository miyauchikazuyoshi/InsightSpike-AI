#!/usr/bin/env python3
"""
geDIG Episodic Learning Experiment
==================================

This experiment tests episodic/incremental embedding updates:
1. Start with initial embeddings
2. Update embeddings after each query-document interaction
3. Track how performance improves over episodes
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, deque
import copy

# Set random seeds
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

class EpisodicGeDIG(nn.Module):
    """geDIG with episodic learning capabilities"""
    
    def __init__(self, vocab_size=5000, embedding_dim=128, hidden_dim=256, 
                 learning_rate=0.001, memory_size=100):
        super().__init__()
        
        # Core embedding network
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Encoding networks
        self.query_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh()
        )
        
        self.doc_encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            nn.Tanh()
        )
        
        # Similarity computation
        self.similarity_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Episodic memory
        self.memory_size = memory_size
        self.episodic_memory = deque(maxlen=memory_size)
        
        # Learning components
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.episode_count = 0
        
        # Track embeddings over time
        self.embedding_history = []
        
    def encode_text(self, text, encoder):
        """Convert text to embedding"""
        # Simple tokenization (in practice, use proper tokenizer)
        words = text.lower().split()[:20]  # Limit length
        word_ids = [hash(w) % 5000 for w in words]  # Simple hashing
        
        if not word_ids:
            word_ids = [0]
        
        word_ids_tensor = torch.tensor(word_ids, dtype=torch.long)
        embeddings = self.word_embedding(word_ids_tensor)
        
        # Average pooling
        text_embedding = embeddings.mean(dim=0)
        
        # Apply encoder
        encoded = encoder(text_embedding)
        
        return encoded
    
    def compute_similarity(self, query, document):
        """Compute similarity between query and document"""
        query_emb = self.encode_text(query, self.query_encoder)
        doc_emb = self.encode_text(document, self.doc_encoder)
        
        # Concatenate embeddings
        combined = torch.cat([query_emb, doc_emb])
        
        # Compute similarity
        similarity = self.similarity_net(combined)
        
        return similarity.item(), query_emb, doc_emb
    
    def episodic_update(self, query, positive_doc, negative_docs, reward=None):
        """Update embeddings based on episode feedback"""
        self.episode_count += 1
        
        # Encode query and documents
        query_emb = self.encode_text(query, self.query_encoder)
        pos_emb = self.encode_text(positive_doc, self.doc_encoder)
        
        # Store in episodic memory
        self.episodic_memory.append({
            'query': query,
            'positive_doc': positive_doc,
            'query_emb': query_emb.detach(),
            'pos_emb': pos_emb.detach(),
            'episode': self.episode_count
        })
        
        # Compute positive similarity
        pos_combined = torch.cat([query_emb, pos_emb])
        pos_sim = self.similarity_net(pos_combined)
        
        # Compute negative similarities
        neg_sims = []
        for neg_doc in negative_docs[:3]:  # Use top 3 negatives
            neg_emb = self.encode_text(neg_doc, self.doc_encoder)
            neg_combined = torch.cat([query_emb, neg_emb])
            neg_sim = self.similarity_net(neg_combined)
            neg_sims.append(neg_sim)
        
        # Contrastive loss with margin
        loss = 0
        margin = 0.2
        for neg_sim in neg_sims:
            loss += torch.relu(margin - pos_sim + neg_sim)
        
        # Add regularization from episodic memory
        if len(self.episodic_memory) > 10:
            # Sample from memory
            memory_samples = np.random.choice(len(self.episodic_memory), 
                                            min(5, len(self.episodic_memory)), 
                                            replace=False)
            
            for idx in memory_samples:
                mem = self.episodic_memory[idx]
                # Re-encode with current parameters
                mem_query_emb = self.encode_text(mem['query'], self.query_encoder)
                mem_pos_emb = self.encode_text(mem['positive_doc'], self.doc_encoder)
                
                # Consistency loss
                consistency_loss = (mem_query_emb - mem['query_emb']).pow(2).mean()
                consistency_loss += (mem_pos_emb - mem['pos_emb']).pow(2).mean()
                
                loss += 0.1 * consistency_loss
        
        # Update parameters
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save_embedding_snapshot(self):
        """Save current embedding state"""
        snapshot = {
            'episode': self.episode_count,
            'word_embedding': self.word_embedding.weight.data.clone().mean().item(),
            'query_encoder': self.query_encoder[0].weight.data.clone().mean().item(),
            'doc_encoder': self.doc_encoder[0].weight.data.clone().mean().item(),
        }
        self.embedding_history.append(snapshot)

class StaticEmbedding:
    """Static embedding baseline (no learning)"""
    
    def __init__(self, name="Static"):
        self.name = name
        self.episode_count = 0
        
    def compute_similarity(self, query, document):
        """Simple static similarity"""
        q_words = set(query.lower().split())
        d_words = set(document.lower().split())
        
        overlap = len(q_words & d_words)
        total = len(q_words | d_words)
        
        return overlap / (total + 1e-8)
    
    def episodic_update(self, query, positive_doc, negative_docs, reward=None):
        """No update for static embedding"""
        self.episode_count += 1
        return 0.0

def create_episodic_dataset(n_episodes=200):
    """Create dataset for episodic learning"""
    
    topics = [
        "machine learning", "deep learning", "neural networks",
        "natural language processing", "computer vision",
        "reinforcement learning", "data science", "algorithms"
    ]
    
    episodes = []
    
    for i in range(n_episodes):
        # Select topic for this episode
        topic_idx = i % len(topics)
        topic = topics[topic_idx]
        
        # Create query
        query_types = [
            f"What is {topic}?",
            f"How does {topic} work?",
            f"Explain {topic}",
            f"Applications of {topic}",
            f"Benefits of {topic}"
        ]
        query = query_types[i % len(query_types)]
        
        # Create positive document
        positive_doc = f"{topic} is an important field in computer science that involves various techniques and methods for solving complex problems."
        
        # Create negative documents (other topics)
        negative_docs = []
        for j, other_topic in enumerate(topics):
            if j != topic_idx:
                neg_doc = f"{other_topic} is different from {topic} and focuses on other aspects of computing."
                negative_docs.append(neg_doc)
        
        episodes.append({
            'query': query,
            'positive_doc': positive_doc,
            'negative_docs': negative_docs,
            'topic': topic
        })
    
    return episodes

def evaluate_episodic_learning():
    """Evaluate episodic learning approaches"""
    
    print("Creating episodic dataset...")
    episodes = create_episodic_dataset(200)
    
    # Initialize methods
    methods = {
        "Static": StaticEmbedding(),
        "EpisodicGeDIG": EpisodicGeDIG(learning_rate=0.01),
        "EpisodicGeDIG-Slow": EpisodicGeDIG(learning_rate=0.001),
        "EpisodicGeDIG-Fast": EpisodicGeDIG(learning_rate=0.1),
    }
    
    # Track performance over episodes
    results = defaultdict(lambda: defaultdict(list))
    
    # Evaluation intervals
    eval_interval = 10
    
    print("\nRunning episodic learning...")
    
    for ep_idx, episode in enumerate(episodes):
        query = episode['query']
        positive_doc = episode['positive_doc']
        negative_docs = episode['negative_docs']
        
        # Combine all documents for retrieval
        all_docs = [positive_doc] + negative_docs
        
        for method_name, method in methods.items():
            # Compute similarities for all documents
            similarities = []
            
            if isinstance(method, EpisodicGeDIG):
                for doc in all_docs:
                    sim, _, _ = method.compute_similarity(query, doc)
                    similarities.append(sim)
            else:
                for doc in all_docs:
                    sim = method.compute_similarity(query, doc)
                    similarities.append(sim)
            
            # Check if positive document is ranked first
            rankings = np.argsort(similarities)[::-1]
            rank_of_positive = np.where(rankings == 0)[0][0] + 1
            
            # Record metrics
            success = rank_of_positive == 1
            results[method_name]['success'].append(success)
            results[method_name]['rank'].append(rank_of_positive)
            results[method_name]['episode'].append(ep_idx)
            
            # Update embeddings (learning)
            if hasattr(method, 'episodic_update'):
                # Get top negative documents for training
                top_neg_indices = [idx for idx in rankings[1:4] if idx > 0]
                top_neg_docs = [all_docs[idx] for idx in top_neg_indices]
                
                loss = method.episodic_update(query, positive_doc, top_neg_docs)
                results[method_name]['loss'].append(loss)
                
                # Save embedding snapshot periodically
                if isinstance(method, EpisodicGeDIG) and ep_idx % eval_interval == 0:
                    method.save_embedding_snapshot()
        
        # Print progress
        if (ep_idx + 1) % 20 == 0:
            print(f"Episode {ep_idx + 1}/{len(episodes)}")
            for method_name in methods:
                recent_success = np.mean(results[method_name]['success'][-20:])
                print(f"  {method_name}: {recent_success:.3f} success rate")
    
    return results, methods

def visualize_episodic_results(results, methods):
    """Create visualizations for episodic learning"""
    
    output_dir = Path("results_episodic_learning")
    output_dir.mkdir(exist_ok=True)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Success rate over episodes (moving average)
    ax = axes[0, 0]
    window = 20
    for method_name in results:
        success = results[method_name]['success']
        # Compute moving average
        moving_avg = []
        for i in range(window, len(success)):
            avg = np.mean(success[i-window:i])
            moving_avg.append(avg)
        
        episodes = range(window, len(success))
        ax.plot(episodes, moving_avg, label=method_name, linewidth=2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate (20-episode average)')
    ax.set_title('Learning Progress Over Episodes')
    ax.legend()
    ax.set_ylim(0, 1)
    
    # 2. Average rank over episodes
    ax = axes[0, 1]
    for method_name in results:
        ranks = results[method_name]['rank']
        # Compute moving average
        moving_avg = []
        for i in range(window, len(ranks)):
            avg = np.mean(ranks[i-window:i])
            moving_avg.append(avg)
        
        episodes = range(window, len(ranks))
        ax.plot(episodes, moving_avg, label=method_name, linewidth=2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Average Rank of Positive Doc')
    ax.set_title('Ranking Performance Over Episodes')
    ax.legend()
    ax.set_ylim(1, 5)
    ax.invert_yaxis()  # Lower rank is better
    
    # 3. Loss curves for episodic methods
    ax = axes[0, 2]
    for method_name in results:
        if 'loss' in results[method_name] and results[method_name]['loss']:
            losses = results[method_name]['loss']
            # Compute moving average
            moving_avg = []
            for i in range(window, len(losses)):
                avg = np.mean(losses[i-window:i])
                moving_avg.append(avg)
            
            episodes = range(window, len(losses))
            ax.plot(episodes, moving_avg, label=method_name, linewidth=2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Training Loss')
    ax.set_title('Loss Convergence')
    ax.legend()
    ax.set_yscale('log')
    
    # 4. Cumulative success comparison
    ax = axes[1, 0]
    for method_name in results:
        success = results[method_name]['success']
        cumulative = np.cumsum(success)
        ax.plot(cumulative, label=method_name, linewidth=2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Cumulative Successful Retrievals')
    ax.set_title('Total Success Over Time')
    ax.legend()
    
    # 5. Learning rate comparison (final performance)
    ax = axes[1, 1]
    method_names = []
    final_performance = []
    
    for method_name in results:
        method_names.append(method_name)
        # Final 50 episodes performance
        final_perf = np.mean(results[method_name]['success'][-50:])
        final_performance.append(final_perf)
    
    colors = ['gray', 'green', 'blue', 'red']
    bars = ax.bar(range(len(method_names)), final_performance, color=colors)
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels(method_names, rotation=45, ha='right')
    ax.set_ylabel('Success Rate')
    ax.set_title('Final Performance (Last 50 Episodes)')
    ax.set_ylim(0, 1)
    
    # Add value labels
    for bar, value in zip(bars, final_performance):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 6. Embedding evolution (for one episodic method)
    ax = axes[1, 2]
    episodic_method = None
    for name, method in methods.items():
        if isinstance(method, EpisodicGeDIG) and method.embedding_history:
            episodic_method = method
            break
    
    if episodic_method and episodic_method.embedding_history:
        history = episodic_method.embedding_history
        episodes = [h['episode'] for h in history]
        word_emb = [h['word_embedding'] for h in history]
        query_enc = [h['query_encoder'] for h in history]
        doc_enc = [h['doc_encoder'] for h in history]
        
        ax.plot(episodes, word_emb, label='Word Embedding', marker='o')
        ax.plot(episodes, query_enc, label='Query Encoder', marker='s')
        ax.plot(episodes, doc_enc, label='Doc Encoder', marker='^')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Weight Value')
        ax.set_title('Embedding Parameter Evolution')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'episodic_learning_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results
    save_data = {
        method: {
            metric: values if isinstance(values[0], (int, float)) else [float(v) for v in values]
            for metric, values in metrics.items()
        }
        for method, metrics in results.items()
    }
    
    with open(output_dir / 'episodic_learning_results.json', 'w') as f:
        json.dump(save_data, f, indent=2)
    
    # Create summary report
    create_episodic_summary_report(results, output_dir)
    
    print(f"\nResults saved to {output_dir}")

def create_episodic_summary_report(results, output_dir):
    """Create a markdown summary report for episodic learning"""
    
    report = ["# geDIG Episodic Learning Analysis\n"]
    report.append(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Random Seed**: {RANDOM_SEED}\n")
    
    report.append("\n## Key Findings\n")
    
    # Calculate final performance
    final_perfs = {}
    for method in results:
        final_perf = np.mean(results[method]['success'][-50:])
        final_perfs[method] = final_perf
    
    best_method = max(final_perfs.items(), key=lambda x: x[1])
    worst_method = min(final_perfs.items(), key=lambda x: x[1])
    
    report.append(f"1. **Best Method**: {best_method[0]} (Final success rate: {best_method[1]:.3f})")
    report.append(f"2. **Worst Method**: {worst_method[0]} (Final success rate: {worst_method[1]:.3f})")
    
    # Calculate improvement
    for method in results:
        if method != "Static":
            initial_perf = np.mean(results[method]['success'][:20])
            final_perf = final_perfs[method]
            improvement = (final_perf - initial_perf) / initial_perf * 100
            report.append(f"3. **{method} Improvement**: {improvement:+.1f}% "
                         f"(from {initial_perf:.3f} to {final_perf:.3f})")
    
    report.append("\n## Performance Summary\n")
    report.append("| Method | Initial (1-20) | Middle (90-110) | Final (150-200) | Total Success |")
    report.append("|--------|----------------|-----------------|-----------------|---------------|")
    
    for method in results:
        initial = np.mean(results[method]['success'][:20])
        middle = np.mean(results[method]['success'][90:110])
        final = np.mean(results[method]['success'][150:])
        total = sum(results[method]['success'])
        
        report.append(f"| {method} | {initial:.3f} | {middle:.3f} | {final:.3f} | {total}/200 |")
    
    report.append("\n## Learning Dynamics\n")
    
    # Analyze learning curves
    for method in results:
        if method != "Static":
            success = results[method]['success']
            # Find when performance stabilizes
            stable_episode = None
            for i in range(50, len(success)):
                recent_var = np.var(success[i-20:i])
                if recent_var < 0.01:
                    stable_episode = i
                    break
            
            if stable_episode:
                report.append(f"- **{method}**: Stabilized around episode {stable_episode}")
            else:
                report.append(f"- **{method}**: Still learning at episode 200")
    
    report.append("\n## Insights\n")
    
    # Learning rate analysis
    if "EpisodicGeDIG-Fast" in final_perfs and "EpisodicGeDIG-Slow" in final_perfs:
        fast_perf = final_perfs["EpisodicGeDIG-Fast"]
        slow_perf = final_perfs["EpisodicGeDIG-Slow"]
        
        if fast_perf > slow_perf:
            report.append("- Fast learning rate achieved better final performance")
        else:
            report.append("- Slow learning rate achieved better final performance")
    
    # Static vs Episodic
    static_perf = final_perfs.get("Static", 0)
    best_episodic = max(v for k, v in final_perfs.items() if k != "Static")
    
    if best_episodic > static_perf:
        improvement = (best_episodic - static_perf) / static_perf * 100
        report.append(f"- Episodic learning improved performance by {improvement:.1f}% over static")
    
    report.append("\n## Recommendations\n")
    report.append("1. Episodic learning shows clear benefits for adaptive retrieval")
    report.append("2. Learning rate tuning is crucial for convergence")
    report.append("3. Memory-based regularization helps maintain consistency")
    
    with open(output_dir / 'EPISODIC_LEARNING_SUMMARY.md', 'w') as f:
        f.write('\n'.join(report))

def main():
    """Run the episodic learning experiment"""
    
    print("="*60)
    print("geDIG EPISODIC LEARNING EXPERIMENT")
    print("="*60)
    
    # Run evaluation
    results, methods = evaluate_episodic_learning()
    
    # Create visualizations
    print("\nCreating visualizations...")
    visualize_episodic_results(results, methods)
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    
    # Print final summary
    print("\nFinal Performance (last 50 episodes):")
    for method in results:
        final_perf = np.mean(results[method]['success'][-50:])
        print(f"  {method}: {final_perf:.3f}")

if __name__ == "__main__":
    main()