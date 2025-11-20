#!/usr/bin/env python3
"""
Prompt Length Optimization Strategy
===================================

Demonstrates strategies to control prompt length when insights are included.
"""

def calculate_prompt_length_example():
    """Calculate typical prompt lengths with different configurations"""
    
    # Typical document/insight lengths
    avg_doc_length = 150  # characters
    avg_insight_length = 100  # characters
    
    print("=== Prompt Length Analysis ===\n")
    
    # Scenario 1: Default configuration
    print("1. Default Configuration:")
    num_docs = 10
    num_insights = 5
    total_chars = (num_docs * avg_doc_length) + (num_insights * avg_insight_length)
    total_tokens = total_chars // 4  # Rough estimate: 4 chars per token
    print(f"   Documents: {num_docs} × {avg_doc_length} chars = {num_docs * avg_doc_length} chars")
    print(f"   Insights: {num_insights} × {avg_insight_length} chars = {num_insights * avg_insight_length} chars")
    print(f"   Total: {total_chars} chars ≈ {total_tokens} tokens\n")
    
    # Scenario 2: Optimized configuration
    print("2. Optimized Configuration:")
    num_docs = 5  # Reduced
    num_insights = 3  # Reduced
    total_chars = (num_docs * avg_doc_length) + (num_insights * avg_insight_length)
    total_tokens = total_chars // 4
    print(f"   Documents: {num_docs} × {avg_doc_length} chars = {num_docs * avg_doc_length} chars")
    print(f"   Insights: {num_insights} × {avg_insight_length} chars = {num_insights * avg_insight_length} chars")
    print(f"   Total: {total_chars} chars ≈ {total_tokens} tokens\n")
    
    # Scenario 3: Aggressive optimization
    print("3. Aggressive Optimization:")
    num_docs = 3
    num_insights = 2
    doc_truncate = 80  # Truncate documents
    insight_truncate = 60  # Truncate insights
    total_chars = (num_docs * doc_truncate) + (num_insights * insight_truncate)
    total_tokens = total_chars // 4
    print(f"   Documents: {num_docs} × {doc_truncate} chars = {num_docs * doc_truncate} chars")
    print(f"   Insights: {num_insights} × {insight_truncate} chars = {num_insights * insight_truncate} chars")
    print(f"   Total: {total_chars} chars ≈ {total_tokens} tokens\n")


def optimization_strategies():
    """Suggest optimization strategies"""
    
    print("=== Optimization Strategies ===\n")
    
    print("1. Dynamic Document Limits:")
    print("   ```python")
    print("   # Adjust based on presence of insights")
    print("   if has_insights:")
    print("       max_docs = 5  # Reduce regular docs")
    print("   else:")
    print("       max_docs = 10  # Normal limit")
    print("   ```\n")
    
    print("2. Token Budget Approach:")
    print("   ```python")
    print("   # Allocate token budget")
    print("   total_budget = 2000  # tokens")
    print("   insight_budget = 500  # Reserve for insights")
    print("   doc_budget = total_budget - insight_budget")
    print("   ```\n")
    
    print("3. Relevance-Based Filtering:")
    print("   ```python")
    print("   # Higher threshold when insights present")
    print("   min_relevance = 0.8 if has_insights else 0.6")
    print("   filtered_docs = [d for d in docs if d['similarity'] > min_relevance]")
    print("   ```\n")
    
    print("4. Content Summarization:")
    print("   ```python")
    print("   # Truncate or summarize long content")
    print("   def truncate_smart(text, max_len=100):")
    print("       if len(text) <= max_len:")
    print("           return text")
    print("       # Find sentence boundary")
    print("       truncated = text[:max_len]")
    print("       last_period = truncated.rfind('.')")
    print("       if last_period > 50:")
    print("           return truncated[:last_period+1]")
    print("       return truncated + '...'")
    print("   ```\n")
    
    print("5. Insight Deduplication:")
    print("   ```python")
    print("   # Remove similar insights")
    print("   def deduplicate_insights(insights):")
    print("       unique = []")
    print("       for insight in insights:")
    print("           if not any(similar(insight, u) > 0.9 for u in unique):")
    print("               unique.append(insight)")
    print("       return unique[:3]  # Max 3 unique insights")
    print("   ```")


def proposed_config_updates():
    """Suggest configuration updates"""
    
    print("\n=== Proposed Configuration Updates ===\n")
    
    print("Add to ProcessingConfig:")
    print("```python")
    print("# Prompt optimization settings")
    print("max_total_context_tokens: int = Field(")
    print("    default=2000,")
    print("    description='Maximum tokens for entire context'")
    print(")")
    print("insight_token_budget: int = Field(")
    print("    default=500,")
    print("    description='Reserved tokens for insights'")
    print(")")
    print("dynamic_doc_adjustment: bool = Field(")
    print("    default=True,")
    print("    description='Reduce docs when insights present'")
    print(")")
    print("min_relevance_with_insights: float = Field(")
    print("    default=0.8,")
    print("    description='Higher relevance threshold with insights'")
    print(")")
    print("```")


if __name__ == "__main__":
    calculate_prompt_length_example()
    optimization_strategies()
    proposed_config_updates()
    
    print("\n=== Summary ===")
    print("Without optimization: ~500-625 tokens typical")
    print("With optimization: ~225-360 tokens")
    print("Token savings: 40-50%")
    print("\nRecommendation: Implement dynamic document limiting")
    print("when insights are present to maintain reasonable prompt length.")