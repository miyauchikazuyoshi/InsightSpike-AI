"""
Configuration Examples for UnifiedMainAgent
=========================================

Shows how to configure the unified agent for different use cases.
"""

from unified_main_agent import UnifiedMainAgent, AgentConfig, AgentMode


def create_basic_agent():
    """Basic agent with minimal features (fastest, simplest)"""
    config = AgentConfig.from_mode(AgentMode.BASIC)
    return UnifiedMainAgent(config)


def create_enhanced_agent():
    """Enhanced agent with graph-aware memory"""
    config = AgentConfig.from_mode(AgentMode.ENHANCED)
    config.verbose = True
    return UnifiedMainAgent(config)


def create_query_transform_agent():
    """Agent with query transformation capabilities"""
    config = AgentConfig.from_mode(AgentMode.QUERY_TRANSFORM)
    config.max_cycles = 5  # Allow more cycles for query evolution
    return UnifiedMainAgent(config)


def create_advanced_agent():
    """Advanced agent with multi-hop reasoning and query branching"""
    config = AgentConfig.from_mode(AgentMode.ADVANCED)
    config.parallel_branches = 4
    return UnifiedMainAgent(config)


def create_production_agent():
    """Production-ready agent with all optimizations"""
    config = AgentConfig.from_mode(AgentMode.OPTIMIZED)
    config.cache_size = 2000
    config.embedding_batch_size = 64
    return UnifiedMainAgent(config)


def create_graph_centric_agent():
    """Graph-centric agent for pure graph-based reasoning"""
    config = AgentConfig.from_mode(AgentMode.GRAPH_CENTRIC)
    return UnifiedMainAgent(config)


def create_custom_agent():
    """Custom configuration example"""
    config = AgentConfig(
        mode=AgentMode.BASIC,
        # Enable specific features
        enable_query_transform=True,
        enable_caching=True,
        enable_graph_aware_memory=False,  # Disable graph for speed
        # Performance tuning
        max_cycles=3,
        cache_size=500,
        # Component configs
        llm_config={
            "model_name": "distilgpt2",
            "temperature": 0.7,
            "max_tokens": 100
        }
    )
    return UnifiedMainAgent(config)


# Migration examples showing how to replace old agents
def migrate_from_main_agent():
    """Replace: from insightspike.core.agents.main_agent import MainAgent"""
    # Old way:
    # agent = MainAgent(config)
    
    # New way:
    agent = create_basic_agent()
    return agent


def migrate_from_enhanced_main_agent():
    """Replace: from insightspike.core.agents.main_agent_enhanced import EnhancedMainAgent"""
    # Old way:
    # agent = EnhancedMainAgent(config)
    
    # New way:
    agent = create_enhanced_agent()
    return agent


def migrate_from_query_transform_agent():
    """Replace: from insightspike.core.agents.main_agent_with_query_transform import MainAgentWithQueryTransform"""
    # Old way:
    # agent = MainAgentWithQueryTransform(config)
    
    # New way:
    agent = create_query_transform_agent()
    return agent


if __name__ == "__main__":
    # Example usage
    agent = create_basic_agent()
    
    if agent.initialize():
        result = agent.process_question("What is consciousness?")
        print(f"Response: {result.get('response', 'No response')}")
        print(f"Quality: {result.get('reasoning_quality', 0):.3f}")
        print(f"Spike detected: {result.get('spike_detected', False)}")