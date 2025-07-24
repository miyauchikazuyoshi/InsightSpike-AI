# InsightSpike Architecture Documentation

## 📚 Documentation Index

### Core Architecture
- **[Directory Structure](directory_structure.md)** - Clean architecture and package organization
- **[Layer Architecture](layer_architecture.md)** - 4-layer neurobiologically-inspired processing system
- **[Agent Types](agent_types.md)** - Available agent implementations and their use cases
- **[MainAgent Behavior](mainagent_behavior.md)** - Detailed behavior and processing flow of MainAgent
- **[Recent Features (July 2024)](recent_features_2024_07.md)** - Layer1 bypass, insight auto-registration, mode-aware prompts, graph search

### User Interface
- **[CLI Commands](cli_commands.md)** - Complete command reference for the `spike` CLI

## 🧠 Quick Overview

InsightSpike implements a brain-inspired architecture with 4 processing layers:

1. **Layer 1 (Error Monitor)** - Cerebellum analog for error detection
2. **Layer 2 (Memory Manager)** - Hippocampus analog for episodic memory
3. **Layer 3 (Graph Reasoner)** - Prefrontal cortex analog for reasoning
4. **Layer 4 (Language Interface)** - Broca's/Wernicke's areas analog for language

## 🚀 Getting Started

### Basic Usage
```bash
# Add knowledge
spike embed ./documents/

# Ask questions
spike query "What is the main concept?"

# Interactive mode
spike chat

# View insights
spike insights
```

### For Developers
```python
from insightspike.implementations.agents import MainAgent

agent = MainAgent()
agent.initialize()

# Process question
result = agent.process_question("Your question here")
print(f"Answer: {result.response}")
print(f"Spike detected: {result.spike_detected}")
```

## 📊 Architecture Highlights

- **Neurobiologically-inspired** design based on brain structures
- **Multi-agent** system with specialized components
- **Graph-based** reasoning for insight detection
- **Configurable** behavior through modes and settings
- **Production-ready** with caching, async processing, and error handling

## 🔄 Recent Updates

### July 2024 Feature Release
- **Layer1 Bypass Mechanism** - 10x speedup for known queries in production
- **Insight Auto-Registration** - Automatic capture and reuse of discovered insights
- **Mode-Aware Prompt Building** - Dynamic prompt sizing based on model capabilities
- **Graph-Based Memory Search** - Multi-hop traversal for associative retrieval
- See **[Recent Features Documentation](recent_features_2024_07.md)** for details

### Core Package Refactoring (2025-07-18)
- **Separated abstractions from implementations**:
  - `core/` now only contains interfaces and base classes
  - `implementations/` contains all concrete implementations
  - `features/` contains feature modules (query transformation, etc.)
  - `tools/` contains standalone tools and experiments
- **Cleaner architecture** following SOLID principles
- **Better separation of concerns**

### Previous Updates (2025-07-17)
- Consolidated 6 agent variants → 1 ConfigurableAgent
- Unified 4 Layer2 variants → 1 L2MemoryManager
- Merged 8 LLM files → 1 L4LLMInterface
- Clear layer-based naming convention
- Full backward compatibility maintained

### CLI Improvements
- Separated business logic from presentation
- MainAgent now handles all core functionality
- CLI focuses purely on user interaction
- Better testability and maintainability

## 📖 Further Reading

- Project README for installation and setup
- API documentation for detailed usage
- Example notebooks for practical demonstrations