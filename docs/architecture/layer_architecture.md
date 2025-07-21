# InsightSpike 4-Layer Architecture

## 🧠 Brain-Inspired Processing Layers

### Layer 1: Error Monitor (`implementations/layers/layer1_error_monitor.py`)
- **Brain Analog**: Cerebellum (error correction and fine-tuning)
- **Purpose**: Query analysis and validation
- **Key Functions**:
  - Calculate prediction error
  - Measure uncertainty
  - Initial query validation

### Layer 2: Memory Manager (`implementations/layers/layer2_memory_manager.py`)
- **Brain Analog**: Hippocampus + Locus Coeruleus
- **Purpose**: Graph-centric episodic memory management
- **Key Functions**:
  - FAISS-indexed vector search
  - Episode management (add, merge, split, prune)
  - C-value → Graph-based importance (transitioning)
- **Modes**: Basic, Enhanced, Scalable, Graph-Centric
- **Variants**:
  - `layer2_compatibility.py`: Backward compatibility wrapper for old API
  - `layer2_working_memory.py`: DataStore-centric implementation for scalability

### Layer 3: Graph Reasoner (`implementations/layers/layer3_graph_reasoner.py`)
- **Brain Analog**: Prefrontal Cortex (executive function)
- **Purpose**: Structural analysis and insight detection
- **Key Functions**:
  - Build similarity graphs
  - Calculate geDIG (GED + IG)
  - Detect "Eureka spikes"
- **Core Innovation**: Mathematical insight modeling

### Layer 4: Language Interface (`implementations/layers/layer4_llm_interface.py`)
- **Brain Analog**: Broca's + Wernicke's areas
- **Purpose**: Natural language synthesis
- **Key Functions**:
  - Generate responses from context
  - Support multiple LLM providers
  - Context-aware synthesis

### Supporting Components
- `implementations/layers/layer4_prompt_builder.py`: Semantic content generation (true Layer 4)
- `implementations/layers/scalable_graph_builder.py`: Efficient graph construction

## 📊 Data Flow

```
User Query
    ↓
Layer 1: Validation & Error Check
    ↓
Layer 2: Memory Search (FAISS)
    ↓
Layer 3: Graph Analysis (geDIG)
    ↓
Layer 4: Response Generation
    ↓
Natural Language Output
```

## 🚀 Recent Refactoring (2025-07-18)

### Directory Structure Update
- **Moved all implementations** from `core/` to `implementations/`
- **Core package** now only contains interfaces and base classes
- **Features** like query transformation moved to `features/`
- **Tools** like standalone L3 moved to `tools/`

### Before: 17 files (chaotic)
- 4 Layer2 variants
- 8 LLM provider files
- Multiple graph builders
- Lots of duplication

### After: Clean separation
- `core/` - Only interfaces and base classes
- `implementations/layers/` - All layer implementations
  - `layer1_error_monitor.py`
  - `layer2_memory_manager.py` (unified)
  - `layer2_compatibility.py` (backward compatibility)
  - `layer2_working_memory.py` (DataStore-centric)
  - `layer3_graph_reasoner.py`
  - `layer4_llm_interface.py` (unified)
  - `layer4_prompt_builder.py`
  - `scalable_graph_builder.py`

### Improvements
- 70% code reduction
- Clear naming (Layer1-4)
- Configurable behaviors
- Backward compatibility
- Brain-inspired organization
- Clean architecture following SOLID principles