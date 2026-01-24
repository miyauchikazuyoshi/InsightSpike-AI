# Hybrid Small-World Architecture
**"Crystallized Memory, Fluid Thought"**

## 1. Philosophical Unification

### The Conflict
*   **HNSW (Hierarchical Navigable Small World)**: Fixed, static graph structure optimized for retrieval speed. Represents **"Crystallized Intelligence"** (Long-term Memory).
*   **geDIG (Graph-energy based Dynamic Inference Geometry)**: Variable, dynamic graph structure generated on-the-fly to minimize entropy. Represents **"Fluid Intelligence"** (Working Memory / Inference).

### The Synthesis
The brain utilizes both. We propose a hybrid architecture where HNSW provides the "fast initial retrieval" (System 1), and geDIG performs the "slow, rigorous structural evaluation" (System 2). Furthermore, repeated geDIG confirmations "crystallize" into the HNSW graph, turning fluid thoughts into permanent memories.

| Component | Role | Cognitive Analog | Speed | Plasticity |
|---|---|---|---|---|
| **HNSW** | Initial Retrieval | Long-term Memory / Hippocampus | Fast ($O(\log N)$) | Low (Periodic Indexing) |
| **geDIG** | Reranking / Gating | Working Memory / Prefrontal Cortex | Moderate (GPU Matrix Ops) | High (Dynamic per Context) |

---

## 2. Roadmap: From Static to Dynamic Crystallization

We propose a 4-phase roadmap to safely introduce this architecture while verifying the "flexibility" of the Small World.

### Phase 1: The Speed Floor (Static HNSW)
*   **Objective**: Establish a high-speed baseline.
*   **Implementation**: Replace `IndexFlatIP` with `IndexHNSWFlat`.
*   **Verification**:
    *   Measure QPS (Queries Per Second) improvement.
    *   Confirm Recall@10 does not degrade significantly vs Exact Search.
*   **Outcome**: "We are fast now."

### Phase 2: The Gating Mechanism (Hybrid RAG) **[Current State]**
*   **Objective**: Filter out "Fast but Wrong" results (Hallucinations).
*   **Implementation**:
    *   Retrieve top-K via HNSW.
    *   Rerank via Flash-geDIG (Banana Test logic).
*   **Verification**:
    *   Does geDIG correctly reject "Random Graph" results returned by HNSW?
*   **Outcome**: "We are fast and logically robust."

### Phase 3: Dynamic Crystallization (Write-Back)
*   **Objective**: "Thoughts that make sense should become easier to recall."
*   **Concept**: If geDIG finds a **High-F connection** between Query $Q$ and Doc $D$ (which HNSW might have ranked low):
    *   **Action**: Explicitly add an edge (or strengthen weight) between $Q$'s embedding cluster and $D$ in the persistent graph.
*   **Implementation**:
    *   Use a secondary "Semantic Graph" or update HNSW entry point heuristics.
*   **Verification**:
    *   Does the system "learn" to associate logical leaps over time?
*   **Outcome**: "The system learns from its own insights."

### Phase 4: Thought-Chain Graph (Recursive Small Worlds)
*   **Objective**: Crystallize entire chains of reasoning.
*   **Concept**: Instead of single documents, store "Graph Snippets" (High-F Subgraphs) as units of memory.
    *   When a new query hits a node in this subgraph, the whole "Thought Crystal" is activated.
*   **Implementation**: A higher-level "Graph of Graphs".
*   **Outcome**: True Episodic Memory where context is preserved structurally.

---

## 3. Implementation Plan (Immediate)

### Step 1: Enable HNSW in `factory.py`
*   Modify `VectorIndexFactory` to accept `index_type="hnsw"`.
*   Wrap `faiss.IndexHNSWFlat`.

### Step 2: Verification Experiment
*   Compare `Flat` vs `HNSW` on retrieval speed and recall.
*   Verify that geDIG Reranker works downstream regardless of the index type.

---

**"The measure of intelligence is the ability to change." — Albert Einstein**
By moving from Phase 1 to Phase 4, we transition the system from a "Static Library" to a "Living Organism" that physically changes its brain structure based on the quality of its thoughts.
