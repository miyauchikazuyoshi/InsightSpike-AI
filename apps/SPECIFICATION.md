# InsightSpike Local Application Specification

## 1. Overview
The InsightSpike Local Application is a personal knowledge management tool that leverages local Large Language Models (LLMs) to construct and query a dynamic knowledge graph. It is designed for privacy, ease of use, and "local-first" operation.

## 2. Architecture

```mermaid
graph TD
    User[User] -->|Interact| UI[Streamlit UI (knowledge_app.py)]
    UI -->|Calls| Wrapper[InsightAppWrapper (public/wrapper.py)]
    Wrapper -->|Manages| Agent[InsightSpike Agent]
    Agent -->|Inference| LLM[Local LLM (Ollama)]
    Agent -->|Storage| FS[Filesystem (data/local_app)]
    
    subgraph "Core Logic"
        Wrapper
        Agent
    end
    
    subgraph "Presentation"
        UI
        PyVis[PyVis Graph Renderer]
    end
```

## 3. Core Components

### 3.1. Frontend (`apps/knowledge_app.py`)
- **Framework**: Streamlit
- **Features**:
    - **Chat Interface**: Messages are persisted in interaction session state.
    - **Graph Visualization**: Uses `pyvis` to render interactive HTML graphs from NetworkX data.
    - **Ingestion UI**: Form for submitting text/documents for learning.
    - **Dark Theme**: Custom CSS injection for premium aesthetic.

### 3.2. Middleware (`src/insightspike/public/wrapper.py`)
- **Class**: `InsightAppWrapper`
- **Responsibility**: Wraps the complex Multi-Agent System (MAS) configuration into simple methods:
    - `ask(question: str)`: Wraps `agent.process_question()`.
    - `learn(text: str)`: Wraps ingestion and triggers `save()`.
    - `get_stats()`: Extracts graph metrics (node/edge count).
    - `save()`: Persists internal state to disk.

### 3.3. Backend / Persistence
- **Location**: `data/local_app/` (default)
- **Format**:
    - Agent state is serialized (typically JSON/Pickle depending on internal implementation).
    - Vector indices (if enabled) are stored as FAISS/Numpy files.

### 4. Data Flow

1.  **Query Flow**:
    - User types prompt in UI.
    - `App` calls `Wrapper.ask()`.
    - `Agent` retrieves relevant context from Graph/Vector Store.
    - `Agent` constructs prompt -> sends to `Ollama`.
    - Response displayed to user.

2.  **Learning Flow**:
    - User submits text in "Ingest" tab.
    - `App` calls `Wrapper.learn()`.
    - `Agent` processes text, extracting entities/relations.
    - `Wrapper` calls `Agent.save()` to commit changes to `data/local_app`.

## 5. Technology Stack

| Component | Technology | Usage |
|-----------|------------|-------|
| **LLM Host** | Ollama | Runs local models (Mistral, Llama3) via HTTP API |
| **App Framework** | Streamlit | UI and State Management |
| **Graph Viz** | PyVis | Interactive Javascript-based graph rendering |
| **Core Logic** | Python 3.10+ | InsightSpike Core Library |
| **Graph DB** | NetworkX | In-memory graph structure (persisted to file) |

## 6. Configuration
- **Environment Variables**: Can be set in `.env` or passed via UI.
- **Launcher**: `run_local_app.sh` handles Python venv creation and dependency installation (`streamlit`, `pyvis`).
