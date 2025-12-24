
"""
Insight App Wrapper
===================

A high-level wrapper to easily build local knowledge applications.
"""

import os
import logging
from typing import Optional, Dict, Any, List

from ..config.models import LLMConfig
from ..quick_start import create_agent

logger = logging.getLogger(__name__)

class InsightAppWrapper:
    """
    A simplified interface for building local knowledge apps.
    
    Usage:
        app = InsightAppWrapper(provider="ollama", model="mistral")
        answer = app.ask("What is geDIG?")
        app.learn("geDIG is a gauge for knowledge graphs.")
    """
    
    def __init__(
        self,
        provider: str = "ollama",
        model: str = "mistral",
        api_base: Optional[str] = "http://localhost:11434/v1",
        api_key: str = "ollama",
        data_dir: str = "data/local_app",
        temperature: float = 0.7
    ):
        """
        Initialize the local application wrapper.
        
        Args:
            provider: LLM provider (ollama, local, openai, mock)
            model: Model identifier
            api_base: API base URL (crucial for Ollama/LocalAI)
            api_key: API Key (dummy for local)
            data_dir: Where to persist knowledge
            temperature: Generation temperature
        """
        self.data_dir = data_dir
        
        # Ensure data directory exists
        os.makedirs(data_dir, exist_ok=True)
        
        # Configure LLM args directly (passing as overrides to create_agent)
        # We avoid creating LLMConfig object manually to prevent conflict with overrides logic
        llm_kwargs = {
            "provider": provider,
            "model": model,
            "api_base": api_base,
            "api_key": api_key,
            "temperature": temperature
        }
        
        # Auto-map 'ollama' to 'openai' provider to use OpenAI-compatible API
        # This avoids needing the 'ollama' pip package installed
        real_provider = provider
        if provider == "ollama":
             real_provider = "openai"
             logger.info("Automatically mapping 'ollama' provider to 'openai' for compatibility")

        # Create Agent
        # We explicitly set up filesystem persistence
        self.agent = create_agent(
            provider=real_provider, # IMPORTANT: Must be passed as explicit arg, otherwise defaults to "mock"
            
            # Pass LLM config as flattened overrides
            llm__model=model,
            llm__api_base=api_base,
            llm__api_key=api_key,
            llm__temperature=temperature,
            
            # Configure persistence
            datastore__type="filesystem",
            datastore__root_path=os.path.join(data_dir, "store"),
            paths__data_dir=data_dir,
            processing__enable_learning=True
        )
        
        # Ensure we load any existing state if the agent supports it
        # (Assuming agents auto-load or have a load method)
        if hasattr(self.agent, "load"):
             try:
                 self.agent.load()
                 logger.info("Agent state loaded.")
             except Exception as e:
                 logger.warning(f"Could not load agent state: {e}")
        
        logger.info(f"InsightAppWrapper initialized with {provider}/{model} at {data_dir}")

    def save(self):
        """Save the current state of the knowledge base."""
        if hasattr(self.agent, "save"):
            try:
                self.agent.save()
                return True
            except Exception as e:
                logger.error(f"Failed to save agent: {e}")
                return False
        return False

    def ask(self, question: str) -> str:
        """
        Ask a question to the agent.
        
        Args:
            question: The user's query
            
        Returns:
            The text response
        """
        response_obj = self.agent.process_question(question)
        
        # Auto-save after interaction if configured or simple heuristic
        # self.save() 
        
        return response_obj.response

    def learn(self, text: str) -> bool:
        """
        Ingest text into the knowledge graph.
        
        Args:
            text: Information to learn
            
        Returns:
            Success status
        """
        try:
            if hasattr(self.agent, "add_knowledge"):
                try:
                    result = self.agent.add_knowledge(text)
                except Exception:
                    result = None
                if isinstance(result, dict):
                    if result.get("success") is True:
                        self.save()
                        logger.info("Learning text completed")
                        return True
                elif result is True:
                    self.save()
                    logger.info("Learning text completed")
                    return True

            # If the agent has a direct ingestion method, use it
            if hasattr(self.agent, "ingest"):
                self.agent.ingest(text)
            else:
                # Fallback: Process as a high-context "study" prompt
                # We prefix to guide the agent to treat this as factual ingestion
                prompt = f"Please learn and remember the following information: {text}"
                self.agent.process_question(prompt)

            # Save state immediately after learning
            self.save()

            logger.info("Learning text completed")
            return True
        except Exception as e:
            logger.error(f"Failed to learn: {e}")
            return False
            
    def _get_memory_manager(self) -> Optional[Any]:
        if hasattr(self.agent, "l2_memory"):
            return self.agent.l2_memory
        if hasattr(self.agent, "memory"):
            return self.agent.memory
        return None

    def _clean_episode_text(self, text: str) -> str:
        cleaned = text or ""
        if cleaned.startswith("Q:") and "\nA:" in cleaned:
            cleaned = cleaned.split("\nA:", 1)[0].replace("Q:", "", 1).strip()
        token = "Please learn and remember the following information:"
        if token in cleaned:
            cleaned = cleaned.split(token, 1)[1].strip()
        return cleaned or text

    def get_recent_episodes(self, limit: int = 5) -> List[Dict[str, Any]]:
        memory = self._get_memory_manager()
        episodes = getattr(memory, "episodes", None) if memory else None
        if not episodes:
            return []
        recent = list(episodes[-limit:])[::-1]
        items = []
        for ep in recent:
            raw_text = getattr(ep, "text", "")
            cleaned_text = self._clean_episode_text(raw_text)
            items.append(
                {
                    "text": cleaned_text,
                    "c_value": float(getattr(ep, "c", 0.5)),
                    "timestamp": float(getattr(ep, "timestamp", 0.0)),
                }
            )
        return items

    def get_graph_networkx(
        self,
        max_nodes: int = 50,
        min_similarity: float = 0.65,
        top_k: int = 3,
        force_connect: bool = True,
    ):
        """
        Build a lightweight NetworkX graph from recent episodes.
        """
        memory = self._get_memory_manager()
        episodes = getattr(memory, "episodes", None) if memory else None
        if not episodes:
            return None

        try:
            import numpy as np
            import networkx as nx
        except Exception:
            return None

        selected = list(episodes[-max_nodes:])
        nodes = []
        vectors = []
        for idx, ep in enumerate(selected):
            vec = getattr(ep, "vec", None)
            if vec is None:
                continue
            nodes.append((idx, ep))
            vectors.append(vec)

        if not nodes:
            return None

        vectors = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = vectors / norms
        sims = normed @ normed.T

        def _color_for_c(value: float) -> str:
            if value >= 0.8:
                return "#2ca02c"
            if value >= 0.6:
                return "#1f77b4"
            if value >= 0.4:
                return "#ff7f0e"
            return "#d62728"

        graph = nx.Graph()
        for i, ep in nodes:
            raw_text = getattr(ep, "text", "")
            text = self._clean_episode_text(raw_text)
            label = text if len(text) <= 50 else text[:47] + "..."
            c_value = float(getattr(ep, "c", 0.5))
            graph.add_node(
                i,
                label=label,
                title=f"{text}\nC={c_value:.2f}",
                value=max(1.0, c_value * 10.0),
                color=_color_for_c(c_value),
                c_value=c_value,
            )

        def add_edges(threshold: Optional[float]) -> None:
            for i in range(len(nodes)):
                row = sims[i]
                candidates = [
                    (j, float(row[j]))
                    for j in range(len(nodes))
                    if j != i and (threshold is None or row[j] >= threshold)
                ]
                candidates.sort(key=lambda x: x[1], reverse=True)
                for j, score in candidates[:top_k]:
                    a = nodes[i][0]
                    b = nodes[j][0]
                    if a != b:
                        graph.add_edge(
                            a,
                            b,
                            weight=score,
                            value=max(0.1, score * 5.0),
                            title=f"sim={score:.3f}",
                        )

        add_edges(min_similarity)
        if force_connect and graph.number_of_edges() == 0 and len(nodes) > 1:
            add_edges(None)

        return graph

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.
        """
        try:
            memory = self._get_memory_manager()
            stats = {}
            if memory and hasattr(memory, "get_memory_stats"):
                stats = memory.get_memory_stats()
            nodes = stats.get("graph_nodes")
            if nodes is None:
                nodes = stats.get("total_episodes", 0)
            edges = stats.get("graph_edges", 0)
            return {
                "nodes": int(nodes or 0),
                "edges": int(edges or 0),
                "episodes": int(stats.get("total_episodes", nodes or 0) or 0),
            }
        except Exception:
            return {"nodes": 0, "edges": 0, "episodes": 0}

    def reset(self):
        """Reset the agent state."""
        # Re-initialize logic would go here
        pass
