"""
DataStore-Centric Main Agent
=============================

Main agent implementation that uses DataStore as the primary storage backend.
This replaces the memory-intensive approach with a scalable, transaction-based approach.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ...config.models import InsightSpikeConfig
from ...core.base.datastore import DataStore
from ...core.episode import Episode
from ...detection.eureka_spike import EurekaDetector
from ...processing.embedder import EmbeddingManager
from ..layers.layer2_working_memory import L2WorkingMemoryManager, WorkingMemoryConfig
from ..layers.layer4_llm_interface import L4LLMInterface

logger = logging.getLogger(__name__)


class DataStoreMainAgent:
    """
    DataStore-centric implementation of the InsightSpike main agent.

    Key differences from MainAgent:
    - Uses DataStore for all persistence (not in-memory)
    - Working memory approach for active data
    - Lazy loading of data on demand
    - Better scalability for large datasets
    """

    def __init__(
        self,
        datastore: DataStore,
        config: Optional[Union[Dict, InsightSpikeConfig]] = None,
        **kwargs,
    ):
        """
        Initialize DataStore-centric agent.

        Args:
            datastore: DataStore instance for persistence
            config: Configuration (dict or InsightSpikeConfig)
            **kwargs: Additional configuration overrides
        """
        self.datastore = datastore
        self.config = self._process_config(config, kwargs)

        # Initialize components
        self.initialized = False
        self._initialize_components()

    def _process_config(self, config, kwargs) -> InsightSpikeConfig:
        """Process configuration from various sources"""
        if isinstance(config, InsightSpikeConfig):
            return config
        elif isinstance(config, dict):
            # Merge with kwargs
            config_dict = {**config, **kwargs}
            return InsightSpikeConfig(**config_dict)
        else:
            # Use default config with kwargs
            return InsightSpikeConfig(**kwargs)

    def _initialize_components(self):
        """Initialize agent components"""
        try:
            # Initialize embedding manager
            self.embedding_manager = EmbeddingManager()

            # Initialize working memory manager
            # Pass the entire config - L2WorkingMemoryManager will extract what it needs
            self.memory_manager = L2WorkingMemoryManager(
                datastore=self.datastore,
                config=self.config,
                embedding_manager=self.embedding_manager,
            )

            # Initialize LLM interface
            self.llm = L4LLMInterface(config=self.config)

            # Initialize spike detector
            # EurekaDetector doesn't take config directly
            self.spike_detector = None  # Disabled for now

            # Graph reasoning would be initialized here if available
            self.graph_reasoner = None

            self.initialized = True
            logger.info("DataStoreMainAgent initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            self.initialized = False

    def process(
        self,
        text: str,
        context: Optional[Union[str, List[str]]] = None,
        metadata: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Process input text through the InsightSpike pipeline.

        Args:
            text: Input text to process
            context: Optional context (string or list of strings)
            metadata: Optional metadata

        Returns:
            Processing results including insights and reasoning
        """
        if not self.initialized:
            logger.error("Agent not initialized")
            return {"error": "Agent not initialized"}

        start_time = time.time()

        try:
            # Phase 1: Store episode in DataStore
            episode_id = self.memory_manager.store_episode(
                text=text, c_value=0.5, metadata=metadata  # Initial confidence
            )

            if not episode_id:
                return {"error": "Failed to store episode"}

            # Phase 2: Search for related episodes
            related_episodes = self.memory_manager.search_episodes(
                query=text, k=self.config.memory.search_k
            )

            # Phase 3: Detect insights/spikes
            spike_result = self._detect_spike(text, related_episodes)

            # Phase 4: Generate reasoning (if spike detected)
            reasoning = None
            if spike_result.get("has_spike", False):
                reasoning = self._generate_reasoning(
                    text=text,
                    spike_info=spike_result,
                    related_episodes=related_episodes,
                )

            # Phase 5: Update episode with results
            if spike_result.get("has_spike", False):
                # Update confidence based on spike
                new_c_value = min(1.0, 0.5 + spike_result.get("confidence", 0.3))
                # In a real implementation, we'd update the episode in DataStore
                # self.datastore.update_episode(episode_id, {'c': new_c_value})

            # Prepare results
            result = {
                "episode_id": episode_id,
                "text": text,
                "has_spike": spike_result.get("has_spike", False),
                "spike_info": spike_result,
                "reasoning": reasoning,
                "related_episodes": len(related_episodes),
                "processing_time": time.time() - start_time,
                "metadata": metadata,
            }

            return result

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            return {"error": str(e), "processing_time": time.time() - start_time}

    def _detect_spike(self, text: str, related_episodes: List[Dict]) -> Dict[str, Any]:
        """Detect if the input represents an insight spike"""
        try:
            # Prepare context from related episodes
            context = []
            for ep in related_episodes[:5]:  # Top 5 most relevant
                context.append(
                    {
                        "text": ep.get("text", ""),
                        "similarity": ep.get("similarity", 0.0),
                        "c": ep.get("c", 0.5),
                    }
                )

            # Use spike detector
            if self.spike_detector:
                return self.spike_detector.detect(text=text, context=context)
            else:
                # Fallback: simple similarity-based detection
                if related_episodes and related_episodes[0].get("similarity", 0) > 0.9:
                    return {"has_spike": False, "reason": "Too similar to existing"}
                else:
                    return {"has_spike": True, "confidence": 0.7}

        except Exception as e:
            logger.error(f"Spike detection failed: {e}")
            return {"has_spike": False, "error": str(e)}

    def _generate_reasoning(
        self, text: str, spike_info: Dict, related_episodes: List[Dict]
    ) -> Optional[str]:
        """Generate reasoning for the detected spike"""
        try:
            # Prepare prompt
            prompt = self._build_reasoning_prompt(text, spike_info, related_episodes)

            # Generate reasoning using LLM
            if self.llm:
                response = self.llm.generate(prompt, max_tokens=200)
                return response
            else:
                return "Insight detected based on novelty and relevance."

        except Exception as e:
            logger.error(f"Reasoning generation failed: {e}")
            return None

    def _build_reasoning_prompt(
        self, text: str, spike_info: Dict, related_episodes: List[Dict]
    ) -> str:
        """Build prompt for reasoning generation"""
        prompt_parts = [
            "Analyze why this represents an insight spike:",
            f"Input: {text}",
            f"Spike confidence: {spike_info.get('confidence', 0.0)}",
            "\nRelated knowledge:",
        ]

        for i, ep in enumerate(related_episodes[:3]):
            prompt_parts.append(f"{i+1}. {ep.get('text', '')[:100]}...")

        prompt_parts.append("\nExplain the insight in 2-3 sentences.")

        return "\n".join(prompt_parts)

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for relevant episodes.

        Args:
            query: Search query
            k: Number of results

        Returns:
            List of relevant episodes
        """
        if not self.initialized:
            return []

        return self.memory_manager.search_episodes(query, k)

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        stats = {
            "initialized": self.initialized,
            "components": {
                "memory_manager": self.memory_manager is not None,
                "llm": self.llm is not None,
                "spike_detector": self.spike_detector is not None,
                "graph_reasoner": self.graph_reasoner is not None,
            },
        }

        # Add memory stats
        if self.memory_manager:
            stats["memory"] = self.memory_manager.get_memory_stats()

        # Add DataStore stats
        try:
            stats["datastore"] = self.datastore.get_stats()
        except Exception as e:
            logger.warning(f"Could not get DataStore stats: {e}")

        return stats

    def save_checkpoint(self, checkpoint_path: str) -> bool:
        """Save agent state (working memory only)"""
        try:
            checkpoint = {"config": self.config.dict(), "timestamp": time.time()}

            # Save working memory state
            if self.memory_manager:
                wm_checkpoint = f"{checkpoint_path}.working_memory"
                self.memory_manager.save_checkpoint(wm_checkpoint)
                checkpoint["working_memory_checkpoint"] = wm_checkpoint

            # Save main checkpoint
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint, f, indent=2)

            logger.info(f"Saved checkpoint to {checkpoint_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load agent state"""
        try:
            with open(checkpoint_path, "r") as f:
                checkpoint = json.load(f)

            # Load working memory if available
            if "working_memory_checkpoint" in checkpoint:
                wm_checkpoint = checkpoint["working_memory_checkpoint"]
                if self.memory_manager:
                    self.memory_manager.load_checkpoint(wm_checkpoint)

            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False
