"""State-persistence service used by :class:`MainAgent`.

The service is intentionally stateless with respect to agent components.
Callers pass the current DataStore, memory, and graph layer on every call so
runtime/test replacements of those public attributes remain observable.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List


class AgentPersistence:
    """Coordinate agent snapshots without owning their dependencies."""

    STATE_NAMESPACE = "agent_state"
    DEFAULT_NAMESPACE = "default"
    MAIN_GRAPH_ID = "main_graph"

    def __init__(self, *, logger: logging.Logger):
        self._logger = logger

    def replace_episode_snapshot(
        self,
        datastore: Any,
        episodes: List[Dict[str, Any]],
        *,
        namespace: str,
    ) -> bool:
        """Persist an exact snapshot, with a legacy append-store fallback."""

        replace = getattr(datastore, "replace_episodes", None)
        if callable(replace):
            return bool(replace(episodes, namespace=namespace))

        save = getattr(datastore, "save_episodes", None)
        if callable(save):
            self._logger.warning(
                "DataStore %s lacks replace_episodes(); using legacy save semantics",
                type(datastore).__name__,
            )
            return bool(save(episodes, namespace=namespace))
        return False

    def save_datastore_state(
        self,
        *,
        datastore: Any,
        memory: Any,
        graph_layer: Any,
        episode_encoder: Callable[[Any], Dict[str, Any]],
        replace_episode_snapshot: Callable[..., bool],
    ) -> bool:
        """Save memory and graph state through a configured DataStore."""

        try:
            if memory and hasattr(memory, "episodes"):
                episodes_to_save = [
                    episode_encoder(episode)
                    for episode in memory.episodes
                ]
                if not replace_episode_snapshot(
                    episodes_to_save,
                    namespace=self.STATE_NAMESPACE,
                ):
                    self._logger.error(
                        "DataStore rejected the episode snapshot"
                    )
                    return False
                self._logger.info(
                    "Saved %d episodes via DataStore",
                    len(episodes_to_save),
                )

            if (
                graph_layer
                and hasattr(graph_layer, "previous_graph")
                and graph_layer.previous_graph is not None
            ):
                graph_saved = datastore.save_graph(
                    graph_layer.previous_graph,
                    graph_id=self.MAIN_GRAPH_ID,
                    namespace=self.STATE_NAMESPACE,
                )
                if graph_saved is False:
                    self._logger.error(
                        "DataStore rejected the graph snapshot"
                    )
                    return False
                self._logger.info("Saved graph via DataStore")

            return True
        except Exception as exc:
            self._logger.error(
                "Failed to save agent state via DataStore: %s",
                exc,
            )
            return False

    def load_datastore_state(
        self,
        *,
        datastore: Any,
        memory: Any,
        graph_layer: Any,
        episode_decoder: Callable[[Any], Any],
    ) -> bool:
        """Load memory and graph state through a configured DataStore."""

        try:
            if memory:
                loaded_episodes = datastore.load_episodes(
                    namespace=self.STATE_NAMESPACE
                )
                memory.episodes = [
                    episode_decoder(record)
                    for record in loaded_episodes
                ]
                rebuild_index = getattr(memory, "_rebuild_index", None)
                if callable(rebuild_index):
                    rebuild_index()

                if loaded_episodes:
                    self._logger.info(
                        "Loaded %d episodes via DataStore",
                        len(loaded_episodes),
                    )
                else:
                    self._logger.warning(
                        "No episodes found in DataStore"
                    )

            if graph_layer:
                loaded_graph = datastore.load_graph(
                    graph_id=self.MAIN_GRAPH_ID,
                    namespace=self.STATE_NAMESPACE,
                )
                if loaded_graph is not None:
                    graph_layer.previous_graph = loaded_graph
                    node_count = getattr(
                        loaded_graph,
                        "num_nodes",
                        None,
                    )
                    detail = (
                        f": {node_count} nodes"
                        if node_count is not None
                        else ""
                    )
                    self._logger.info(
                        "Loaded graph via DataStore%s",
                        detail,
                    )
                else:
                    self._logger.warning(
                        "No graph found in DataStore"
                    )

            return True
        except Exception as exc:
            self._logger.error(
                "Failed to load agent state via DataStore: %s",
                exc,
            )
            return False

    def persist_live_episode_snapshots(
        self,
        *,
        episodes: Iterable[Any],
        episode_encoder: Callable[[Any], Dict[str, Any]],
        replace_episode_snapshot: Callable[..., bool],
        namespaces: Iterable[str] = (
            DEFAULT_NAMESPACE,
            STATE_NAMESPACE,
        ),
    ) -> Dict[str, bool]:
        """Best-effort synchronization used after live knowledge updates."""

        records = [episode_encoder(episode) for episode in episodes]
        outcomes: Dict[str, bool] = {}
        for namespace in namespaces:
            saved = bool(
                replace_episode_snapshot(
                    records,
                    namespace=namespace,
                )
            )
            outcomes[namespace] = saved
            if not saved:
                self._logger.warning(
                    "Failed to persist episode snapshot to %s",
                    namespace,
                )
        return outcomes

    def save_legacy_state(
        self,
        *,
        memory: Any,
        graph_layer: Any,
    ) -> bool:
        """Save through the legacy L2/L3 persistence methods."""

        try:
            success = True
            if memory:
                if not memory.save():
                    self._logger.warning("Failed to save L2 memory")
                    success = False
                else:
                    self._logger.info(
                        "L2 memory saved successfully"
                    )

            if (
                graph_layer
                and graph_layer.previous_graph is not None
            ):
                try:
                    graph_layer.save_graph(
                        graph_layer.previous_graph
                    )
                    self._logger.info(
                        "L3 graph saved successfully"
                    )
                except Exception as exc:
                    self._logger.warning(
                        "Failed to save L3 graph: %s",
                        exc,
                    )
                    success = False

            return success
        except Exception as exc:
            self._logger.error(
                "Failed to save agent state: %s",
                exc,
            )
            return False

    def load_legacy_state(
        self,
        *,
        memory: Any,
        graph_layer: Any,
    ) -> bool:
        """Load through the legacy L2/L3 persistence methods."""

        try:
            success = True
            if memory:
                if memory.load():
                    self._logger.info(
                        "L2 memory loaded: %d episodes",
                        len(memory.episodes),
                    )
                else:
                    self._logger.warning(
                        "No existing L2 memory found"
                    )
                    success = False

            if graph_layer:
                loaded_graph = graph_layer.load_graph()
                if loaded_graph is not None:
                    graph_layer.previous_graph = loaded_graph
                    self._logger.info(
                        "L3 graph loaded: %s nodes",
                        loaded_graph.num_nodes,
                    )
                else:
                    self._logger.warning(
                        "No existing L3 graph found"
                    )
                    success = False

            return success
        except Exception as exc:
            self._logger.error(
                "Failed to load agent state: %s",
                exc,
            )
            return False


__all__ = ["AgentPersistence"]
