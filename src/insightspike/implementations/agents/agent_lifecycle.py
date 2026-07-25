"""Runtime lifecycle coordination for ``MainAgent`` components."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Callable


@dataclass(frozen=True)
class RuntimeInitialization:
    """Result of one runtime-initialization attempt."""

    llm: Any
    initialized: bool


class _FallbackLLM:
    """Minimal provider used when the configured provider cannot be built."""

    initialized = True

    def initialize(self) -> bool:
        return True

    def generate(self, *args: Any, **kwargs: Any) -> dict:
        return {
            "text": "[fallback-llm-response]",
            "raw": "[fallback-llm-response]",
        }

    def generate_response(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> dict:
        return {
            "response": "[fallback-llm-response]",
            "success": True,
        }

    def generate_response_detailed(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> dict:
        return {
            "response": "[fallback-llm-response]",
            "success": True,
        }


class AgentLifecycle:
    """Initialize current public component instances in a fixed order."""

    def __init__(self, *, logger: logging.Logger):
        self._logger = logger

    def resolve_llm(
        self,
        *,
        llm: Any,
        llm_factory: Callable[[], Any],
    ) -> Any:
        """Return the current or newly-created L4 provider."""

        if llm is not None:
            return llm
        try:
            return llm_factory()
        except Exception as exc:
            self._logger.warning(
                "Failed to create LLM provider (fallback mock). Error: %s",
                exc,
            )
            return _FallbackLLM()

    def initialize(
        self,
        *,
        llm: Any,
        memory: Any,
        datastore: Any,
        lite_mode: bool,
        error_monitor: Any,
    ) -> RuntimeInitialization:
        """Initialize L4, optional legacy memory, then reset L1."""

        if hasattr(llm, "initialize"):
            try:
                if not getattr(llm, "initialized", False):
                    if not llm.initialize():
                        self._logger.error(
                            "Failed to initialize LLM provider"
                        )
                        return RuntimeInitialization(
                            llm=llm,
                            initialized=False,
                        )
            except Exception as exc:
                self._logger.error(
                    "LLM initialization failed: %s",
                    exc,
                )
                raise

        if datastore is None and not lite_mode:
            if not memory.load():
                self._logger.info(
                    "No existing memory found, starting fresh"
                )
        else:
            self._logger.debug(
                "Skipping legacy L2 load "
                "(datastore present or lite/min mode)"
            )

        error_monitor.reset()
        return RuntimeInitialization(
            llm=llm,
            initialized=True,
        )


__all__ = ["AgentLifecycle", "RuntimeInitialization"]
