"""
Custom Exceptions for InsightSpike
==================================

Defines specific exception classes for better error handling and debugging.
"""

from typing import Any, Optional


class InsightSpikeException(Exception):
    """Base exception for all InsightSpike errors."""

    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message)
        self.message = message
        self.details = details


# DataStore related exceptions
class DataStoreError(InsightSpikeException):
    """Base exception for DataStore operations."""

    pass


class DataStoreSaveError(DataStoreError):
    """Raised when saving data fails."""

    pass


class DataStoreLoadError(DataStoreError):
    """Raised when loading data fails."""

    pass


class DataStoreNotFoundError(DataStoreError):
    """Raised when requested data is not found."""

    pass


class DataStorePermissionError(DataStoreError):
    """Raised when there are permission issues accessing storage."""

    pass


# Configuration related exceptions
class ConfigurationError(InsightSpikeException):
    """Base exception for configuration issues."""

    pass


class InvalidConfigError(ConfigurationError):
    """Raised when configuration is invalid or malformed."""

    pass


class ConfigNotFoundError(ConfigurationError):
    """Raised when configuration file is not found."""

    pass


# Agent related exceptions
class AgentError(InsightSpikeException):
    """Base exception for agent operations."""

    pass


class AgentInitializationError(AgentError):
    """Raised when agent initialization fails."""

    pass


class AgentProcessingError(AgentError):
    """Raised when agent processing encounters an error."""

    pass


# Memory related exceptions
class MemoryError(InsightSpikeException):
    """Base exception for memory operations."""

    pass


class MemoryCapacityError(MemoryError):
    """Raised when memory capacity is exceeded."""

    pass


class MemorySearchError(MemoryError):
    """Raised when memory search fails."""

    pass


# Graph related exceptions
class GraphError(InsightSpikeException):
    """Base exception for graph operations."""

    pass


class GraphBuildError(GraphError):
    """Raised when graph construction fails."""

    pass


class GraphAnalysisError(GraphError):
    """Raised when graph analysis fails."""

    pass


# LLM related exceptions
class LLMError(InsightSpikeException):
    """Base exception for LLM operations."""

    pass


class LLMConnectionError(LLMError):
    """Raised when connection to LLM service fails."""

    pass


class LLMGenerationError(LLMError):
    """Raised when LLM text generation fails."""

    pass


class LLMTokenLimitError(LLMError):
    """Raised when token limit is exceeded."""

    pass
