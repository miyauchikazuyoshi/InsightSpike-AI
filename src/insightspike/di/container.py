"""
Dependency Injection Container
=============================

Simple but effective DI container for managing dependencies.
"""

from typing import Any, Callable, Dict, Optional, Type, TypeVar, Union
from abc import ABC, abstractmethod
import inspect
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ServiceProvider(ABC):
    """
    Abstract base for service providers.
    """
    
    @abstractmethod
    def provide(self, container: 'DIContainer') -> Any:
        """
        Provide an instance of the service.
        
        Args:
            container: DI container for resolving dependencies
            
        Returns:
            Service instance
        """
        pass


class SingletonProvider(ServiceProvider):
    """
    Provider that returns the same instance every time.
    """
    
    def __init__(self, factory: Callable[['DIContainer'], Any]):
        self.factory = factory
        self._instance = None
    
    def provide(self, container: 'DIContainer') -> Any:
        if self._instance is None:
            self._instance = self.factory(container)
        return self._instance


class FactoryProvider(ServiceProvider):
    """
    Provider that creates a new instance every time.
    """
    
    def __init__(self, factory: Callable[['DIContainer'], Any]):
        self.factory = factory
    
    def provide(self, container: 'DIContainer') -> Any:
        return self.factory(container)


class DIContainer:
    """
    Simple dependency injection container.
    
    Example:
        container = DIContainer()
        
        # Register singleton
        container.singleton(IDataStore, lambda c: FileSystemDataStore())
        
        # Register factory
        container.factory(IEmbedder, lambda c: EmbeddingManager())
        
        # Resolve
        datastore = container.resolve(IDataStore)
    """
    
    def __init__(self):
        self._providers: Dict[Any, ServiceProvider] = {}
        self._instances: Dict[Any, Any] = {}
    
    def register(self, key: Any, provider: ServiceProvider) -> None:
        """
        Register a service provider.
        
        Args:
            key: Service key (usually interface or class)
            provider: Service provider
        """
        self._providers[key] = provider
        logger.debug(f"Registered provider for {key}")
    
    def singleton(self, key: Any, factory: Callable[['DIContainer'], Any]) -> None:
        """
        Register a singleton service.
        
        Args:
            key: Service key
            factory: Factory function that takes container and returns instance
        """
        self.register(key, SingletonProvider(factory))
    
    def factory(self, key: Any, factory: Callable[['DIContainer'], Any]) -> None:
        """
        Register a factory service.
        
        Args:
            key: Service key
            factory: Factory function that takes container and returns instance
        """
        self.register(key, FactoryProvider(factory))
    
    def instance(self, key: Any, instance: Any) -> None:
        """
        Register an existing instance.
        
        Args:
            key: Service key
            instance: Pre-created instance
        """
        self._instances[key] = instance
        logger.debug(f"Registered instance for {key}")
    
    def resolve(self, key: Type[T]) -> T:
        """
        Resolve a service.
        
        Args:
            key: Service key to resolve
            
        Returns:
            Service instance
            
        Raises:
            KeyError: If service not registered
        """
        # Check for existing instance
        if key in self._instances:
            return self._instances[key]
        
        # Check for provider
        if key in self._providers:
            provider = self._providers[key]
            return provider.provide(self)
        
        # Try to resolve by name if key is a type
        if inspect.isclass(key):
            type_name = key.__name__
            for registered_key in self._providers:
                if hasattr(registered_key, '__name__') and registered_key.__name__ == type_name:
                    return self._providers[registered_key].provide(self)
        
        raise KeyError(f"No provider registered for {key}")
    
    def has(self, key: Any) -> bool:
        """
        Check if a service is registered.
        
        Args:
            key: Service key
            
        Returns:
            True if registered
        """
        return key in self._providers or key in self._instances
    
    def clear(self) -> None:
        """Clear all registrations."""
        self._providers.clear()
        self._instances.clear()
    
    def create_child(self) -> 'DIContainer':
        """
        Create a child container that inherits from this one.
        
        Returns:
            Child container
        """
        child = DIContainer()
        # Copy providers (not instances, as child should have its own)
        child._providers = self._providers.copy()
        return child