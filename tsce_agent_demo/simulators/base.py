"""Base class for simple simulators."""

from abc import ABC, abstractmethod


class BaseSimulator(ABC):
    """Abstract simulator interface."""

    @abstractmethod
    def run(self):
        """Run the simulation and return the result."""
        raise NotImplementedError
