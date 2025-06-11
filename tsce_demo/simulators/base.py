from __future__ import annotations

from abc import ABC, abstractmethod

class BaseSimulator(ABC):
    """Abstract base class for all simulators."""

    @abstractmethod
    def run(self, *args, **kwargs):
        """Run the simulator."""
        raise NotImplementedError

__all__ = ["BaseSimulator"]
