"""Abstract base interface for Speech-to-Text (STT) engines."""

from abc import ABC, abstractmethod


class BaseSTTEngine(ABC):
    """Abstract base class for all STT engine implementations."""

    @abstractmethod
    async def run(self) -> None:
        """Run the STT engine processing loop."""
        pass
