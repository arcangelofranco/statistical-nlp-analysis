from abc import ABC, abstractmethod

class TextCleaner(ABC):
    """Abstract base class for text cleaning strategies.

    Defines the interface for components that perform specific text normalization
    or noise reduction tasks.
    """

    @abstractmethod
    def clean(self, text: str) -> str:
        """Applies a cleaning transformation to the provided text.

        Args:
            `text` (str): The raw text string to be cleaned.

        Returns:
            str: The cleaned text string.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the unique name of the cleaner.

        Returns:
            str: The cleaner's identifier.
        """
        pass

    