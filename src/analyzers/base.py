from abc import ABC, abstractmethod
from typing import Any, Dict

from src.models import Corpora


class Analyzer(ABC):
    """Abstract base class for all corpus analyzers.

    Defines the interface for components that extract metrics and insights
    from a collection of linguistic data.
    """

    @abstractmethod
    def analyze(self, corpora: Corpora) -> Dict[str, Any]:
        """Performs analysis on the provided corpora.

        Args:
            `corpora` (Corpora): The corpora object to be analyzed.

        Returns:
            Dict[str, Any]: A dictionary containing the analysis results.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the unique name of the analyzer.

        Returns:
            str: The identifier for this analyzer.
        """
        pass