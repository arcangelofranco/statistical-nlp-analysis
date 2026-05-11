from abc import ABC, abstractmethod
from typing import List

from src.models import Corpora


class UnitExtractor(ABC):
    """Abstract strategy for extracting units used in n-gram calculations.

    Defines the interface for components that pull specific linguistic units
    (e.g., words, POS tags) from a corpora.
    """

    @abstractmethod
    def extract(self, corpora: Corpora) -> List[str]:
        """Extracts a list of units from the provided corpora.

        Args:
            `corpora` (Corpora): The corpora object to extract units from.

        Returns:
            List[str]: A list of extracted unit strings.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the unique name of the extractor.

        Returns:
            str: The identifier for this extractor.
        """
        pass


class WordExtractor(UnitExtractor):
    """Extractor that retrieves raw tokens from the corpora."""

    @property
    def name(self) -> str:
        """Returns the unique identifier for this extractor.

        Returns:
            str: The string "word".
        """
        return "word"

    def extract(self, corpora: Corpora) -> List[str]:
        """Extracts flat tokens from the corpora.

        Args:
            `corpora` (Corpora): The corpora object.

        Returns:
            List[str]: The list of tokens.
        """
        return corpora.flat_tokens


class POSExtractor(UnitExtractor):
    """Extractor that retrieves Part-of-Speech tags from the corpora."""

    @property
    def name(self) -> str:
        """Returns the unique identifier for this extractor.

        Returns:
            str: The string "pos".
        """
        return "pos"

    def extract(self, corpora: Corpora) -> List[str]:
        """Extracts POS tags from the corpora using cached data.

        Args:
            `corpora` (Corpora): The corpora object.

        Returns:
            List[str]: The list of POS tags.
        """
        # Use the Corpora cache: POS tagging is not recalculated
        return [tag for _, tag in corpora.pos_tags]

