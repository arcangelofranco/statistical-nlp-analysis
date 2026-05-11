from collections import Counter
from abc import ABC, abstractmethod
from typing import List, Tuple

class NgramScorer(ABC):
    """Abstract base class for n-gram scoring strategies.

    Defines the interface for components that calculate statistical association
    measures for n-grams.
    """

    @abstractmethod
    def score(
        self,
        ngram_list: List[tuple],
        ngram_freq: Counter,
        unigram_freq: Counter,
        total_tokens: int,
        top_n: int,
    ) -> List[Tuple[tuple, float]]:
        """Calculates scores for n-grams based on frequency data.

        Args:
            `ngram_list` (List[tuple]): The full list of n-grams in the corpus.
            `ngram_freq` (Counter): Frequency distribution of n-grams.
            `unigram_freq` (Counter): Frequency distribution of individual units.
            `total_tokens` (int): Total number of units in the corpus.
            `top_n` (int): Number of top-scoring n-grams to return.

        Returns:
            List[Tuple[tuple, float]]: A list of tuples containing the n-gram and its score.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Returns the unique name of the scoring strategy.

        Returns:
            str: The scorer's identifier.
        """
        pass

