from typing import Any, Dict, Union

import nltk

from src.models import Corpora
from .base import Analyzer


class POSAnalyzer(Analyzer):
    """Analyzer that computes Part-of-Speech statistics.

    Calculates the distribution of POS tags, counts of nouns and verbs,
    and the noun-to-verb ratio for a given corpus.
    """

    _NOUN_TAGS = frozenset({"NN", "NNP", "NNPS", "NNS"})
    _VERB_TAGS = frozenset({"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"})

    def __init__(self, top_n: int = 10):
        """Initializes the POS analyzer.

        Args:
            `top_n` (int): Number of top frequent POS tags to return. Defaults to 10.
        """
        self.top_n = top_n

    @property
    def name(self) -> str:
        """Returns the unique identifier for this analyzer.

        Returns:
            str: The string "pos_analysis".
        """
        return "pos_analysis"

    def analyze(self, corpora: Corpora) -> Dict[str, Any]:
        """Performs POS analysis on the provided corpora.

        Args:
            `corpora` (Corpora): The corpora object to analyze.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - `noun_verb_ratio` (Union[float, str]): Ratio of nouns to verbs.
                - `noun_count` (int): Total number of nouns.
                - `verb_count` (int): Total number of verbs.
                - `pos_frequency` (List[Tuple]): Top N most frequent POS tags.
        """
        # Use the Corpora cache: O(1) if pos_tags has already been calculated
        pos_tags = corpora.pos_tags

        noun_count = sum(1 for _, tag in pos_tags if tag in self._NOUN_TAGS)
        verb_count = sum(1 for _, tag in pos_tags if tag in self._VERB_TAGS)

        ratio: Union[float, str] = (
            noun_count / verb_count if verb_count > 0 else "N/A"
        )

        tag_freq = nltk.FreqDist(tag for _, tag in pos_tags)

        return {
            "noun_verb_ratio": ratio,
            "noun_count": noun_count,
            "verb_count": verb_count,
            "pos_frequency": tag_freq.most_common(self.top_n),
        }