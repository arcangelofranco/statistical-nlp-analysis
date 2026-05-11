from collections import Counter
from typing import Any, Dict

from src.models import Corpora
from .base import Analyzer


class HapaxAnalyzer(Analyzer):
    """Analyzer that calculates the distribution of Hapax Legomena in a corpus.

    Computes how the vocabulary size and the number of words occurring only once
    (hapax) evolve as the corpus size increases in fixed increments.
    """

    def __init__(self, step: int = 1000):
        """Initializes the HapaxAnalyzer with a specific increment step.

        Args:
            `step` (int): The number of tokens to add in each increment of the
                distribution analysis. Defaults to 1000.
        """
        self._step = step

    @property
    def name(self) -> str:
        """Returns the unique identifier for this analyzer.

        Returns:
            str: The string "hapax".
        """
        return "hapax"

    def analyze(self, corpora: Corpora) -> Dict[str, Any]:
        """Performs hapax distribution analysis on the provided corpora.

        Args:
            `corpora` (Corpora): The corpora object to analyze.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - `vocabulary_size` (int): Total unique tokens in the full corpus.
                - `hapax_distribution` (List[Dict]): A list of dicts for each increment,
                  each containing `tokens`, `vocab_size`, `hapax_count`, and `hapax_ratio`.
        """
        tokens = corpora.flat_tokens
        total = len(tokens)

        increments = list(range(self._step, total + 1, self._step))
        if total not in increments:
            increments.append(total)

        distribution = []
        for size in increments:
            subset = tokens[:size]
            sub_freq = Counter(subset)
            vocab_size = len(sub_freq)
            n_hapax = sum(1 for c in sub_freq.values() if c == 1)
            distribution.append({
                "tokens": size,
                "vocab_size": vocab_size,
                "hapax_count": n_hapax,
                "hapax_ratio": n_hapax / len(subset),
            })

        return {
            "vocabulary_size": len(corpora.token_freq),
            "hapax_distribution": distribution,
        }