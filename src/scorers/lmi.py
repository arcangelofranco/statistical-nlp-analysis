import math
from collections import Counter
from typing import List, Tuple

from .base import NgramScorer


class LMIScorer(NgramScorer):
    """Scorer that calculates Local Mutual Information (LMI).

    LMI = frequency * log2( (frequency * N^{order-1}) / ∏(unigram_counts) )
    """

    @property
    def name(self) -> str:
        """Returns the unique identifier for this scorer.

        Returns:
            str: The string "lmi".
        """
        return "lmi"

    def score(
        self,
        ngram_list: List[tuple],
        ngram_freq: Counter,
        unigram_freq: Counter,
        total_tokens: int,
        top_n: int,
    ) -> List[Tuple[tuple, float]]:
        """Calculates Local Mutual Information for each unique n-gram.

        Args:
            `ngram_list` (List[tuple]): Full list of n-grams (not used in this scorer).
            `ngram_freq` (Counter): Frequencies of n-grams.
            `unigram_freq` (Counter): Frequencies of individual tokens.
            `total_tokens` (int): Total number of units in the corpus.
            `top_n` (int): Number of top results to return.

        Returns:
            List[Tuple[tuple, float]]: Top N n-grams sorted by LMI score.
        """
        n = total_tokens
        scored = []

        for ngram, ng_count in ngram_freq.items():
            denom = 1
            for token in ngram:
                denom *= unigram_freq[token]

            if denom == 0:
                continue

            order = len(ngram)
            mi = math.log2((ng_count * (n ** (order - 1))) / denom)
            local_mi = ng_count * mi
            scored.append((ngram, local_mi))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]