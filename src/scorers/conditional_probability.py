from collections import Counter
from typing import List, Tuple

from .base import NgramScorer


class ConditionalProbabilityScorer(NgramScorer):
    """Scorer that calculates the conditional probability of the last token given the prefix.

    P(w_n | w_1, ..., w_{n-1}) = count(w_1, ..., w_n) / count(w_1, ..., w_{n-1})
    """

    @property
    def name(self) -> str:
        """Returns the unique identifier for this scorer.

        Returns:
            str: The string "conditional_probability".
        """
        return "conditional_probability"

    def score(
        self,
        ngram_list: List[tuple],
        ngram_freq: Counter,
        unigram_freq: Counter,
        total_tokens: int,
        top_n: int,
    ) -> List[Tuple[tuple, float]]:
        """Calculates conditional probability for each unique n-gram.

        Args:
            `ngram_list` (List[tuple]): Full list of n-grams.
            `ngram_freq` (Counter): Frequencies of n-grams.
            `unigram_freq` (Counter): Frequencies of unigrams (not used in this scorer).
            `total_tokens` (int): Total token count (not used in this scorer).
            `top_n` (int): Number of top results to return.

        Returns:
            List[Tuple[tuple, float]]: Top N n-grams sorted by probability.
        """
        prefix_freq: Counter = Counter()
        for ngram in ngram_list:
            prefix_freq[ngram[:-1]] += 1

        scored = []
        for ngram, freq in ngram_freq.items():
            prefix = ngram[:-1]
            prob = freq / prefix_freq[prefix]
            scored.append((ngram, prob))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_n]