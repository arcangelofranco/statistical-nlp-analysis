from collections import Counter
from typing import Any, Dict, List, Optional

import nltk

from src.models import Corpora
from src.scorers import NgramScorer
from .base import Analyzer
from .extractors import UnitExtractor, WordExtractor


class NgramAnalyzer(Analyzer):
    """Analyzer that computes n-gram frequencies and scoring metrics.

    Flexible analyzer that can use different unit extractors (words, POS tags)
    and multiple scoring strategies (e.g., LMI, T-Score).
    """

    def __init__(
        self,
        n: int = 2,
        top_n: int = 10,
        scorers: Optional[List[NgramScorer]] = None,
        extractor: Optional[UnitExtractor] = None,
    ):
        """Initializes the NgramAnalyzer.

        Args:
            `n` (int): The order of the n-grams (e.g., 2 for bigrams). Defaults to 2.
            `top_n` (int): Number of top results to return. Defaults to 10.
            `scorers` (Optional[List[NgramScorer]]): List of scoring strategies to apply.
            `extractor` (Optional[UnitExtractor]): Strategy to extract units from corpora.
                Defaults to WordExtractor if None.

        Raises:
            ValueError: If `n` is less than 2.
        """
        if n < 2:
            raise ValueError(f"L'ordine n deve essere >= 2, ricevuto {n}")
        self.n = n
        self.top_n = top_n
        self.scorers = scorers or []
        self.extractor = extractor or WordExtractor()

    @property
    def name(self) -> str:
        """Returns the unique identifier for this analyzer.

        Returns:
            str: Identifier based on extractor name and n-gram order.
        """
        return f"{self.extractor.name}_{self.n}gram_analysis"

    def analyze(self, corpora: Corpora) -> Dict[str, Any]:
        """Performs n-gram analysis and scoring on the provided corpora.

        Args:
            `corpora` (Corpora): The corpora object to analyze.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - `order` (int): The n-gram order.
                - `unit` (str): The name of the extraction unit.
                - `total_ngrams` (int): Total count of n-grams found.
                - `unique_ngrams` (int): Count of unique n-grams.
                - `most_common` (List[Tuple]): Most frequent n-grams.
                - Additional keys for each provided scorer.
        """
        units = self.extractor.extract(corpora)
        ngram_list = list(nltk.ngrams(units, self.n))
        ngram_freq = Counter(ngram_list)

        results: Dict[str, Any] = {
            "order": self.n,
            "unit": self.extractor.name,
            "total_ngrams": len(ngram_list),
            "unique_ngrams": len(ngram_freq),
            "most_common": ngram_freq.most_common(self.top_n),
        }

        # Calculate unigram frequencies for the extracted unit
        unigram_freq = Counter(units)

        for scorer in self.scorers:
            results[scorer.name] = scorer.score(
                ngram_list=ngram_list,
                ngram_freq=ngram_freq,
                unigram_freq=unigram_freq,
                total_tokens=len(units),
                top_n=self.top_n,
            )

        return results

