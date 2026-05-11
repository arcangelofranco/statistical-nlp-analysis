from typing import Any, Dict

from src.models import Corpora
from .base import Analyzer


class BaseStatisticsAnalyzer(Analyzer):
    """Analyzer that computes basic descriptive statistics for a corpus.

    Calculates metrics such as total tokens, total sentences, average sentence
    length (in characters and tokens), and average word length.
    """

    @property
    def name(self) -> str:
        """Returns the unique identifier for this analyzer.

        Returns:
            str: The string "basic_statistics".
        """
        return "basic_statistics"

    def analyze(self, corpora: Corpora) -> Dict[str, Any]:
        """Performs statistical analysis on the provided corpora.

        Args:
            `corpora` (Corpora): The corpora object containing tokens and sentences.

        Returns:
            Dict[str, Any]: A dictionary containing the following keys:
                - `total_tokens` (int): Total number of tokens.
                - `total_sentences` (int): Total number of sentences.
                - `avg_sentence_length_chars` (float): Average characters per sentence.
                - `avg_sentence_length_tokens` (float): Average tokens per sentence.
                - `avg_word_length` (float): Average length of words in characters.
        """
        tokens = corpora.flat_tokens
        sentences = corpora.sentences

        avg_sent_chars = (
            sum(len(str(s)) for s in sentences) / len(sentences)
            if sentences else 0.0
        )
        avg_sent_tokens = (
            sum(len(s) for s in sentences) / len(sentences)
            if sentences else 0.0
        )
        avg_word_len = (
            sum(len(w) for w in tokens) / len(tokens)
            if tokens else 0.0
        )

        return {
            "total_tokens": corpora.total_tokens,
            "total_sentences": corpora.total_sentences,
            "avg_sentence_length_chars": avg_sent_chars,
            "avg_sentence_length_tokens": avg_sent_tokens,
            "avg_word_length": avg_word_len,
        }
