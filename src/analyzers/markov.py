from collections import Counter
from typing import List

import nltk

from src.models import Corpora


class MarkovModel:
    """A flexible Markov model for linguistic probability estimation.

    Attributes:
        `order` (int): The order of the Markov chain.
        `unigram_freq` (Counter): Frequency of single tokens.
        `ngram_freq` (Counter): Frequency of n-grams of size (order + 1).
        `context_freq` (Counter): Frequency of prefixes of size (order).
        `total` (int): Total number of tokens in the training corpus.
        `fitted` (bool): Whether the model has been trained.
    """

    def __init__(self, order: int = 0):
        """Initializes the Markov model with a specific order.

        Args:
            `order` (int): The Markov order (k). Must be >= 0. Defaults to 0.

        Raises:
            ValueError: If `order` is less than 0.
        """
        if order < 0:
            raise ValueError(f"L'ordine deve essere >= 0, ricevuto {order}")
        self.order = order
        self.unigram_freq: Counter = Counter()
        self.ngram_freq: Counter = Counter()
        self.context_freq: Counter = Counter()
        self.total: int = 0
        self.fitted: bool = False

    @property
    def get_order(self) -> int:
        """Returns the order of the Markov model.

        Returns:
            int: The Markov order.
        """
        return self.order

    @property
    def is_fitted(self) -> bool:
        """Checks if the model has been trained.

        Returns:
            bool: True if the model is fitted, False otherwise.
        """
        return self.fitted

    def fit(self, corpora: Corpora) -> "MarkovModel":
        """Trains the model on the provided corpora.

        Args:
            `corpora` (Corpora): The corpora object to train on.

        Returns:
            MarkovModel: The fitted model instance.
        """
        tokens = corpora.flat_tokens
        self.total = len(tokens)
        self.unigram_freq = corpora.token_freq

        if self.order >= 1:
            ngrams = list(nltk.ngrams(tokens, self.order + 1))
            self.ngram_freq = Counter(ngrams)
            self.context_freq = Counter(ng[:-1] for ng in ngrams)

        self.fitted = True
        return self

    def score_sentence(self, tokens: List[str]) -> float:
        """Calculates the probability of a sequence of tokens.

        Args:
            `tokens` (List[str]): The list of tokens representing a sentence.

        Returns:
            float: The calculated probability (0.0 to 1.0).

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if not self.fitted:
            raise RuntimeError(
                "Modello non addestrato. Chiamare fit(corpora) prima di score_sentence()."
            )

        if not tokens or self.total == 0:
            return 0.0

        if self.order == 0:
            return self._score_order_zero(tokens)
        else:
            return self._score_higher_order(tokens)

    def _score_order_zero(self, tokens: List[str]) -> float:
        """Calculates order-0 probability (independent tokens).

        Args:
            `tokens` (List[str]): The sentence tokens.

        Returns:
            float: The cumulative probability.
        """
        prob = 1.0
        for token in tokens:
            tf = self.unigram_freq[token]
            if tf == 0:
                return 0.0
            prob *= tf / self.total
        return prob

    def _score_higher_order(self, tokens: List[str]) -> float:
        """Calculates higher-order probability with progressive context backoff.

        For the first k tokens, where full context is unavailable, uses a
        progressively increasing context window:
            - token 0   : unigram P(w_1)
            - token 1   : bigram  P(w_2 | w_1)              [only if k >= 1]
            - token 2   : trigram P(w_3 | w_1, w_2)         [only if k >= 2]
            - ...
            - token i >= k : full k-gram P(w_i | w_{i-k}, ..., w_{i-1})

        Args:
            `tokens` (List[str]): The sentence tokens.

        Returns:
            float: The cumulative probability.
        """
        prob = 1.0

        for i, token in enumerate(tokens):
            effective_order = min(i, self.order)

            if effective_order == 0:
                tf = self.unigram_freq[token]
                if tf == 0:
                    return 0.0
                prob *= tf / self.total
            else:
                context = tuple(tokens[i - effective_order : i])
                ngram   = tuple(tokens[i - effective_order : i + 1])

                context_count = self.context_freq[context]
                ngram_count   = self.ngram_freq[ngram]

                if context_count == 0 or ngram_count == 0:
                    return 0.0

                prob *= ngram_count / context_count

        return prob