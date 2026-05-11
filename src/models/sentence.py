from typing import List

from .token import Token

class Sentence:
    """Represents a sentence composed of tokens.

    Provides a convenient wrapper around a list of Token objects with
    string representation and iteration support.
    """

    def __init__(self, tokens: List[Token]) -> None:
        """Initializes a Sentence instance.

        Args:
            `tokens` (List[Token]): The list of tokens forming the sentence.
        """
        self.tokens = tokens

    def __repr__(self) -> str:
        return f"Sentence(tokens={self.tokens})"

    def __str__(self) -> str:
        return " ".join(token.word for token in self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __iter__(self):
        return iter(self.tokens)