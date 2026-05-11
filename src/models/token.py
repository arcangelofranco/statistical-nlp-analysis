class Token:
    """Represents a single linguistic token with its associated metadata.

    Attributes:
        `word` (str): The raw text of the token.
        `postag` (str): The Part-of-Speech tag of the token.
        `lemma` (str): The lemmatized form of the token.
    """

    def __init__(self, word: str, postag: str = "", lemma: str = "") -> None:
        """Initializes a Token instance.

        Args:
            `word` (str): The raw string.
            `postag` (str): The POS tag. Defaults to empty string.
            `lemma` (str): The lemma. Defaults to empty string.
        """
        self.word = word
        self.postag = postag
        self.lemma = lemma

    def __repr__(self) -> str:
        return f"Token(word={self.word}, lemma={self.lemma}, postag={self.postag})"

    def __str__(self) -> str:
        return self.word

    def __len__(self):
        return len(self.word)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Token):
            return self.word == other.word
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.word)