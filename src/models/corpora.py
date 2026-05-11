from collections import Counter
from functools import cached_property
from typing import TYPE_CHECKING, List, Optional, Tuple

import nltk

from .sentence import Sentence
from .token import Token

if TYPE_CHECKING:
    from src.preprocessors import CleaningPipeline # type: ignore

class Corpora:
    """A collection of sentences representing a linguistic corpus.

    Provides high-level access to the corpus data, including lazy-loaded
    properties for tokens, frequencies, and POS tags.

    Attributes:
        `sentences` (List[Sentence]): The list of sentences in the corpus.
    """

    def __init__(self, sentences: List[Sentence]) -> None:
        """Initializes a Corpora instance.

        Args:
            `sentences` (List[Sentence]): List of Sentence objects.
        """
        self.sentences = sentences

    def __repr__(self) -> str:
        return f"Corpora(text={self.sentences}, n_sentenes={len(self.sentences)}, n_tokens={self.total_tokens})"

    def __str__(self) -> str:
        return "\n".join(str(s) for s in self.sentences)

    def __iter__(self):
        return iter(self.sentences)

    def __len__(self):
        return len(self.sentences)

    @classmethod
    def from_raw_text(cls, text: str, lang: str = "english", cleaning_pipe: Optional["CleaningPipeline"] = None) -> "Corpora":
        """Creates a Corpora instance from a raw string.

        Args:
            `text` (str): The raw text string.
            `lang` (str): Language for tokenization. Defaults to "english".
            `cleaning_pipe` (Optional[CleaningPipeline]): A pipeline to clean text before tokenization.

        Returns:
            Corpora: A new Corpora instance.
        """
        if cleaning_pipe is not None:
            text = cleaning_pipe.run(text)

        splitter = nltk.data.load(f"tokenizers/punkt/{lang}.pickle")
        raw_sentences = splitter.tokenize(text) # type: ignore
        sentences = [
            Sentence([Token(w) for w in nltk.word_tokenize(raw, language=lang)])
            for raw in raw_sentences
        ]
        return cls(sentences)

    @classmethod
    def from_file(cls, filepath: str, encoding: str = "utf-8-sig", lang: str = "english", cleaning_pipe: Optional["CleaningPipeline"] = None) -> "Corpora":
        """Creates a Corpora instance by reading from a file.

        Args:
            `filepath` (str): Path to the text file.
            `encoding` (str): File encoding. Defaults to "utf-8-sig".
            `lang` (str): Language for tokenization. Defaults to "english".
            `cleaning_pipe` (Optional[CleaningPipeline]): A pipeline to clean text.

        Returns:
            Corpora: A new Corpora instance.
        """
        with open(filepath, "r", encoding=encoding) as f:
            return cls.from_raw_text(f.read(), lang=lang, cleaning_pipe=cleaning_pipe)

    @cached_property
    def flat_tokens(self) -> List[str]:
        """Returns a flat list of all token strings in the corpus.

        Returns:
            List[str]: All tokens in the corpus.
        """
        return [tok.word for sent in self.sentences for tok in sent]

    @cached_property
    def total_tokens(self) -> int:
        """Returns the total number of tokens in the corpus.

        Returns:
            int: Token count.
        """
        return len(self.flat_tokens)

    @cached_property
    def total_sentences(self) -> int:
        """Returns the total number of sentences in the corpus.

        Returns:
            int: Sentence count.
        """
        return len(self.sentences)

    @cached_property
    def token_freq(self) -> Counter:
        """Returns a frequency distribution of all tokens.

        Returns:
            Counter: Mapping of tokens to their occurrence counts.
        """
        return Counter(self.flat_tokens)

    @cached_property
    def pos_tags(self) -> List[Tuple[str, str]]:
        """Computes Part-of-Speech tags for all tokens in the corpus.

        This property is cached because POS tagging is computationally expensive.

        Returns:
            List[Tuple[str, str]]: A list of (word, tag) tuples.
        """
        return nltk.pos_tag(self.flat_tokens)

    def get_token(self, index: int) -> Optional[Token]:
        """Retrieves a token by its global index in the corpus.

        Args:
            `index` (int): The zero-based index of the token.

        Returns:
            Optional[Token]: The Token at the specified index, or None if out of bounds.
        """
        for sentence in self.sentences:
            if index < len(sentence.tokens):
                return sentence.tokens[index]
            index -= len(sentence.tokens)
        return None

