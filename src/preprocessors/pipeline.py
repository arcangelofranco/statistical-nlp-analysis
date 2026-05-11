from typing import List

from .base import TextCleaner


class CleaningPipeline:
    """Orchestrates a sequence of text cleaners to normalize raw text.

    Allows for the sequential application of multiple cleaning strategies
    (e.g., BOM removal, whitespace normalization).
    """

    def __init__(self):
        """Initializes an empty cleaning pipeline."""
        self.cleaners: List[TextCleaner] = []

    def add(self, cleaner: TextCleaner) -> "CleaningPipeline":
        """Adds a text cleaner to the pipeline.

        Args:
            `cleaner` (TextCleaner): The cleaner instance to add.

        Returns:
            CleaningPipeline: The pipeline instance for chaining.

        Raises:
            TypeError: If the provided object is not a TextCleaner.
        """
        if not isinstance(cleaner, TextCleaner):
            raise TypeError(
                f"Atteso un TextCleaner, ricevuto {type(cleaner).__name__}"
            )
        self.cleaners.append(cleaner)
        return self

    def run(self, text: str) -> str:
        """Executes all cleaners in the pipeline sequentially on the input text.

        Args:
            `text` (str): The raw input text.

        Returns:
            str: The fully cleaned and normalized text.
        """
        for cleaner in self.cleaners:
            text = cleaner.clean(text)
        return text

