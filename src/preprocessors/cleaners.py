import re
from .base import TextCleaner

class BOMCleaner(TextCleaner):
    """Cleaner that removes the Byte Order Mark (BOM) from the beginning of a string."""

    @property
    def name(self) -> str:
        """Returns the identifier for this cleaner.

        Returns:
            str: The string "BOM Cleaner".
        """
        return "BOM Cleaner"

    def clean(self, text: str) -> str:
        """Removes the BOM character from the start of the text.

        Args:
            `text` (str): The input text.

        Returns:
            str: Text without the leading BOM.
        """
        return text.lstrip("\ufeff")


class FormattingMarkerCleaner(TextCleaner):
    """Cleaner that removes underscores used for formatting (e.g., italics)."""

    @property
    def name(self) -> str:
        """Returns the identifier for this cleaner.

        Returns:
            str: The string "FormattingMarker Cleaner".
        """
        return "FormattingMarker Cleaner"

    def clean(self, text: str) -> str:
        """Removes underscores surrounding words.

        Args:
            `text` (str): The input text.

        Returns:
            str: Text with underscores removed.
        """
        text = re.sub(r'(?<!\w)_([^_\n]+?)_(?!\w)', r'\1', text)
        return text


class SectionSeparatorCleaner(TextCleaner):
    """Cleaner that removes section separators composed of multiple asterisks."""

    @property
    def name(self) -> str:
        """Returns the identifier for this cleaner.

        Returns:
            str: The string "SectionSeparator Cleaner".
        """
        return "SectionSeparator Cleaner"

    def clean(self, text: str) -> str:
        """Removes asterisk-based separators.

        Args:
            `text` (str): The input text.

        Returns:
            str: Text without separators.
        """
        return re.sub(r'^\s*(\*\s+){2,}\*\s*$', '', text, flags=re.MULTILINE)


class HeaderCleaner(TextCleaner):
    """Cleaner that removes lines composed entirely of uppercase letters."""

    @property
    def name(self) -> str:
        """Returns the identifier for this cleaner.

        Returns:
            str: The string "HeaderCleaner".
        """
        return "HeaderCleaner"

    def clean(self, text: str) -> str:
        """Removes uppercase header lines.

        Args:
            `text` (str): The input text.

        Returns:
            str: Text without uppercase headers.
        """
        return re.sub(
            r'^[A-Z][A-Z\s\d\'\.\-\,IVXLCDM]*$',
            '',
            text,
            flags=re.MULTILINE
        )


class WhitespaceNormalizer(TextCleaner):
    """Cleaner that normalizes whitespace and empty lines."""

    @property
    def name(self) -> str:
        """Returns the identifier for this cleaner.

        Returns:
            str: The string "WhiteSpace Normalizer".
        """
        return "WhiteSpace Normalizer"

    def clean(self, text: str) -> str:
        """Collapses spaces and limits consecutive newlines.

        Args:
            `text` (str): The input text.

        Returns:
            str: Normalized text.
        """
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        return text.strip()

