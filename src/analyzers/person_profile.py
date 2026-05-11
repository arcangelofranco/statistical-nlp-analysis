import re
from typing import Any, Dict, List, Optional, Set

from nltk import FreqDist

from src.models import Corpora
from .base import Analyzer
from .markov import MarkovModel
from .ner import SentenceNERResult


class PersonProfileAnalyzer(Analyzer):
    """Analyzer that builds detailed linguistic profiles for specific characters.

    Aggregates co-occurring entities (locations, persons), common nouns,
    temporal references (dates, months, days), and identifies the "best" representative
    sentence based on a Markov probability model.
    """

    _DATE_RE = re.compile(
        r'\b(?:'
        r'\d{1,2}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{2,4}'
        r'|\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?'
        r'(?:January|February|March|April|May|June'
        r'|July|August|September|October|November|December)'
        r'(?:\s*,?\s*\d{2,4})?'
        r'|(?:January|February|March|April|May|June'
        r'|July|August|September|October|November|December)'
        r'\s+\d{1,2}(?:st|nd|rd|th)?'
        r'(?:\s*,?\s*\d{2,4})?'
        r')\b',
        re.IGNORECASE,
    )

    _MONTH_RE = re.compile(
        r'\b(January|February|March|April|May|June'
        r'|July|August|September|October|November|December)\b',
        re.IGNORECASE,
    )

    _DAY_RE = re.compile(
        r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
        re.IGNORECASE,
    )

    def __init__(
        self,
        person_names: Set[str],
        sentence_cache: List[SentenceNERResult],
        top_n: int = 10,
        min_tokens: int = 8,
        max_tokens: int = 12,
        markov_model: Optional[MarkovModel] = None,
    ):
        """Initializes the PersonProfileAnalyzer.

        Args:
            `person_names` (Set[str]): The set of names to build profiles for.
            `sentence_cache` (List[SentenceNERResult]): Pre-computed NER results
                for each sentence in the corpus.
            `top_n` (int): Number of top frequent co-occurring entities to return.
                Defaults to 10.
            `min_tokens` (int): Minimum tokens for a sentence to be considered
                for Markov selection. Defaults to 8.
            `max_tokens` (int): Maximum tokens for a sentence to be considered
                for Markov selection. Defaults to 12.
            `markov_model` (Optional[MarkovModel]): Model used to score sentence
                probability. Defaults to order-0 MarkovModel if None.
        """
        self.person_names = person_names
        self.sentence_cache = sentence_cache
        self.top_n = top_n
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.markov_model = markov_model or MarkovModel(order=0)

    @property
    def name(self) -> str:
        """Returns the unique identifier for this analyzer.

        Returns:
            str: The string "person_profiles".
        """
        return "person_profiles"

    def analyze(self, corpora: Corpora) -> Dict[str, Any]:
        """Generates profiles for all configured persons.

        Args:
            `corpora` (Corpora): The corpora object used to train the Markov model.

        Returns:
            Dict[str, Dict[str, Any]]: A mapping of person names to their profiles.
        """
        self.markov_model.fit(corpora)

        profiles: Dict[str, Dict[str, Any]] = {}

        for person_name in sorted(self.person_names):
            # Filter phrases that contain this name
            person_cache = [
                sc for sc in self.sentence_cache
                if person_name in sc.text
            ]

            profiles[person_name] = self._build_profile(
                person_name, person_cache
            )

        return profiles

    def _build_profile(
        self,
        person_name: str,
        person_cache: List[SentenceNERResult],
    ) -> Dict[str, Any]:
        """Constructs a comprehensive linguistic profile for a single person.

        Args:
            `person_name` (str): The name of the person.
            `person_cache` (List[SentenceNERResult]): Sentences containing this person.

        Returns:
            Dict[str, Any]: Profile containing sentence counts, top co-entities,
                nouns, dates, and the best Markov sentence.
        """
        all_locations: List[str] = []
        all_persons: List[str] = []
        all_nouns: List[str] = []
        all_sentences_text: List[str] = []
        all_dates: List[str] = []
        all_months: List[str] = []
        all_days: List[str] = []

        for sc in person_cache:
            all_sentences_text.append(sc.text)
            all_locations.extend(sc.locations)
            # Co-occurring cases
            all_persons.extend(p for p in sc.persons if p != person_name)
            all_nouns.extend(sc.nouns)
            # Regular expression on sentence text
            all_dates.extend(self._DATE_RE.findall(sc.text))
            all_months.extend(
                m.capitalize() for m in self._MONTH_RE.findall(sc.text)
            )
            all_days.extend(
                d.capitalize() for d in self._DAY_RE.findall(sc.text)
            )

        best_markov = self._find_best_markov(person_cache)

        return {
            "sentence_count": len(person_cache),
            "sentences": all_sentences_text,
            "top_locations": FreqDist(all_locations).most_common(self.top_n),
            "top_persons": FreqDist(all_persons).most_common(self.top_n),
            "top_nouns": FreqDist(all_nouns).most_common(self.top_n),
            "dates": sorted(set(d.strip() for d in all_dates)),
            "months": sorted(set(all_months)),
            "days": sorted(set(all_days)),
            "best_markov_sentence": best_markov,
        }

    def _find_best_markov(
        self,
        person_cache: List[SentenceNERResult],
    ) -> Dict[str, Any]:
        """Identifies the sentence with the highest probability according to the Markov model.

        Args:
            `person_cache` (List[SentenceNERResult]): Sentences to evaluate.

        Returns:
            Dict[str, Any]: A dict with the `sentence`, its `probability`,
                and the `markov_order` used.
        """
        best_prob = 0.0
        best_sentence: Optional[str] = None

        for sc in person_cache:
            length = len(sc.sentence)
            if not (self.min_tokens <= length <= self.max_tokens):
                continue

            # Use the Markov model to calculate the probability
            tokens = [token.word for token in sc.sentence]
            prob = self.markov_model.score_sentence(tokens)

            if prob > best_prob:
                best_prob = prob
                best_sentence = sc.text

        return {
            "sentence": best_sentence,
            "probability": best_prob,
            "markov_order": self.markov_model.order,
        }

