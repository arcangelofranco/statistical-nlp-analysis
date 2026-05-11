import re
from typing import Any, Dict, List, NamedTuple

import nltk
from nltk import FreqDist

from src.models import Corpora, Sentence
from .base import Analyzer


class SentenceNERResult(NamedTuple):
    """Container for the NER results of a single sentence.

    Attributes:
        `sentence` (Sentence): The original sentence object.
        `text` (str): The string representation of the sentence.
        `persons` (List[str]): List of person names found in the sentence.
        `locations` (List[str]): List of geographic entities (GPE) found.
        `nouns` (List[str]): List of relevant nouns found in the sentence.
    """
    sentence: Sentence
    text: str
    persons: List[str]
    locations: List[str]
    nouns: List[str]


class NERAnalyzer(Analyzer):
    """Analyzer that performs Named Entity Recognition (NER) and entity consolidation.

    Extracts persons, locations (GPE), and relevant nouns from the corpora.
    It includes logic to merge fragmented entity names (e.g., "Van" -> "Van Helsing")
    and filters out common tokenization artifacts.

    Attributes:
        `top_n` (int): Number of top frequent entities to return.
    """
    _NOUN_TAGS = frozenset({"NN", "NNS", "NNP", "NNPS"})

    # False nouns: typographical punctuation, individual characters,
    # honorific abbreviations that NLTK incorrectly tags as NN.
    _SPURIOUS_NOUN_RE = re.compile(
        r'^[\u2018\u2019\u201C\u201D\'\"]+'   # smart quotes / quotes
        r'|^[a-zA-Z]$'                        # single character (e.g., "s", "D")
        r'|^[A-Z]\.$'                         # single-letter abbreviation (e.g., "M.", "D.")
        r'|^(?:Dr|Mr|Mrs|Ms|Prof|St|Rev|etc)\.$'  # Honorific abbreviations
    )

    def __init__(self, top_n: int = 10):
        """Initializes the NER analyzer.

        Args:
            `top_n` (int): Number of top frequent entities to track. Defaults to 10.
        """
        self.top_n = top_n

    @property
    def name(self) -> str:
        """Returns the unique identifier for this analyzer.

        Returns:
            str: The string "ner_analysis".
        """
        return "ner_analysis"

    @staticmethod
    def _build_merge_map(raw_freq: FreqDist) -> Dict[str, str]:
        """Builds a short-to-long mapping for fragmented entities.

        If a short entity (e.g., "Van", "Mr.") is a fragment of a longer entity,
        it is remapped to the longer form. Merging only occurs if the cumulative
        frequency of the longer forms is greater than or equal to the frequency
        of the short form, suggesting the latter is a chunking artifact.

        Args:
            `raw_freq` (FreqDist): Frequency distribution of raw extracted entities.

        Returns:
            Dict[str, str]: A mapping from fragment strings to consolidated strings.
        """
        names = list(raw_freq.keys())
        merge: Dict[str, str] = {}

        for short in names:
            # We are only looking for entities that are prefixes of another
            candidates = [
                long for long in names
                if long != short
                and long.startswith(short)
                and (len(long) > len(short))
                and (long[len(short)] == " ")  # word-boundary break
            ]
            if not candidates:
                continue

            # Merge only if the total number of long forms is greater than or equal to the number of short forms,
            # indicating that the short form is a fragment, not a standalone name.
            total_long = sum(raw_freq[c] for c in candidates)
            if total_long < raw_freq[short]:
                continue

            best = max(candidates, key=lambda c: raw_freq[c])
            merge[short] = best

        return merge

    @staticmethod
    def _apply_merge(entities: List[str], merge_map: Dict[str, str]) -> List[str]:
        """Applies the merge mapping to a list of entity strings.

        Args:
            `entities` (List[str]): List of entity names to process.
            `merge_map` (Dict[str, str]): Mapping of fragments to consolidated names.

        Returns:
            List[str]: The list of processed entity names.
        """
        return [merge_map.get(e, e) for e in entities]

    def _is_spurious_noun(self, word: str) -> bool:
        """Checks if a noun is a tokenization artifact or honorific abbreviation.

        Args:
            `word` (str): The word to validate.

        Returns:
            bool: True if the word is considered spurious, False otherwise.
        """
        return bool(self._SPURIOUS_NOUN_RE.match(word))

    def analyze(self, corpora: Corpora) -> Dict[str, Any]:
        """Performs NER analysis on the provided corpora.

        Extracts entities, consolidates fragmented names, and generates a
        per-sentence cache for downstream consumption.

        Args:
            `corpora` (Corpora): The corpora object to analyze.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - `persons` (List[Tuple[str, int]]): Top frequent person names.
                - `gpe` (List[Tuple[str, int]]): Top frequent geographic entities.
                - `person_set` (Set[str]): Names of the top persons.
                - `sentence_cache` (List[SentenceNERResult]): Cache of results per sentence.
        """
        sentences = corpora.sentences

        # Batch processing
        word_lists = [
            [tok.word for tok in sentence]
            for sentence in sentences
        ]

        # POS tagging batch
        tagged_sents = nltk.pos_tag_sents(word_lists)

        # NE chunking batch
        chunked_sents = nltk.ne_chunk_sents(tagged_sents, binary=False)

        all_persons: List[str] = []
        all_gpe: List[str] = []
        raw_cache: List[SentenceNERResult] = []

        for sentence, _, tree in zip(sentences, tagged_sents, chunked_sents):
            text = str(sentence)
            persons: List[str] = []
            locations: List[str] = []
            nouns: List[str] = []

            for node in tree:
                if isinstance(node, nltk.Tree):
                    entity = " ".join(leaf[0] for leaf in node.leaves())
                    if not entity or not entity[0].isalpha() or not entity[0].isupper():
                        continue
                    if node.label() == "PERSON":
                        persons.append(entity)
                    elif node.label() == "GPE":
                        locations.append(entity)
                else:
                    word, tag = node
                    if tag in self._NOUN_TAGS and not self._is_spurious_noun(word):
                        nouns.append(word)

            all_persons.extend(persons)
            all_gpe.extend(locations)
            raw_cache.append(SentenceNERResult(
                sentence=sentence,
                text=text,
                persons=persons,
                locations=locations,
                nouns=nouns,
            ))

        person_merge = self._build_merge_map(FreqDist(all_persons))
        all_persons = self._apply_merge(all_persons, person_merge)

        # Rebuild the cache with the merged entities
        sentence_cache: List[SentenceNERResult] = []
        for sc in raw_cache:
            sentence_cache.append(SentenceNERResult(
                sentence=sc.sentence,
                text=sc.text,
                persons=self._apply_merge(sc.persons, person_merge),
                locations=sc.locations,
                nouns=sc.nouns,
            ))

        person_freq = FreqDist(all_persons).most_common(self.top_n)
        person_set = {name for name, _ in person_freq}

        gpe_filtered = [g for g in all_gpe if g not in person_set]
        gpe_freq = FreqDist(gpe_filtered).most_common(self.top_n)

        return {
            "persons": person_freq,
            "gpe": gpe_freq,
            "person_set": person_set,
            # Cache for PersonProfileAnalyzer
            "sentence_cache": sentence_cache,
        }
