from typing import Any, Callable, Dict, List

from src.models import Corpora
from src.analyzers.base import Analyzer


class AnalysisPipeline:
    """Orchestrates a sequence of analyzers to process a corpus.

    Supports both static analyzers and deferred (lazy) analyzers that can depend
    on the results of previously executed steps.
    """

    def __init__(self):
        """Initializes an empty analysis pipeline."""
        self.steps: List[
            Analyzer | Callable[[Dict[str, Dict[str, Any]]], Analyzer]
        ] = []

    def add(self, analyzer: Analyzer) -> "AnalysisPipeline":
        """Adds a static analyzer to the pipeline.

        Args:
            `analyzer` (Analyzer): The analyzer instance to add.

        Returns:
            AnalysisPipeline: The pipeline instance for chaining.

        Raises:
            TypeError: If the provided object is not an Analyzer.
        """
        if not isinstance(analyzer, Analyzer):
            raise TypeError(
                f"Atteso un Analyzer, ricevuto {type(analyzer).__name__}"
            )
        self.steps.append(analyzer)
        return self

    def add_deferred(
        self, factory: Callable[[Dict[str, Dict[str, Any]]], Analyzer]
    ) -> "AnalysisPipeline":
        """Adds a deferred analyzer factory to the pipeline.

        Deferred analyzers are instantiated at runtime with access to the
        results of previous steps. This is useful for analyzers that depend
        on outputs like entity sets or caches generated earlier in the pipeline.

        Args:
            `factory` (Callable): A function that takes the current report
                dictionary and returns an Analyzer instance.

        Returns:
            AnalysisPipeline: The pipeline instance for chaining.
        """
        self.steps.append(factory)
        return self

    def run(self, corpora: Corpora) -> Dict[str, Dict[str, Any]]:
        """Executes all steps in the pipeline sequentially.

        Resolves deferred analyzers and aggregates results into a single report.

        Args:
            `corpora` (Corpora): The corpus to be analyzed.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary mapping analyzer names to
                their respective analysis results.
        """
        report: Dict[str, Dict[str, Any]] = {}
        for step in self.steps:
            if isinstance(step, Analyzer):
                analyzer = step
            else:
                analyzer = step(report)
            report[analyzer.name] = analyzer.analyze(corpora)
        return report

