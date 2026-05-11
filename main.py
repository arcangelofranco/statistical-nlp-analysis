"""Main entry point for the Statistical NLP Analysis project.

This script orchestrates two main analysis programs:
1. Program 1: Basic statistics, Hapax distribution, POS analysis, and POS bigram scoring.
2. Program 2: Named Entity Recognition (NER) and character profiling using Markov models.

The script performs the following steps:
- Initializes a cleaning pipeline to normalize input text.
- Loads text files from the 'text/input' directory.
- Executes Program 1 pipeline and saves results to 'text/output/output_first_program.txt'.
- Executes Program 2 pipeline (including deferred character profiling) and saves
  results to 'text/output/output_second_programma.txt'.
"""

import sys
import io
import os
import glob

from src.formatters import first_program_formatter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

if __name__ == "__main__":
    from src.models import Corpora
    from src.pipeline import AnalysisPipeline

    from src.preprocessors import (
        CleaningPipeline,
        BOMCleaner,
        FormattingMarkerCleaner,
        SectionSeparatorCleaner,
        HeaderCleaner,
        WhitespaceNormalizer,
    )

    from src.analyzers import (
        BaseStatisticsAnalyzer,
        HapaxAnalyzer,
        POSAnalyzer,
        NgramAnalyzer,
        NERAnalyzer,
        MarkovModel,
        PersonProfileAnalyzer,
        POSExtractor,
    )

    from src.scorers import ConditionalProbabilityScorer, LMIScorer
    from src.formatters import second_program_formatter

    cleaning = CleaningPipeline()

    cleaning.add(BOMCleaner())
    cleaning.add(FormattingMarkerCleaner())
    cleaning.add(SectionSeparatorCleaner())
    cleaning.add(HeaderCleaner())
    cleaning.add(WhitespaceNormalizer())

    input_dir = "text/input"
    input_files = sorted(glob.glob(os.path.join(input_dir, "*.txt")))
    
    if not input_files:
        print(f"Nessun file .txt trovato in {input_dir}")
        sys.exit(1)

    corpora_data = []
    
    for file_path in input_files:
        title = os.path.basename(file_path)
        corpus = Corpora.from_file(file_path, cleaning_pipe=cleaning)
        corpora_data.append((title, corpus))

    first_program_pipeline = AnalysisPipeline()
    first_program_pipeline.add(BaseStatisticsAnalyzer())
    first_program_pipeline.add(HapaxAnalyzer())
    first_program_pipeline.add(POSAnalyzer(top_n=10))
    first_program_pipeline.add(NgramAnalyzer(
        n=2,
        top_n=10,
        extractor=POSExtractor(),
        scorers=[ConditionalProbabilityScorer(), LMIScorer()],
    ))

    first_program_output = []
    for title, corpus in corpora_data:
        report = first_program_pipeline.run(corpus)
        first_program_output.append(first_program_formatter.format_report(title, report))
        
    first_program_results = "\n".join(first_program_output)

    with open("text/output/output_first_program.txt", "w", encoding="utf-8") as f:
        f.write(first_program_results)


    markov_model = MarkovModel(order=0)
    second_program_pipeline = AnalysisPipeline()
    second_program_pipeline.add(NERAnalyzer(top_n=10))
    second_program_pipeline.add_deferred(
        lambda report: PersonProfileAnalyzer(
            person_names=report["ner_analysis"]["person_set"],
            sentence_cache=report["ner_analysis"]["sentence_cache"],
            top_n=10,
            min_tokens=8,
            max_tokens=12,
            markov_model=markov_model,
        )
    )

    second_program_output = []
    for title, corpus in corpora_data:
        report = second_program_pipeline.run(corpus)
        second_program_output.append(second_program_formatter.format_report(title, report))

    second_program_results = "\n".join(second_program_output)

    with open("text/output/output_second_programma.txt", "w", encoding="utf-8") as f:
        f.write(second_program_results)