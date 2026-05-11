def format_report(name: str, report: dict) -> str:
    """Formats an analysis report into a human-readable text string.

    Processes various sections of the report (basic statistics, hapax distribution,
    POS analysis, and POS n-grams) and formats them with consistent indentation
    and headers.

    Args:
        `name` (str): The name of the corpus being reported.
        `report` (dict): A dictionary containing the analysis results from various analyzers.

    Returns:
        str: A formatted string ready for printing or writing to a file.
    """
    lines = [f"\n{'='*60}", f"  CORPUS: {name}", f"{'='*60}\n"]


    # Basic Statistics
    if "basic_statistics" in report:
        stats = report["basic_statistics"]
        lines.append("Statistiche di Base")
        lines.append(f"\tFrasi totali:\t{stats['total_sentences']}")
        lines.append(f"\tToken totali:\t{stats['total_tokens']}")
        lines.append(f"\tLunghezza media frasi (token):\t{stats['avg_sentence_length_tokens']:.2f}")
        lines.append(f"\tLunghezza media parole (char):\t{stats['avg_word_length']:.2f}")
        lines.append("")

    # Hapax and vocabulary
    if "hapax" in report:
        hapax = report["hapax"]
        lines.append("Vocabolario e Distribuzione Hapax")
        lines.append(f"\tGrandezza vocabolario totale:\t{hapax['vocabulary_size']}")
        lines.append(f"\t{'Porzione':>10}\t{'Vocabolario':>12}\t{'Hapax':>8}\t{'Rapporto':>10}")
        lines.append(f"\t{'─'*10}\t{'─'*12}\t{'─'*8}\t{'─'*10}")
        for entry in hapax["hapax_distribution"]:
            lines.append(
                f"\t{entry['tokens']:>10,}\t{entry['vocab_size']:>12,}\t"
                f"{entry['hapax_count']:>8,}\t{entry['hapax_ratio']:>10.4f}"
            )
        lines.append("")

    # PoS Analysis
    if "pos_analysis" in report:
        pos = report["pos_analysis"]
        lines.append("Analisi Part-of-Speech")
        lines.append(f"\tSostantivi:\t{pos['noun_count']}")
        lines.append(f"\tVerbi:\t{pos['verb_count']}")
        lines.append(f"\tRapporto N/V:\t{pos['noun_verb_ratio']}")
        lines.append("\tTop 10 PoS:")
        for tag, count in pos["pos_frequency"]:
            lines.append(f"\t\t{tag:>6}\t{count:>8,}")
        lines.append("")

    # PoS Bigrams
    if "pos_2gram_analysis" in report:
        ngram = report["pos_2gram_analysis"]
        lines.append("Bigrammi di PoS")
        lines.append(f"\tUnità:\t{ngram['unit']} | Ordine:\t{ngram['order']}")
        lines.append(f"\tBigrammi totali:\t{ngram['total_ngrams']:,}")
        lines.append(f"\tBigrammi unici:\t{ngram['unique_ngrams']:,}")

        if "conditional_probability" in ngram:
            lines.append("\n\tTop 10 per Probabilità Condizionata:")
            for bigram, prob in ngram["conditional_probability"]:
                lines.append(f"\t\t{str(bigram):<30}\tP = {prob:.6f}")

        if "lmi" in ngram:
            lines.append("\n\tTop 10 per Local Mutual Information:")
            for bigram, lmi in ngram["lmi"]:
                lines.append(f"\t\t{str(bigram):<30}\tLMI = {lmi:.4f}")

        lines.append("")

    return "\n".join(lines)
