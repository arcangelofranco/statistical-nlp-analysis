def format_person_profile(name: str, profile: dict) -> str:
    """Formats a linguistic profile for a single person.

    Generates a detailed summary including sentence counts, top co-occurring locations
    and persons, top nouns, temporal references (dates, months, days), and the
    best representative sentence according to the Markov model.

    Args:
        `name` (str): The name of the person.
        `profile` (dict): The profile data dictionary for this person.

    Returns:
        str: A formatted block of text.
    """
    lines = []
    lines.append(f"\t\t - {name} ({profile['sentence_count']} frasi)")
    lines.append(f"\t")

    # Top 10 Places
    lines.append(f"\t\t - Luoghi più frequenti:")
    if profile["top_locations"]:
        for loc, count in profile["top_locations"]:
            lines.append(f"\t\t\t{loc:<25} ({count})")
    else:
        lines.append(f"\t\t\t(nessun luogo trovato)")

    # Top 10 Co-occurring Conditions
    lines.append(f"")
    lines.append(f"\t\t - Persone co-occorrenti:")
    if profile["top_persons"]:
        for person, count in profile["top_persons"]:
            lines.append(f"\t\t\t{person:<25} ({count})")
    else:
        lines.append(f"\t\t\t(nessuna persona co-occorrente)")

    # Top 10 Nouns
    lines.append(f"")
    lines.append(f"\t\t - Sostantivi più frequenti:")
    if profile["top_nouns"]:
        for noun, count in profile["top_nouns"]:
            lines.append(f"\t\t\t{noun:<25} ({count})")
    else:
        lines.append(f"\t\t\t(nessun sostantivo trovato)")

    # Dates
    lines.append(f"")
    lines.append(f"\t\t - Date trovate:")
    if profile["dates"]:
        for date in profile["dates"]:
            lines.append(f"\t\t\t{date}")
    else:
        lines.append(f"\t\t\t(nessuna data trovata)")

    # Months
    lines.append(f"\t\t - Mesi trovati:")
    if profile["months"]:
        lines.append(f"\t\t\t{', '.join(profile['months'])}")
    else:
        lines.append(f"\t\t\t(nessun mese trovato)")

    # Days
    lines.append(f"\t\t - Giorni della settimana trovati:")
    if profile["days"]:
        lines.append(f"\t\t\t{', '.join(profile['days'])}")
    else:
        lines.append(f"\t\t\t(nessun giorno trovato)")

    # Markov sentence
    lines.append(f"")
    markov = profile["best_markov_sentence"]
    lines.append(f"\t\t - Frase Markov (8-12 token, prob. massima):")
    if markov["sentence"]:
        lines.append(f"\t\t\t\"{markov['sentence']}\"")
        lines.append(f"\t\t\tP = {markov['probability']:.2e}")
    else:
        lines.append(f"\t\t\t(nessuna frase trovata nel range 8-12 token)")

    lines.append(f"\t{'─' * 50}")
    return "\n".join(lines)


def format_report(corpus_name: str, report: dict) -> str:
    """Formats a complete analysis report for a corpus.

    Includes a header, the list of top 10 global person names, and detailed
    profiles for each person.

    Args:
        `corpus_name` (str): The name of the corpus.
        `report` (dict): The full report dictionary containing NER and profile results.

    Returns:
        str: The complete formatted report.
    """
    lines = []
    lines.append(f"\n{'=' * 60}")
    lines.append(f"\tCORPUS: {corpus_name}")
    lines.append(f"{'=' * 60}\n")

    # Top 10 Global Figures
    if "ner_analysis" in report:
        ner = report["ner_analysis"]
        lines.append("\t Top 10 Nomi Propri di Persona")
        for i, (name, count) in enumerate(ner["persons"], 1):
            lines.append(f"\t\t{i:>2}. {name:<25} ({count} occorrenze)")
        lines.append("")

    # Personal Profiles
    if "person_profiles" in report:
        profiles = report["person_profiles"]
        lines.append("\t Profili Per-Persona\n")
        for name in sorted(profiles.keys()):
            lines.append(format_person_profile(name, profiles[name]))
            lines.append("")

    return "\n".join(lines)

