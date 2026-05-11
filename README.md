<div align="center">

# Statistical NLP Analysis

**A modular, SOLID-compliant Python pipeline for deep statistical and semantic analysis of English literary corpora.**

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![NLTK](https://img.shields.io/badge/NLTK-Natural%20Language%20Toolkit-154f3c)
![Architecture](https://img.shields.io/badge/Architecture-SOLID-blueviolet)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
  - [Program 1 — Statistical & Linguistic Analysis](#program-1--statistical--linguistic-analysis)
  - [Program 2 — Named Entity Recognition & Profiling](#program-2--named-entity-recognition--profiling)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Sample Output](#sample-output)
- [License](#license)

---

## Overview

**Statistical NLP Analysis** is a pipeline-based system that processes English text corpora to extract rich statistical metrics, morphological annotations, and named entity profiles. Built around an `AnalysisPipeline` orchestrator, the system allows specialized **analyzers**, **scorers**, **preprocessors**, and **formatters** to be registered and composed independently — making it straightforward to extend, swap, or test individual components without touching the rest of the codebase.

The pipeline ships with two complementary analysis programs, demonstrated on *Bram Stoker's Dracula* and *R. L. Stevenson's Strange Case of Dr Jekyll and Mr Hyde*.

---

## Features

### Program 1 — Statistical & Linguistic Analysis

Compares corpora by extracting quantitative and morphological patterns:

| Category | Description |
|---|---|
| **Basic Statistics** | Total sentence / token counts, average sentence length (tokens), average word length (characters) |
| **Lexical Distribution** | Vocabulary size and *hapax legomena* growth over incremental 1 000-token windows |
| **Morphological Analysis** | Noun / Verb ratio across the full corpus |
| **Part-of-Speech** | Top 10 most frequent PoS tags |
| **PoS Bigrams** | Top 10 most significant PoS bigrams, ranked by **Conditional Probability** and **Local Mutual Information (LMI)** |

### Program 2 — Named Entity Recognition & Profiling

Builds a deep semantic profile for each major character in the corpus:

| Component | Description |
|---|---|
| **Entity Identification** | Top 10 most frequent person names with **entity consolidation** (merges fragmented NER chunks like *"Van"* → *"Van Helsing"*) |
| **Location Context** | Top 10 most frequent GPE entities co-occurring with each character |
| **Co-occurrence Network** | Top 10 most co-occurring persons per character |
| **Lexical Context** | Top 10 most frequent nouns in each character's sentences, with **spurious noun filtering** (removes smart quotes, abbreviations, single-character artifacts) |
| **Temporal Extraction** | Dates, months, and weekdays extracted via targeted regular expressions |
| **Markov Modelling** | Highest-probability sentence (8–12 tokens) per character, computed with a configurable **k-order Markov model** (defaults to zero-order unigram) |

---

## Project Structure

```
statistical-nlp-analysis/
│
├── main.py                                 # Entry point — configures and runs both pipelines
│
├── src/                                    # Main package
│   ├── __init__.py
│   ├── pipeline.py                         # AnalysisPipeline orchestrator (supports deferred steps)
│   │
│   ├── models/                             # Domain models and data structures
│   │   ├── corpora.py                      #   Corpora: sentence list + cached properties
│   │   ├── sentence.py                     #   Sentence: ordered token container
│   │   └── token.py                        #   Token: word + optional POS tag / lemma
│   │
│   ├── preprocessors/                      # Raw text cleaning (operates on text before tokenization)
│   │   ├── base.py                         #   TextCleaner ABC
│   │   ├── pipeline.py                     #   CleaningPipeline: sequential cleaner runner
│   │   ├── cleaners.py                     #   BOM, formatting markers, headers, whitespace
│   │   └── filters.py                      #   (deprecated — retained for history)
│   │
│   ├── analyzers/                          # Atomic, extensible analysis units
│   │   ├── base.py                         #   Analyzer ABC
│   │   ├── base_stats.py                   #   BaseStatisticsAnalyzer: counts, averages
│   │   ├── hapax.py                        #   HapaxAnalyzer: vocabulary & hapax growth
│   │   ├── pos.py                          #   POSAnalyzer: POS distribution, N/V ratio
│   │   ├── ngram.py                        #   NgramAnalyzer: n-gram extraction + scoring
│   │   ├── extractors.py                   #   UnitExtractor strategies (word / POS)
│   │   ├── ner.py                          #   NERAnalyzer: entity recognition + consolidation
│   │   ├── person_profile.py               #   PersonProfileAnalyzer: per-character profiling
│   │   └── markov.py                       #   MarkovModel: k-order sentence scoring
│   │
│   ├── scorers/                            # Interchangeable scoring strategies
│   │   ├── base.py                         #   BigramScorer ABC
│   │   ├── conditional_probability.py      #   P(w₂|w₁) scorer
│   │   └── lmi.py                          #   Local Mutual Information scorer
│   │
│   └── formatters/                         # Presentation logic (fully decoupled)
│       ├── first_program_formatter.py      #   Program 1 text report builder
│       └── second_program_formatter.py     #   Program 2 text report builder
│
└── text/
    ├── input/                              # Input corpora (UTF-8 .txt files)
    │   ├── dracula-UTF8.txt
    │   └── hydeandjack-UTF8.txt
    └── output/                             # Generated reports
        ├── output_first_program.txt
        └── output_second_programma.txt
```

---

## Installation

### Prerequisites

- **Python 3.10+**

### Setup

**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/statistical-nlp-analysis.git
cd statistical-nlp-analysis
```

**2. Create a virtual environment and install dependencies**

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install nltk
```

**3. Download the required NLTK resources**

```python
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')
```

> [!TIP]
> Alternatively, run `nltk.download('all')` to download all NLTK data at once (~3.5 GB).

---

## Usage

Place your UTF-8 encoded English text files in `text/input/`, then run:

```bash
python main.py
```

Results are written to `text/output/`:

| File | Contents |
|---|---|
| `output_first_program.txt` | Statistical, morphological, and PoS bigram analysis |
| `output_second_programma.txt` | Named entity profiling, co-occurrence networks, and Markov sentences |

### Customization

The pipeline is fully configurable from `main.py`. Examples:

```python
# Change the number of top entities to extract
NERAnalyzer(top_n=20)

# Use first-order (bigram) Markov model instead of zero-order
MarkovModel(order=1)

# Add a custom scorer to the n-gram pipeline
NgramAnalyzer(n=3, top_n=10, extractor=POSExtractor(), scorers=[LMIScorer()])

# Adjust the Markov sentence length window
PersonProfileAnalyzer(..., min_tokens=5, max_tokens=15)
```

---

## Sample Output

<details>
<summary><strong>Program 1 — Dracula (excerpt)</strong></summary>

```
============================================================
  CORPUS: dracula-UTF8.txt
============================================================

Statistiche di Base
	Frasi totali:	8525
	Token totali:	188621
	Lunghezza media frasi (token):	22.13
	Lunghezza media parole (char):	3.57

Vocabolario e Distribuzione Hapax
	Grandezza vocabolario totale:	10395
	  Porzione	 Vocabolario	   Hapax	  Rapporto
	──────────	────────────	────────	──────────
	     1,000	         415	     291	    0.2910
	     2,000	         677	     447	    0.2235
	     3,000	         914	     598	    0.1993
...

Analisi Part-of-Speech
	Sostantivi:	30886
	Verbi:	30758
	Rapporto N/V:	1.0041615189544184
	Top 10 PoS:
		    NN	  20,268
		    IN	  19,942
		   PRP	  17,736
...

Bigrammi di PoS
	Unità:	pos | Ordine:	2
	Bigrammi totali:	188,620
	Bigrammi unici:	1,095

	Top 10 per Probabilità Condizionata:
		('$', 'CD')                   	P = 1.000000
		('PDT', 'DT')                 	P = 0.906355
		('UH', ',')                   	P = 0.721154
...

	Top 10 per Local Mutual Information:
		('DT', 'NN')                  	LMI = 17170.8851
		('PRP', 'VBD')                	LMI = 13274.6187
		('IN', 'DT')                  	LMI = 12138.6481
...
```

</details>

<details>
<summary><strong>Program 2 — Dracula (excerpt)</strong></summary>

```
============================================================
  CORPUS: dracula-UTF8.txt
============================================================

	 Top 10 Nomi Propri di Persona
		 1. Van Helsing               (305 occorrenze)
		 2. Lucy                      (186 occorrenze)
		 3. Jonathan                  (144 occorrenze)
...

	 Profili Per-Persona

		 - Arthur (142 frasi)
	
		 - Luoghi più frequenti:
			Arthur                    (23)
			Telegram                  (2)
			Lucy                      (2)
...

		 - Frase Markov (8-12 token, prob. massima):
			"This time the question was by Arthur ."
			P = 3.13e-21
```

</details>

---

## License

This project is licensed under the [MIT License](LICENSE).
