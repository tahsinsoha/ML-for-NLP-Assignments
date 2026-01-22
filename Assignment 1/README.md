# Project 1: MT evaluation and implementing IBM Model 1

## Team 8
- Sabiha Tahsin Soha
- Halleluyah Brhanemesqel
- Harshith Ravi Kopparam

## Part 1: Evaluating Machine Translation

See [ML for NLP Team 8 Part 1.pdf](ML%20for%20NLP%20Team%208%20Part%201.pdf)

## Part 2: Implementing IBM Model 1 (for Spanish to English Translation)

Implementation of IBM Model 1 statistical machine translation using the Europarl parallel corpus of Spanish-English.

### Overview

This project implements the Expectation-Maximization (EM) algorithm for IBM Model 1 to learn word-level translation probabilities from Spanish-English parallel text.

### Quick Start

```bash
./run_pipeline.sh
```

This runs the complete pipeline:
1. **preprocessing.py** - Downloads Europarl Spanish-English corpus and tokenizes text
2. **training.py** - Trains IBM Model 1 using EM algorithm
3. **translation_tables.py** - Generates translation tables and perplexity comparisons

### Results

- [Translation Tables](tables/translation_tables.txt) - Top 10 most common source-language(Spanish) words
- [Perplexity Comparison](tables/perplexity_comparison.txt) - Perplexity scores for Real sentence from the training data vs randomly-sampled target-language(English) sentences of the same length

### Requirements

- Python 3.7+
- No external dependencies (uses only standard library)

### Running Individual Steps

```bash
python3 preprocessing.py       # Step 1: Preprocess data
python3 training.py            # Step 2: Train model
python3 translation_tables.py  # Step 3: Generate results
```

### Configuration

Edit `config.py` to change parameters:

```python
NUM_TRAINING_PAIRS = 10000  # Number of sentence pairs
NUM_EM_ITERATIONS = 5       # EM iterations
TOP_N_SOURCE_WORDS = 10     # Source words to display
TOP_N_TRANSLATIONS = 5      # Translations per word
```

### Preprocessing

**Tokenizer:** Custom tokenizer that:
- Lowercases all text (reduces vocabulary size by merging case variants like "Hello"/"hello", and improves alignment quality with limited training data by combining sparse occurrences)
- Separates punctuation marks (.,!?;:"'()[]{}) from words by adding spaces
- Splits on whitespace
- Removes empty strings that may result from multiple consecutive spaces

Example: `"Hello, world!"` → `["hello", ",", "world", "!"]`

### Output Files

```
tables/
├── translation_tables.txt    # Translation probability tables
└── perplexity_comparison.txt # Perplexity analysis results
```

`checkpoints/` ([Google Drive](https://drive.google.com/drive/folders/1dnlo23Wedv0p1g_zPhfvPqN1TOHA0GPq?usp=sharing))

```
├── model_iter_*.pkl          # Checkpoints after each iteration
└── model_final.pkl           # Final trained model
```

```
preprocessed_data.pkl         # Tokenized sentence pairs
```

### Project Structure

```
├── config.py              # Configuration settings
├── preprocessing.py       # Data download and tokenization
├── training.py            # IBM Model 1 EM training
├── translation_tables.py  # Analysis and evaluation
├── run_pipeline.sh        # Shell script to run all steps
└── README.md
```
