# IBM Model 1: Spanish to English Translation

## Team 8
- Sabiha Tahsin Soha
- Halleluyah Brhanemesqel
- Harshith Ravi Kopparam

Implementation of IBM Model 1 statistical machine translation using the Europarl parallel corpus of Spanish-English.

## Overview

This project implements the Expectation-Maximization (EM) algorithm for IBM Model 1 to learn word-level translation probabilities from Spanish-English parallel text.

## Quick Start

```bash
./run_pipeline.sh
```

This runs the complete pipeline:
1. **Preprocessing** - Downloads Europarl corpus and tokenizes text
2. **Training** - Trains IBM Model 1 using EM algorithm
3. **Analysis** - Generates translation tables and perplexity comparisons

## Results

- [Translation Tables](tables/translation_tables.txt) - Top translations for most common Spanish words
- [Perplexity Comparison](tables/perplexity_comparison.txt) - Real vs random translation analysis

## Requirements

- Python 3.7+
- No external dependencies (uses only standard library)

## Running Individual Steps

```bash
python3 preprocessing.py       # Step 1: Preprocess data
python3 training.py            # Step 2: Train model
python3 translation_tables.py  # Step 3: Generate results
```

## Configuration

Edit `config.py` to change parameters:

```python
NUM_TRAINING_PAIRS = 10000  # Number of sentence pairs
NUM_EM_ITERATIONS = 5       # EM iterations
TOP_N_SOURCE_WORDS = 10     # Source words to display
TOP_N_TRANSLATIONS = 5      # Translations per word
```

## Preprocessing

**Tokenizer:** Custom whitespace + punctuation tokenizer
- Lowercasing: Yes (reduces vocabulary size, improves alignment with limited data)
- Punctuation: Separated from words

## Output Files

```
tables/
├── translation_tables.txt    # Translation probability tables
└── perplexity_comparison.txt # Perplexity analysis results

checkpoints/
├── model_iter_*.pkl          # Checkpoints after each iteration
└── model_final.pkl           # Final trained model

preprocessed_data.pkl         # Tokenized sentence pairs
```

## Project Structure

```
├── config.py              # Configuration settings
├── preprocessing.py       # Data download and tokenization
├── training.py            # IBM Model 1 EM training
├── translation_tables.py  # Analysis and evaluation
├── run_pipeline.sh        # Shell script to run all steps
└── README.md
```
