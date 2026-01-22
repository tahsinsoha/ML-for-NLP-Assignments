# IBM Model 1: Spanish to English Translation

## Team 8
- Sabiha Tahsin Soha
- Halleluyah Brhanemesqel
- Harshith Ravi Kopparam

Implementation of IBM Model 1 statistical machine translation using the Europarl parallel corpus.

## Overview

This project implements the Expectation-Maximization (EM) algorithm for IBM Model 1 to learn word-level translation probabilities from Spanish-English parallel text.

## Requirements

- Python 3.7+
- No external dependencies (uses only standard library)

## Quick Start

```bash
./run_pipeline.sh
```

This runs the complete pipeline:
1. **Preprocessing** - Downloads Europarl corpus and tokenizes text
2. **Training** - Trains IBM Model 1 using EM algorithm
3. **Analysis** - Generates translation tables and perplexity comparisons

## Running Individual Steps

```bash
# Step 1: Preprocess data
python3 preprocessing.py

# Step 2: Train model
python3 training.py

# Step 3: Generate translation tables and perplexity analysis
python3 translation_tables.py
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
checkpoints/
├── model_iter_1.pkl    # Checkpoint after iteration 1
├── model_iter_2.pkl    # Checkpoint after iteration 2
├── ...
└── model_final.pkl     # Final trained model

preprocessed_data.pkl   # Tokenized sentence pairs
```

## Resuming Training

Training automatically resumes from the latest checkpoint:

```bash
# Change NUM_EM_ITERATIONS in config.py to a higher number
python3 training.py  # Resumes from last checkpoint
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
