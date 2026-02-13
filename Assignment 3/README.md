# Assignment 3: Phrase Extraction for Phrase-Based Machine Translation

## Team 8
- Sabiha Tahsin Soha
- Halleluyah Brhanemesqel
- Harshith Ravi Kopparam

We pair programmed throughout this assignment, with all team members contributing equally to the code, debugging, and writeup.

## Overview

This project extends our IBM Model 1 implementation (from Assignments 1) into the beginnings of a **phrase-based statistical machine translation** system for Spanish → English.

### Quick Start

**Windows (PowerShell):**
```powershell
.\run_pipeline.ps1
```

**Linux / macOS (Bash):**
```bash
./run_pipeline.sh
```

This runs the complete pipeline:
1. **preprocessing.py** — Downloads Europarl Spanish-English corpus and tokenizes text
2. **training.py** — Trains forward & reverse IBM Model 1 using EM
3. **alignment.py** — Extracts Viterbi alignments and symmetrizes with grow-diag-final
4. **phrase_extraction.py** — Extracts phrase pairs, scores them, and displays top phrases by length

### Results

- [Phrase Table](tables/phrase_table.txt) — Scored phrase pairs with translation costs in bits
- [Top Phrases](tables/top_phrases.txt) — Top 5 most common phrases per length (5, 4, 3, 2, 1)
- [Alignments](tables/alignments.txt) — Result from grow-diag-final symmetrization of word alignments

### Requirements

- Python 3.13.2 or Python 3.12.4
- No external dependencies (uses only standard library)

### Running Individual Steps

```bash
python preprocessing.py         # Step 1: Preprocess data
python training.py              # Step 2: Train forward & reverse IBM Model 1
python alignment.py             # Step 3: Viterbi alignments + symmetrization
python phrase_extraction.py     # Step 4: Phrase extraction, scoring & top phrases by length
```

### Configuration

Edit `config.py` to change parameters:

```python
# IBM Model 1
NUM_TRAINING_PAIRS = 10000   # Sentence pairs for IBM Model 1 training
NUM_EM_ITERATIONS  = 15      # EM iterations (both directions)

# Phrase extraction
NUM_ALIGNMENT_PAIRS = 1000   # Sentence pairs for alignment / phrase extraction
MAX_PHRASE_LENGTH   = 5      # Maximum phrase length to extract
TOP_N_PHRASES       = 5      # Top-N phrases to display per length
```

### Preprocessing

**Tokenizer:** Custom tokenizer that:
- Lowercases all text
- Separates punctuation marks from words by adding spaces
- Splits on whitespace
- Removes empty strings from multiple consecutive spaces

Example: `"Hello, world!"` → `["hello", ",", "world", "!"]`

### Training

- Implements IBM Model 1 with the Expectation-Maximization algorithm
- Trains the **forward model** P(e | f) — Spanish → English
- Trains the **reverse model** P(f | e) — English → Spanish (by swapping sentence pairs)
- Both models support checkpointing and resume

### Alignment

**Viterbi alignments:** For each target word, find the source word with the highest translation probability. Extracted in both directions:
- **f→e (forward):** For each English word eⱼ, find argmax_i P(eⱼ | fᵢ)
- **e→f (reverse):** For each Spanish word fᵢ, find argmax_j P(fᵢ | eⱼ)

**Grow-diag-final symmetrization**:
1. Start with the **intersection** of the two alignments (high-precision points)
2. **Grow-diag:** Iteratively add neighboring points (including diagonals) from the union if the English or foreign word is unaligned
3. **Final:** Add remaining alignment points for any still-unaligned positions

### Phrase Extraction & Scoring

**Consistent phrase pairs:** A phrase pair (ē, f̄) is *consistent* with alignment A if every aligned word in ē has its partner inside f̄ (and vice versa), and at least one alignment point connects the two spans.

**Scoring:** Phrase pairs are scored by relative frequency in both directions, converted to **costs in bits** (−log₂ probability). Lower cost = more probable.

### Output Files

```
tables/
├── alignments.txt             # Result from grow-diag-final symmetrization of word alignments
├── alignments.pkl             # Binary version of the symmetrized word alignments loaded by phrase_extraction.py
├── phrase_table.txt           # Scored phrase pairs with translation costs in bits
└── top_phrases.txt            # Top 5 most common phrases per length (5, 4, 3, 2, 1)
```

`checkpoints/`

```
├── model_iter_*.pkl           # Forward model checkpoints
├── model_final.pkl            # Final forward model P(e|f)
├── reverse_model_iter_*.pkl   # Reverse model checkpoints
└── model_reverse_final.pkl    # Final reverse model P(f|e)
```

```
preprocessed_data.pkl          # Tokenized sentence pairs
```

### Project Structure

```
├── config.py                  # Configuration settings
├── preprocessing.py           # Data download and tokenization
├── training.py                # IBM Model 1 EM training (forward & reverse)
├── alignment.py               # Viterbi alignments + grow-diag-final symmetrization
├── phrase_extraction.py       # Phrase extraction, scoring & top phrases by length
├── run_pipeline.sh            # Bash script to run the complete pipeline (Linux/macOS)
├── run_pipeline.ps1           # PowerShell script to run the complete pipeline (Windows)
└── README.md
```

### Reproducibility

All results and output files can be reproduced by executing `./run_pipeline.sh` (Linux/macOS) or `.\run_pipeline.ps1` (Windows), or by running the individual pipeline steps as described in the **Running Individual Steps** section above.

For convenience, pre-computed results are also available for download, including:
- All files in the `tables/` directory (alignments, phrase tables, top phrases)
- All model checkpoints in the `checkpoints/` directory
- The tokenized corpus `preprocessed_data.pkl`

These files can be accessed via [Google Drive](https://drive.google.com/drive/folders/1wP_hpa1FS4uDDNV83SIFXrFphgWjxcbm?usp=sharing).
