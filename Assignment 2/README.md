# Assignment 2: N-gram Language Model

## Team 8
- Sabiha Tahsin Soha
- Halleluyah Brhanemesqel
- Harshith Ravi Kopparam

We pair programmed throughout this assignment, with all team members contributing equally to the code, debugging, and writeup.

## Overview

Implementation of a 4-gram language model for English, trained on the Europarl corpus. The model can:
- Estimate P(word | context) using n-gram counts
- Compute sentence probabilities
- Generate new sentences by sampling

## Corpus

**Dataset**: Europarl English monolingual corpus
- **Source**: https://www.statmt.org/europarl/v7/
- **Sentences**: ~50,000
- **Register**: Formal European Parliament proceedings

## Quick Start

```bash
./run_pipeline.sh
```

Or run individually:
```bash
python3 preprocessing.py   # Download data, tokenize, build vocabulary
python3 training.py        # Train 4-gram model
python3 evaluate.py        # Compute perplexity
python3 generate.py        # Generate sentences
```

## Model Details

- **N-gram order**: 4 (conditions on 3 previous words)
- **Smoothing**: Stupid Backoff (α = 0.4)
- **Special tokens**: `<START>`, `<END>`, `<UNK>`
- **UNK threshold**: Words appearing < 2 times → `<UNK>`

### Stupid Backoff

When an n-gram is unseen, back off to shorter context with discount:
```
S(w | w1, w2, w3) = count(w1,w2,w3,w) / count(w1,w2,w3)  if seen
                  = 0.4 × S(w | w2, w3)                   otherwise
```

## Results

### Corpus Statistics
| Metric | Value |
|--------|-------|
| Sentences | 49,945 |
| Total tokens | 1,396,479 |
| Vocabulary | 14,617 |
| 4-gram contexts | 671,649 |

### Perplexity
| Set | Perplexity |
|-----|------------|
| Train (sample) | 2.93 |
| Test | 181.78 |

### Real vs Shuffled Sentences

The model assigns higher probability to properly ordered sentences:

| Sentence | Log P |
|----------|-------|
| "we really cannot tolerate behaviour of this kind !" | -61.26 |
| "tolerate we ! kind this cannot really of behaviour" | -90.58 |
| **Difference** | **29.3 better** |

## Sample Generated Sentences

1. i do not usually speak too long .
2. this is not a european army .
3. ( applause )
4. in my opinion , a slightly short-sighted view .
5. with regard to eurodac .
6. one of the member states a transitional period of six years .
7. that is why , in our own country , greece , with its never-ending trail of destruction in its wake .
8. for that reason that a careful approach towards the control of isa is needed .
9. you cannot abolish risk because it is behaviour which can include physical attacks on individuals .
10. this is a crucial task , one which is too detailed , too extensive and need to be drawn up in collaboration with the council and parliament .

## Project Structure

```
├── config.py           # Configuration
├── preprocessing.py    # Data loading and tokenization
├── ngram_model.py      # NGramModel class
├── training.py         # Training script
├── generate.py         # Sentence generation
├── evaluate.py         # Perplexity evaluation
└── run_pipeline.sh     # Run all steps
```
