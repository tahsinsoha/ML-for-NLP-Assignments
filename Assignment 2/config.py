#!/usr/bin/env python3
"""Configuration for N-gram Language Model"""

# N-gram order (3 = trigram, 4 = 4-gram, etc.)
NGRAM_ORDER = 4

# Number of sentences to use for training (more = better model)
NUM_TRAINING_SENTENCES = 50000

# Unknown word threshold - words appearing fewer times become <UNK>
UNK_THRESHOLD = 2

# Stupid Backoff parameter (typically 0.4)
BACKOFF_ALPHA = 0.4

# Paths
DATA_DIR = "europarl_data"
EN_FILE = f"{DATA_DIR}/europarl-v7.es-en.en"

MODEL_DIR = "models"
MODEL_FILE = f"{MODEL_DIR}/ngram_model.pkl"

# Output
OUTPUT_DIR = "output"
SAMPLES_FILE = f"{OUTPUT_DIR}/generated_samples.txt"
PERPLEXITY_FILE = f"{OUTPUT_DIR}/perplexity_results.txt"

# Special tokens
START_TOKEN = "<START>"
END_TOKEN = "<END>"
UNK_TOKEN = "<UNK>"
