#!/usr/bin/env python3

NGRAM_ORDER = 4
NUM_TRAINING_SENTENCES = 50000
UNK_THRESHOLD = 2
BACKOFF_ALPHA = 0.4

DATA_DIR = "europarl_data"
EN_FILE = f"{DATA_DIR}/europarl-v7.es-en.en"

MODEL_DIR = "models"
MODEL_FILE = f"{MODEL_DIR}/ngram_model.pkl"

OUTPUT_DIR = "output"
SAMPLES_FILE = f"{OUTPUT_DIR}/generated_samples.txt"
PERPLEXITY_FILE = f"{OUTPUT_DIR}/perplexity_results.txt"

START_TOKEN = "<START>"
END_TOKEN = "<END>"
UNK_TOKEN = "<UNK>"
