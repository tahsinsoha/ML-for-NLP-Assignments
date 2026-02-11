#!/usr/bin/env python3
"""Configuration for IBM Model 1 and Phrase-Based MT Pipeline"""

NUM_TRAINING_PAIRS = 10000
NUM_EM_ITERATIONS = 15

NUM_ALIGNMENT_PAIRS = 1000
MAX_PHRASE_LENGTH = 5
TOP_N_PHRASES = 5

DATA_DIR = "europarl_data"
CHECKPOINT_DIR = "checkpoints"
TABLES_DIR = "tables"

ES_FILE = f"{DATA_DIR}/europarl-v7.es-en.es"
EN_FILE = f"{DATA_DIR}/europarl-v7.es-en.en"
PREPROCESSED_DATA_FILE = "preprocessed_data.pkl"

MODEL_CHECKPOINT_PREFIX = f"{CHECKPOINT_DIR}/model_iter"
FINAL_MODEL_FILE = f"{CHECKPOINT_DIR}/model_final.pkl"

REVERSE_CHECKPOINT_PREFIX = f"{CHECKPOINT_DIR}/reverse_model_iter"
REVERSE_MODEL_FILE = f"{CHECKPOINT_DIR}/model_reverse_final.pkl"

ALIGNMENTS_FILE = f"{TABLES_DIR}/alignments.txt"
PHRASE_TABLE_FILE = f"{TABLES_DIR}/phrase_table.txt"
TOP_PHRASES_FILE = f"{TABLES_DIR}/top_phrases.txt"
