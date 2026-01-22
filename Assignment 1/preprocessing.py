#!/usr/bin/env python3
"""
Preprocessing module for IBM Model 1
Downloads Europarl corpus and tokenizes the data
"""

import os
import re
import pickle
import urllib.request
from typing import List, Tuple

from config import (
    NUM_TRAINING_PAIRS, DATA_DIR, ES_FILE, EN_FILE, PREPROCESSED_DATA_FILE
)


def download_europarl():
    """Download Europarl Spanish-English parallel corpus if not present."""
    base_url = "https://www.statmt.org/europarl/v7/"
    es_en_file = "es-en.tgz"
    
    os.makedirs(DATA_DIR, exist_ok=True)
    tgz_path = os.path.join(DATA_DIR, es_en_file)
    
    if os.path.exists(ES_FILE) and os.path.exists(EN_FILE):
        print("Europarl data already downloaded.")
        return ES_FILE, EN_FILE
    
    if not os.path.exists(tgz_path):
        print(f"Downloading Europarl corpus...")
        urllib.request.urlretrieve(base_url + es_en_file, tgz_path)
        print("Download complete.")
    
    print("Extracting files...")
    import tarfile
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(DATA_DIR, filter='data')
    
    return ES_FILE, EN_FILE


def tokenize(text: str) -> List[str]:
    """Tokenize text: lowercase, split on whitespace, separate punctuation."""
    text = text.lower()
    text = re.sub(r'([.,!?;:"\'\(\)\[\]{}])', r' \1 ', text)
    tokens = [t for t in text.split() if t]
    return tokens


def load_and_preprocess_corpus(es_file: str, en_file: str, num_pairs: int) -> List[Tuple[List[str], List[str]]]:
    """Load parallel corpus and return tokenized sentence pairs."""
    print(f"Loading first {num_pairs} sentence pairs...")
    
    parallel_data = []
    with open(es_file, 'r', encoding='utf-8') as f_es, \
         open(en_file, 'r', encoding='utf-8') as f_en:
        
        for i, (es_line, en_line) in enumerate(zip(f_es, f_en)):
            if i >= num_pairs:
                break
            
            es_tokens = tokenize(es_line.strip())
            en_tokens = tokenize(en_line.strip())
            
            if es_tokens and en_tokens:
                parallel_data.append((es_tokens, en_tokens))
    
    print(f"Loaded {len(parallel_data)} sentence pairs.")
    return parallel_data


def save_preprocessed_data(parallel_data: List[Tuple[List[str], List[str]]], filepath: str):
    with open(filepath, 'wb') as f:
        pickle.dump(parallel_data, f)
    print(f"Saved to {filepath}")


def main():
    print("Preprocessing: Spanish-English (Europarl)\n")
    
    es_file, en_file = download_europarl()
    parallel_data = load_and_preprocess_corpus(es_file, en_file, NUM_TRAINING_PAIRS)
    
    print(f"\nTokenizer: whitespace + punctuation, lowercased")
    print(f"Sample: {parallel_data[0][0][:8]}...")
    
    save_preprocessed_data(parallel_data, PREPROCESSED_DATA_FILE)


if __name__ == "__main__":
    main()
