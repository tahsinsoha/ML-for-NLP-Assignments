#!/usr/bin/env python3
"""
Preprocessing module for N-gram Language Model
Loads and tokenizes monolingual English data from Europarl corpus
"""

import os
import re
import pickle
import urllib.request
from collections import Counter
from typing import List, Tuple

from config import (
    NUM_TRAINING_SENTENCES, EN_FILE, UNK_THRESHOLD,
    START_TOKEN, END_TOKEN, UNK_TOKEN, MODEL_DIR, DATA_DIR
)


def download_europarl():
    """Download Europarl Spanish-English parallel corpus if not present."""
    base_url = "https://www.statmt.org/europarl/v7/"
    es_en_file = "es-en.tgz"
    
    os.makedirs(DATA_DIR, exist_ok=True)
    tgz_path = os.path.join(DATA_DIR, es_en_file)
    
    # Check if English file exists
    if os.path.exists(EN_FILE):
        print("Europarl data already downloaded.")
        return
    
    if not os.path.exists(tgz_path):
        print(f"Downloading Europarl corpus...")
        urllib.request.urlretrieve(base_url + es_en_file, tgz_path)
        print("Download complete.")
    
    print("Extracting files...")
    import tarfile
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(DATA_DIR, filter='data')
    
    print("Europarl data ready.")


def tokenize(text: str) -> List[str]:
    """Tokenize text: lowercase, split on whitespace, separate punctuation."""
    text = text.lower()
    text = re.sub(r'([.,!?;:"\'\(\)\[\]{}])', r' \1 ', text)
    tokens = [t for t in text.split() if t]
    return tokens


def load_sentences(filepath: str, num_sentences: int) -> List[List[str]]:
    """Load and tokenize sentences from a file."""
    print(f"Loading {num_sentences} sentences from {filepath}...")
    
    sentences = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= num_sentences:
                break
            tokens = tokenize(line.strip())
            if tokens:  # Skip empty lines
                sentences.append(tokens)
    
    print(f"Loaded {len(sentences)} sentences.")
    return sentences


def build_vocabulary(sentences: List[List[str]], unk_threshold: int) -> Tuple[set, Counter]:
    """
    Build vocabulary and replace rare words with <UNK>.
    Returns (vocabulary set, word counts).
    """
    print(f"Building vocabulary (UNK threshold = {unk_threshold})...")
    
    # Count all words
    word_counts = Counter()
    for sentence in sentences:
        word_counts.update(sentence)
    
    # Build vocabulary (words appearing >= threshold times)
    vocab = {word for word, count in word_counts.items() if count >= unk_threshold}
    vocab.add(START_TOKEN)
    vocab.add(END_TOKEN)
    vocab.add(UNK_TOKEN)
    
    print(f"Total unique words: {len(word_counts)}")
    print(f"Vocabulary size (after UNK): {len(vocab)}")
    
    return vocab, word_counts


def replace_unk(sentences: List[List[str]], vocab: set) -> List[List[str]]:
    """Replace out-of-vocabulary words with <UNK>."""
    processed = []
    for sentence in sentences:
        processed.append([
            word if word in vocab else UNK_TOKEN 
            for word in sentence
        ])
    return processed


def prepare_data() -> Tuple[List[List[str]], set, Counter]:
    """Main preprocessing pipeline. Returns (sentences, vocab, word_counts)."""
    
    # Download data if needed
    if not os.path.exists(EN_FILE):
        download_europarl()
    
    # Load sentences
    sentences = load_sentences(EN_FILE, NUM_TRAINING_SENTENCES)
    
    # Build vocabulary
    vocab, word_counts = build_vocabulary(sentences, UNK_THRESHOLD)
    
    # Replace rare words with <UNK>
    sentences = replace_unk(sentences, vocab)
    
    # Count UNK tokens
    unk_count = sum(1 for sent in sentences for word in sent if word == UNK_TOKEN)
    total_tokens = sum(len(sent) for sent in sentences)
    print(f"UNK tokens: {unk_count}/{total_tokens} ({100*unk_count/total_tokens:.2f}%)")
    
    return sentences, vocab, word_counts


def get_corpus_stats(sentences: List[List[str]]) -> dict:
    """Compute corpus statistics for the writeup."""
    total_tokens = sum(len(sent) for sent in sentences)
    avg_length = total_tokens / len(sentences)
    
    return {
        'num_sentences': len(sentences),
        'total_tokens': total_tokens,
        'avg_sentence_length': avg_length,
    }


def main():
    print("Preprocessing English Monolingual Data (Europarl)\n")
    
    sentences, vocab, word_counts = prepare_data()
    
    # Print statistics
    stats = get_corpus_stats(sentences)
    print(f"\nCorpus Statistics:")
    print(f"  Sentences: {stats['num_sentences']:,}")
    print(f"  Total tokens: {stats['total_tokens']:,}")
    print(f"  Avg sentence length: {stats['avg_sentence_length']:.1f} words")
    print(f"  Vocabulary size: {len(vocab):,}")
    
    # Sample sentence
    print(f"\nSample tokenized sentence:")
    print(f"  {sentences[0][:10]}...")
    
    # Save preprocessed data
    os.makedirs(MODEL_DIR, exist_ok=True)
    data = {
        'sentences': sentences,
        'vocab': vocab,
        'word_counts': word_counts,
        'stats': stats
    }
    
    preprocessed_file = f"{MODEL_DIR}/preprocessed_data.pkl"
    with open(preprocessed_file, 'wb') as f:
        pickle.dump(data, f)
    print(f"\nSaved preprocessed data to {preprocessed_file}")


if __name__ == "__main__":
    main()
