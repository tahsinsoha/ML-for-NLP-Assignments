#!/usr/bin/env python3

import os
import pickle
from typing import List

from config import NGRAM_ORDER, MODEL_DIR, MODEL_FILE
from ngram_model import NGramModel


def load_preprocessed_data(filepath: str) -> dict:
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def split_data(sentences: List[List[str]], train_ratio: float = 0.9):
    split_idx = int(len(sentences) * train_ratio)
    return sentences[:split_idx], sentences[split_idx:]


def main():
    print("=" * 60)
    print("N-gram Language Model Training")
    print("=" * 60)
    
    preprocessed_file = f"{MODEL_DIR}/preprocessed_data.pkl"
    
    if not os.path.exists(preprocessed_file):
        print(f"Preprocessed data not found. Running preprocessing...")
        import preprocessing
        preprocessing.main()
    
    print(f"\nLoading preprocessed data from {preprocessed_file}...")
    data = load_preprocessed_data(preprocessed_file)
    sentences = data['sentences']
    vocab = data['vocab']
    stats = data['stats']
    
    print(f"  Sentences: {stats['num_sentences']:,}")
    print(f"  Vocabulary: {len(vocab):,}")
    
    train_sentences, test_sentences = split_data(sentences, train_ratio=0.9)
    print(f"\nData split:")
    print(f"  Training: {len(train_sentences):,} sentences")
    print(f"  Test: {len(test_sentences):,} sentences")
    
    print(f"\n{'='*60}")
    print(f"Training {NGRAM_ORDER}-gram model...")
    print(f"{'='*60}")
    
    model = NGramModel(n=NGRAM_ORDER)
    model.collect_counts(train_sentences)
    model.compute_probabilities()
    
    print(f"\nEvaluating on test set...")
    train_perplexity = model.perplexity(train_sentences[:1000])
    test_perplexity = model.perplexity(test_sentences)
    
    print(f"  Train perplexity (sample): {train_perplexity:.2f}")
    print(f"  Test perplexity: {test_perplexity:.2f}")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(MODEL_FILE)
    
    test_file = f"{MODEL_DIR}/test_sentences.pkl"
    with open(test_file, 'wb') as f:
        pickle.dump(test_sentences, f)
    print(f"Test sentences saved to {test_file}")
    
    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
