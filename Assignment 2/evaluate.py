#!/usr/bin/env python3

import os
import pickle
import random

from config import MODEL_FILE, MODEL_DIR, NGRAM_ORDER
from ngram_model import NGramModel


def load_test_sentences():
    test_file = f"{MODEL_DIR}/test_sentences.pkl"
    with open(test_file, 'rb') as f:
        return pickle.load(f)


def evaluate_sentence_probabilities(model: NGramModel, sentences: list, num_examples: int = 5):
    """Show log probability for sample sentences."""
    print("\nExample Sentence Probabilities:")
    print("-" * 60)
    
    for i, sentence in enumerate(random.sample(sentences, num_examples)):
        text = ' '.join(sentence[:10])
        if len(sentence) > 10:
            text += "..."
        log_prob = model.prob_of_sentence(sentence)
        print(f"  {i+1}. \"{text}\"")
        print(f"     log P = {log_prob:.4f}")


def main():
    print("=" * 60)
    print("N-gram Language Model Evaluation")
    print("=" * 60)
    
    if not os.path.exists(MODEL_FILE):
        print(f"Model not found at {MODEL_FILE}. Please run training.py first.")
        return
    
    print(f"\nLoading {NGRAM_ORDER}-gram model...")
    model = NGramModel.load(MODEL_FILE)
    
    test_sentences = load_test_sentences()
    print(f"Loaded {len(test_sentences)} test sentences")
    
    evaluate_sentence_probabilities(model, test_sentences)
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
