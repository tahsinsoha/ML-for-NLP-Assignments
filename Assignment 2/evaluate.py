#!/usr/bin/env python3

import os
import pickle
import random

from config import MODEL_FILE, MODEL_DIR, OUTPUT_DIR, PERPLEXITY_FILE, NGRAM_ORDER
from ngram_model import NGramModel


def load_test_sentences():
    test_file = f"{MODEL_DIR}/test_sentences.pkl"
    with open(test_file, 'rb') as f:
        return pickle.load(f)


def evaluate_sentence_probabilities(model: NGramModel, sentences: list, num_examples: int = 5):
    print("\nExample Sentence Probabilities:")
    print("-" * 60)
    
    for i, sentence in enumerate(random.sample(sentences, num_examples)):
        text = ' '.join(sentence[:10])
        if len(sentence) > 10:
            text += "..."
        log_prob = model.prob_of_sentence(sentence)
        print(f"  {i+1}. \"{text}\"")
        print(f"     log P = {log_prob:.4f}")


def compare_real_vs_shuffled(model: NGramModel, real_sentences: list, num_comparisons: int = 5):
    print("\nReal vs Shuffled Sentence Comparison:")
    print("-" * 60)
    print("(Comparing original sentences to shuffled versions of the same words)")
    
    results = []
    medium_sentences = [s for s in real_sentences if 8 <= len(s) <= 15]
    
    for i in range(num_comparisons):
        real_sent = random.choice(medium_sentences)
        real_log_prob = model.prob_of_sentence(real_sent)
        
        shuffled_sent = real_sent.copy()
        random.shuffle(shuffled_sent)
        shuffled_log_prob = model.prob_of_sentence(shuffled_sent)
        
        results.append({
            'real_sent': real_sent,
            'real_log_prob': real_log_prob,
            'shuffled_sent': shuffled_sent,
            'shuffled_log_prob': shuffled_log_prob
        })
        
        print(f"\n  Comparison {i+1}:")
        print(f"    Original: \"{' '.join(real_sent)}\"")
        print(f"      → log P = {real_log_prob:.4f}")
        print(f"    Shuffled: \"{' '.join(shuffled_sent)}\"")
        print(f"      → log P = {shuffled_log_prob:.4f}")
        diff = real_log_prob - shuffled_log_prob
        print(f"    Difference: {diff:.4f} (original is {abs(diff):.1f} log-units {'better' if diff > 0 else 'worse'})")
    
    return results


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
    
    print(f"\n{'='*60}")
    print("Perplexity Evaluation")
    print("=" * 60)
    
    perplexity = model.perplexity(test_sentences)
    print(f"\nTest Set Perplexity: {perplexity:.2f}")
    
    evaluate_sentence_probabilities(model, test_sentences)
    
    print(f"\n{'='*60}")
    comparison_results = compare_real_vs_shuffled(model, test_sentences)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(PERPLEXITY_FILE, 'w') as f:
        f.write(f"N-gram Language Model Evaluation Results\n")
        f.write(f"Model: {NGRAM_ORDER}-gram\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test Set Perplexity: {perplexity:.2f}\n\n")
        f.write("Real vs Shuffled Comparison:\n")
        f.write("-" * 60 + "\n")
        for i, result in enumerate(comparison_results, 1):
            f.write(f"\nComparison {i}:\n")
            f.write(f"  Original: \"{' '.join(result['real_sent'])}\"\n")
            f.write(f"    log P = {result['real_log_prob']:.4f}\n")
            f.write(f"  Shuffled: \"{' '.join(result['shuffled_sent'])}\"\n")
            f.write(f"    log P = {result['shuffled_log_prob']:.4f}\n")
    
    print(f"\nResults saved to {PERPLEXITY_FILE}")
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
