#!/usr/bin/env python3

import os

from config import MODEL_FILE, OUTPUT_DIR, SAMPLES_FILE, NGRAM_ORDER
from ngram_model import NGramModel


def generate_samples(model: NGramModel, num_samples: int = 20, max_length: int = 30) -> list:
    samples = []
    for i in range(num_samples):
        sentence = model.generate_sentence(max_length=max_length)
        samples.append(' '.join(sentence))
    return samples


def main():
    print("=" * 60)
    print("Sentence Generation from N-gram Model")
    print("=" * 60)
    
    if not os.path.exists(MODEL_FILE):
        print(f"Model not found at {MODEL_FILE}. Please run training.py first.")
        return
    
    print(f"\nLoading {NGRAM_ORDER}-gram model...")
    model = NGramModel.load(MODEL_FILE)
    
    print(f"  Vocabulary size: {len(model.vocab):,}")
    print(f"  N-gram order: {model.n}")
    
    print(f"\n{'='*60}")
    print("Generated Sentences:")
    print("=" * 60)
    
    num_samples = 20
    samples = generate_samples(model, num_samples=num_samples)
    
    for i, sample in enumerate(samples, 1):
        print(f"{i:2}. {sample}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(SAMPLES_FILE, 'w') as f:
        f.write(f"Generated sentences from {NGRAM_ORDER}-gram model\n")
        f.write("=" * 60 + "\n\n")
        for i, sample in enumerate(samples, 1):
            f.write(f"{i:2}. {sample}\n")
    
    print(f"\nSamples saved to {SAMPLES_FILE}")
    
    print(f"\n{'='*60}")
    print("Interactive Generation (press Ctrl+C to exit)")
    print("=" * 60)
    
    try:
        while True:
            input("\nPress Enter to generate a new sentence...")
            sentence = model.generate_sentence(max_length=40)
            print(f"  → {' '.join(sentence)}")
            log_prob = model.prob_of_sentence(sentence)
            print(f"    (log probability: {log_prob:.4f})")
    except KeyboardInterrupt:
        print("\n\nDone!")


if __name__ == "__main__":
    main()
