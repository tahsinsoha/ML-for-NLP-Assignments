#!/usr/bin/env python3
"""
Translation table generation and perplexity comparison for IBM Model 1
"""

import os
import math
import random
import pickle
from collections import Counter
from typing import List, Tuple

from config import (
    TOP_N_SOURCE_WORDS, TOP_N_TRANSLATIONS, 
    PREPROCESSED_DATA_FILE, FINAL_MODEL_FILE,
    TABLES_DIR, TRANSLATION_TABLE_FILE, PERPLEXITY_FILE
)
from training import IBMModel1


def load_preprocessed_data(filepath: str) -> List[Tuple[List[str], List[str]]]:
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_top_translations(model: IBMModel1, source_word: str, n: int = 5, exclude_punct: bool = False) -> List[Tuple[str, float]]:
    translations = []
    for e in model.target_vocab:
        if exclude_punct and is_punctuation(e):
            continue
        prob = model.t[e][source_word]
        if prob > 0:
            translations.append((e, prob))
    translations.sort(key=lambda x: x[1], reverse=True)
    return translations[:n]


def is_punctuation(word: str) -> bool:
    return all(c in '.,!?;:"\'()[]{}' for c in word)


def sentence_log2_probability(model: IBMModel1, source_sent: List[str], target_sent: List[str]) -> float:
    """log₂ P(e|f) = Σⱼ log₂( (1/(l+1)) × Σᵢ t(eⱼ|fᵢ) )"""
    source_with_null = ["NULL"] + source_sent
    l_plus_1 = len(source_with_null)
    
    log2_prob = 0.0
    for e in target_sent:
        prob_e = sum(model.t[e][f] for f in source_with_null)
        prob_e = prob_e / l_plus_1
        
        if prob_e > 0:
            log2_prob += math.log2(prob_e)
        else:
            log2_prob += math.log2(1e-10)
    
    return log2_prob


def log2_perplexity(model: IBMModel1, source_sent: List[str], target_sent: List[str]) -> float:
    """log₂ PP = -Σ log₂ p(e|f)"""
    log2_prob = sentence_log2_probability(model, source_sent, target_sent)
    if len(target_sent) == 0:
        return float('inf')
    return -log2_prob


def generate_single_table(model: IBMModel1, source_freq: Counter, exclude_punct: bool = False) -> List[str]:
    lines = []
    
    if exclude_punct:
        filtered_freq = Counter({w: f for w, f in source_freq.items() if not is_punctuation(w)})
        most_common = filtered_freq.most_common(TOP_N_SOURCE_WORDS)
    else:
        most_common = source_freq.most_common(TOP_N_SOURCE_WORDS)
    
    max_word_len = max(len(word) for word, _ in most_common)
    max_word_len = max(max_word_len, len("Spanish"))
    
    lines.append(f"{'Spanish':<{max_word_len}} {'Freq':>8}  {'#1':<20} {'#2':<20} {'#3':<20} {'#4':<20} {'#5':<20}")
    lines.append("-" * (max_word_len + 8 + 5 * 22))
    
    for source_word, freq in most_common:
        translations = get_top_translations(model, source_word, TOP_N_TRANSLATIONS, exclude_punct=exclude_punct)
        trans_strs = [f"{t} ({p:.3f})" for t, p in translations]
        while len(trans_strs) < 5:
            trans_strs.append("-")
        
        line = f"{source_word:<{max_word_len}} {freq:>8}  "
        line += "  ".join(f"{s:<20}" for s in trans_strs)
        lines.append(line)
    
    return lines


def generate_translation_tables(model: IBMModel1, parallel_data: List[Tuple[List[str], List[str]]]) -> str:
    lines = []
    
    lines.append(f"Translation Tables: Top {TOP_N_TRANSLATIONS} translations for {TOP_N_SOURCE_WORDS} most common Spanish words\n")
    
    source_freq = Counter()
    for f_sent, e_sent in parallel_data:
        source_freq.update(f_sent)
    
    lines.append("Table 1: Including punctuation\n")
    lines.extend(generate_single_table(model, source_freq, exclude_punct=False))
    lines.append("\n")
    lines.append("\nTable 2: Excluding punctuation\n")
    lines.extend(generate_single_table(model, source_freq, exclude_punct=True))
    
    return "\n".join(lines)


def generate_perplexity_comparison(model: IBMModel1, parallel_data: List[Tuple[List[str], List[str]]]) -> str:
    lines = []
    lines.append("Perplexity scores for Real sentences from the training data vs")
    lines.append("randomly-sampled target-language (English) sentences of the same length.\n")
    lines.append(f"{'#':<4} {'log₂ PP (Real)':>18} {'log₂ PP (Random)':>20}")
    lines.append("-" * 42)
    
    target_vocab_list = list(model.target_vocab)
    sample_indices = random.sample(range(len(parallel_data)), 5)
    
    details = []
    
    for i, idx in enumerate(sample_indices, 1):
        f_sent, e_sent_real = parallel_data[idx]
        e_sent_random = random.choices(target_vocab_list, k=len(e_sent_real))
        
        log2_ppl_real = log2_perplexity(model, f_sent, e_sent_real)
        log2_ppl_random = log2_perplexity(model, f_sent, e_sent_random)
        
        lines.append(f"{i:<4} {log2_ppl_real:>18,.2f} {log2_ppl_random:>20,.2f}")
        
        details.append(f"\nExample {i} ({len(e_sent_real)} words):")
        details.append(f"  Source (Spanish): {' '.join(f_sent)}")
        details.append(f"  Real Translation from Training data(English): {' '.join(e_sent_real)}")
        details.append(f"  Random (English): {' '.join(e_sent_random)}")
    
    lines.append("\nDetails:")
    lines.extend(details)
    
    return "\n".join(lines)


def main():
    random.seed(42)
    os.makedirs(TABLES_DIR, exist_ok=True)
    
    print(f"Loading data from {PREPROCESSED_DATA_FILE}...")
    parallel_data = load_preprocessed_data(PREPROCESSED_DATA_FILE)
    print(f"Loaded {len(parallel_data)} sentence pairs.")
    
    print(f"Loading model from {FINAL_MODEL_FILE}...")
    model = IBMModel1.load(FINAL_MODEL_FILE)
    print(f"Model loaded ({model.current_iteration} iterations)\n")
    
    tables = generate_translation_tables(model, parallel_data)
    with open(TRANSLATION_TABLE_FILE, 'w') as f:
        f.write(tables)
    print(f"Translation tables saved to {TRANSLATION_TABLE_FILE}")
    print(tables)
    print("\n")
    
    perplexity = generate_perplexity_comparison(model, parallel_data)
    with open(PERPLEXITY_FILE, 'w') as f:
        f.write(perplexity)
    print(f"Perplexity comparison saved to {PERPLEXITY_FILE}")
    print(perplexity)


if __name__ == "__main__":
    main()
