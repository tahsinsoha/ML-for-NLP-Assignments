#!/usr/bin/env python3
"""
N-gram Language Model Implementation
Following Professor Rudnick's lecture code style

Our goal here is to build probability tables for P(word | context)
"""

import math
import random
import pickle
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional

from config import (
    NGRAM_ORDER, BACKOFF_ALPHA,
    START_TOKEN, END_TOKEN, UNK_TOKEN
)


class NGramModel:
    """
    N-gram language model with Stupid Backoff smoothing.
    
    Stores conditional probability tables P(word | context) for all n-gram orders
    from unigram up to the specified order.
    """
    
    def __init__(self, n: int = NGRAM_ORDER):
        self.n = n  # n-gram order (e.g., 3 for trigram)
        
        # counts[order][context][word] = count
        # For trigrams: counts[3][("the", "big")]["dog"] = 5
        self.counts: Dict[int, Dict[Tuple, Counter]] = {}
        for order in range(1, n + 1):
            self.counts[order] = defaultdict(Counter)
        
        # Probability tables (computed from counts)
        # probs[order][context][word] = probability
        self.probs: Dict[int, Dict[Tuple, Dict[str, float]]] = {}
        
        self.vocab: set = set()
        self.total_unigrams: int = 0
    
    def collect_counts(self, sentences: List[List[str]]):
        """
        Count all n-grams in the training data.
        Uses sliding window approach from professor's code.
        """
        print(f"Collecting {self.n}-gram counts...")
        
        for sentence in sentences:
            # Pad sentence with START tokens and END token
            # For trigrams: <START> <START> word1 word2 ... <END>
            padded = [START_TOKEN] * (self.n - 1) + sentence + [END_TOKEN]
            
            # Slide window across the sentence
            window = [START_TOKEN] * self.n
            
            for token in padded[self.n - 1:]:  # Start after initial padding
                window.append(token)
                window = window[1:]  # Slide window
                
                # Extract context and last token 
                lasttoken = window[-1]
                
                # Count n-grams of all orders (1 to n)
                for order in range(1, self.n + 1):
                    if order == 1:
                        context = ()  # Unigram has empty context
                    else:
                        context = tuple(window[-(order):][:-1])
                    
                    self.counts[order][context][lasttoken] += 1
        
        # Build vocabulary from unigram counts
        self.vocab = set(self.counts[1][()].keys())
        self.total_unigrams = sum(self.counts[1][()].values())
        
        # Print statistics
        for order in range(1, self.n + 1):
            num_contexts = len(self.counts[order])
            total_ngrams = sum(sum(c.values()) for c in self.counts[order].values())
            print(f"  {order}-grams: {total_ngrams:,} tokens, {num_contexts:,} unique contexts")
    
    def compute_probabilities(self):
        """
        Convert counts to conditional probabilities.
        P(word | context) = count(context, word) / count(context)
        """
        print("Computing conditional probabilities...")
        
        for order in range(1, self.n + 1):
            self.probs[order] = defaultdict(dict)
            
            for context in self.counts[order].keys():
                denominator = sum(self.counts[order][context].values())
                
                for word, count in self.counts[order][context].items():
                    prob = count / denominator
                    self.probs[order][context][word] = prob
    
    def get_probability(self, word: str, context: Tuple[str, ...]) -> float:
        """
        Get P(word | context) using Stupid Backoff.
        
        If we haven't seen (context, word), back off to shorter context
        with discount factor alpha.
        """
        # Adjust context length to match model order
        if len(context) >= self.n:
            context = context[-(self.n - 1):]
        
        order = len(context) + 1
        
        # Try to find probability at current order
        if order <= self.n and context in self.probs[order]:
            if word in self.probs[order][context]:
                return self.probs[order][context][word]
        
        # Stupid Backoff: try shorter context with discount
        if order > 1:
            shorter_context = context[1:] if len(context) > 0 else ()
            return BACKOFF_ALPHA * self.get_probability(word, shorter_context)
        
        # Unigram fallback
        if word in self.probs[1][()]:
            return self.probs[1][()][word]
        
        # Unknown word: return small probability
        return 1.0 / (self.total_unigrams + len(self.vocab))
    
    def prob_of_sentence(self, sentence: List[str]) -> float:
        """
        Compute probability of a sentence.
        Returns log probability to avoid underflow.
        
        P(w1, w2, ..., wn) = P(w1|start) * P(w2|start,w1) * ... * P(</s>|wn-1,wn)
        """
        # Pad with start tokens
        padded = [START_TOKEN] * (self.n - 1) + sentence + [END_TOKEN]
        
        log_prob = 0.0
        
        for i in range(self.n - 1, len(padded)):
            word = padded[i]
            context = tuple(padded[max(0, i - self.n + 1):i])
            
            prob = self.get_probability(word, context)
            
            if prob > 0:
                log_prob += math.log(prob)
            else:
                log_prob += math.log(1e-10)  # Small probability for unseen
        
        return log_prob
    
    def perplexity(self, sentences: List[List[str]]) -> float:
        """
        Compute perplexity on a set of sentences.
        Perplexity = exp(-1/N * sum(log P(sentence)))
        
        Lower perplexity = better model.
        """
        total_log_prob = 0.0
        total_tokens = 0
        
        for sentence in sentences:
            total_log_prob += self.prob_of_sentence(sentence)
            total_tokens += len(sentence) + 1  # +1 for <END> token
        
        avg_log_prob = total_log_prob / total_tokens
        perplexity = math.exp(-avg_log_prob)
        
        return perplexity
    
    def sample_next_word(self, context: Tuple[str, ...]) -> str:
        """
        Sample the next word given a context.
        Uses the probability distribution P(word | context).
        """
        # Adjust context length
        if len(context) >= self.n:
            context = context[-(self.n - 1):]
        
        order = len(context) + 1
        
        # Get distribution for this context
        if order <= self.n and context in self.probs[order]:
            distribution = self.probs[order][context]
        elif order > 1:
            # Back off to shorter context
            return self.sample_next_word(context[1:])
        else:
            # Use unigram distribution
            distribution = self.probs[1][()]
        
        # Sample from distribution
        words = list(distribution.keys())
        probs = list(distribution.values())
        
        # Normalize probabilities 
        total = sum(probs)
        probs = [p / total for p in probs]
        
        return random.choices(words, weights=probs, k=1)[0]
    
    def generate_sentence(self, max_length: int = 50) -> List[str]:
        """
        Generate a new sentence by sampling from the model.
        Starts with <START> context and samples until <END> or max_length.
        """
        # Start with n-1 START tokens as context
        context = tuple([START_TOKEN] * (self.n - 1))
        sentence = []
        
        for _ in range(max_length):
            next_word = self.sample_next_word(context)
            
            if next_word == END_TOKEN:
                break
            
            sentence.append(next_word)
            
            # Update context (slide window)
            context = tuple(list(context)[1:] + [next_word])
        
        return sentence
    
    def save(self, filepath: str):
        """Save model to disk."""
        model_data = {
            'n': self.n,
            'counts': {order: dict(contexts) for order, contexts in self.counts.items()},
            'probs': {order: dict(contexts) for order, contexts in self.probs.items()},
            'vocab': self.vocab,
            'total_unigrams': self.total_unigrams
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'NGramModel':
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        model = cls(n=model_data['n'])
        model.vocab = model_data['vocab']
        model.total_unigrams = model_data['total_unigrams']
        
        # Restore counts
        for order, contexts in model_data['counts'].items():
            model.counts[int(order)] = defaultdict(Counter)
            for context, counter in contexts.items():
                model.counts[int(order)][context] = Counter(counter)
        
        # Restore probs
        for order, contexts in model_data['probs'].items():
            model.probs[int(order)] = defaultdict(dict)
            for context, words in contexts.items():
                model.probs[int(order)][context] = dict(words)
        
        print(f"Model loaded from {filepath}")
        return model


def main():
    """Demo of the NGramModel class."""
    # Example usage
    sentences = [
        ["the", "cat", "sat", "on", "the", "mat"],
        ["the", "dog", "sat", "on", "the", "rug"],
        ["the", "cat", "chased", "the", "dog"],
    ]
    
    model = NGramModel(n=3)
    model.collect_counts(sentences)
    model.compute_probabilities()
    
    # Test probability lookup
    print("\nSample probabilities:")
    print(f"  P('sat' | 'cat') = {model.get_probability('sat', ('cat',)):.4f}")
    print(f"  P('the' | '<START>') = {model.get_probability('the', (START_TOKEN,)):.4f}")
    
    # Test sentence probability
    test_sent = ["the", "cat", "sat"]
    log_prob = model.prob_of_sentence(test_sent)
    print(f"\nlog P('{' '.join(test_sent)}') = {log_prob:.4f}")
    
    # Generate a sentence
    print("\nGenerated sentence:")
    generated = model.generate_sentence()
    print(f"  {' '.join(generated)}")


if __name__ == "__main__":
    main()
