#!/usr/bin/env python3

"""
Language model wrapper that loads our Assignment 2 n-gram model and
provides the same interface as the class code:

    lm.prob_of_sentence(words)           -> total cost in bits
    lm.joint_prob_of_new_words(ctx, new) -> cost in bits for new words
"""

import os
import sys
import math
import pickle
from collections import Counter, defaultdict

LM_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "Assignment 2", "models", "ngram_model.pkl"
)

START_TOKEN = "<START>"
END_TOKEN = "<END>"


class LanguageModel:
    def __init__(self, model_path=LM_MODEL_PATH):
        print(f"Loading language model from {model_path} ...")
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        self.n = model_data["n"]
        self.vocab = model_data["vocab"]
        self.total_unigrams = model_data["total_unigrams"]

        self.probs = {}
        for order, contexts in model_data["probs"].items():
            self.probs[int(order)] = {}
            for context, words in contexts.items():
                self.probs[int(order)][context] = dict(words)

        self.backoff_alpha = 0.4
        print(f"  Loaded {self.n}-gram model  (vocab size: {len(self.vocab):,})")

    def get_probability(self, word, context):
        """Return P(word | context) with backoff."""
        if len(context) >= self.n:
            context = context[-(self.n - 1):]

        order = len(context) + 1

        if order <= self.n and context in self.probs.get(order, {}):
            if word in self.probs[order][context]:
                return self.probs[order][context][word]

        if order > 1:
            shorter = context[1:] if len(context) > 0 else ()
            return self.backoff_alpha * self.get_probability(word, shorter)

        if 1 in self.probs and () in self.probs[1]:
            if word in self.probs[1][()]:
                return self.probs[1][()][word]

        return 1.0 / (self.total_unigrams + len(self.vocab))

    def prob_of_sentence(self, sentence):
        """Return total cost in bits for a complete sentence."""
        padded = [START_TOKEN] * (self.n - 1) + sentence + [END_TOKEN]
        total_bits = 0.0

        for i in range(self.n - 1, len(padded)):
            word = padded[i]
            context = tuple(padded[max(0, i - self.n + 1):i])
            prob = self.get_probability(word, context)

            if prob > 0:
                total_bits += -math.log2(prob)
            else:
                total_bits += 30.0

        return total_bits

    def joint_prob_of_new_words(self, previous_output, new_words):
        """Return cost in bits for new_words given the previous output context.

        This matches the interface from the class decoder code:
            lm.joint_prob_of_new_words(hypothesis.output, phrase[1])

        previous_output: list of English words generated so far
        new_words: string of new English words (space-separated) to score
        """
        if isinstance(new_words, str):
            new_words = new_words.split()

        window = [START_TOKEN] * (self.n - 1)

        if previous_output:
            tail = previous_output[-(self.n - 1):]
            for i in range(len(tail)):
                window[-(len(tail) - i)] = tail[i]

        total_bits = 0.0
        for w in new_words:
            window.append(w)
            window = window[1:]
            context = tuple(window[:-1])
            prob = self.get_probability(w, context)

            if prob > 0:
                total_bits += -math.log2(prob)
            else:
                total_bits += 30.0

        return total_bits


if __name__ == "__main__":
    lm = LanguageModel()
    print()
    test_sentences = [
        "this is a report".split(),
        "the european commission".split(),
        "report home go not is".split(),
    ]
    for s in test_sentences:
        bits = lm.prob_of_sentence(s)
        print(f"  {' '.join(s):40s}  cost = {bits:.2f} bits")
