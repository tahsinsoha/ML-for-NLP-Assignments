#!/usr/bin/env python3

"""
Stack decoder for phrase-based machine translation.

Started from the code written in class (CSE 244B, lecture 09), then
expanded following the Koehn slides:
  - distortion penalty  (slide 10: d(x) = alpha^|x|)
  - coverage-aware phrase location (fixes the class XXX for duplicate words)
  - histogram pruning   (slide 58)
  - recombination       (slides 51-53)
"""

import math
from phrasetable import PHRASETABLE
import ngram_lm

BEAM_WIDTH = 200
DISTORTION_ALPHA = 0.5


class Hypothesis:
    def __init__(self, output, score, coverage, last_foreign_end):
        self.score = score
        self.output = output
        self.coverage = coverage
        self.last_foreign_end = last_foreign_end


def find_sublist(small, big):
    """Find the starting index of `small` inside `big`, or -1 if not found."""
    shorter_big = big[: len(small)]
    if small == shorter_big:
        return 0
    if len(small) >= len(big):
        return -1
    sub_answer = find_sublist(small, big[1:])
    if sub_answer == -1:
        return -1
    else:
        return 1 + sub_answer


def applicable(source_sentence, coverage, phrase_pair):
    """Check if a phrase pair can be applied given current coverage."""
    source_phrase, target_phrase, _ = phrase_pair
    source_phrase = source_phrase.split()

    with_coverage = [
        w if not covered else "" for (w, covered) in zip(source_sentence, coverage)
    ]
    if find_sublist(source_phrase, with_coverage) != -1:
        return True
    return False


def find_in_uncovered(phrase_words, source_sentence, coverage):
    """Find phrase in the uncovered positions of the source sentence.

    This fixes the class code's XXX about duplicate words: we search
    the coverage-masked version so we find the right occurrence.
    """
    masked = [
        w if not covered else "" for (w, covered) in zip(source_sentence, coverage)
    ]
    return find_sublist(phrase_words, masked)


def recombination_key(hypothesis, n):
    """Key for recombination (slides 51-53).

    Two hypotheses are indistinguishable for future search if they have:
      - same coverage vector
      - same last (n-1) English words (for the n-gram LM)
    """
    cov = tuple(hypothesis.coverage)
    lm_state = tuple(hypothesis.output[-(n - 1):]) if hypothesis.output else ()
    return (cov, lm_state)


def decode(source_sentence, lm, phrase_table=PHRASETABLE, beam_width=BEAM_WIDTH):
    """Decode a single source sentence using stack-based beam search.

    Follows the algorithm from Koehn slide 57:
      1: place empty hypothesis into stack 0
      2: for all stacks 0...n-1 do
      3:   for all hypotheses in stack do
      4:     for all translation options do
      5:       if applicable then
      6:         create new hypothesis
      7:         place in stack
      8:         recombine with existing hypothesis if possible
      9:         prune stack if too big
    """
    LENGTH = len(source_sentence)

    source_text = " ".join(source_sentence)
    relevant_phrases = [
        p for p in phrase_table if p[0] in source_text
    ]

    stacks = [[] for _ in range(LENGTH + 1)]

    stacks[0].append(Hypothesis([], 0, [0] * LENGTH, -1))

    for stack_num in range(len(stacks)):
        for hypothesis in stacks[stack_num]:
            for phrase in relevant_phrases:
                if applicable(source_sentence, hypothesis.coverage, phrase):

                    new_output = hypothesis.output + phrase[1].split()

                    new_score = hypothesis.score + phrase[2]

                    new_score += lm.joint_prob_of_new_words(
                        hypothesis.output, phrase[1]
                    )

                    phrase_as_list = phrase[0].split()
                    location = find_in_uncovered(
                        phrase_as_list, source_sentence, hypothesis.coverage
                    )
                    if location == -1:
                        continue

                    # Distortion penalty (Koehn slide 10)
                    # d(start_i - end_{i-1} - 1) = alpha^|distance|
                    distance = location - (hypothesis.last_foreign_end + 1)
                    distortion_cost = -math.log2(
                        DISTORTION_ALPHA ** abs(distance)
                    ) if distance != 0 else 0.0
                    new_score += distortion_cost

                    new_coverage = [x for x in hypothesis.coverage]
                    for i in range(len(phrase_as_list)):
                        new_coverage[location + i] = 1

                    new_last_end = location + len(phrase_as_list) - 1
                    new_hypothesis = Hypothesis(
                        new_output, new_score, new_coverage, new_last_end
                    )
                    new_stack_num = new_coverage.count(1)

                    stacks[new_stack_num].append(new_hypothesis)

                    # Recombination (Koehn slides 51-53):
                    # keep only the best hypothesis for each (coverage, lm_state)
                    seen = {}
                    pruned = []
                    for h in stacks[new_stack_num]:
                        key = recombination_key(h, lm.n)
                        if key not in seen or h.score < seen[key].score:
                            seen[key] = h
                    pruned = list(seen.values())

                    # Histogram pruning (Koehn slide 58)
                    pruned.sort(key=lambda h: h.score)
                    stacks[new_stack_num] = pruned[:beam_width]

    return stacks[LENGTH]


if __name__ == "__main__":
    lm = ngram_lm.LanguageModel()

    source_sentence = "la comisión europea".split()
    print(f"\nSource: {' '.join(source_sentence)}")
    print(f"Decoding with beam width {BEAM_WIDTH} ...\n")

    results = decode(source_sentence, lm)

    if results:
        print(f"Found {len(results)} complete translations:")
        for h in results[:10]:
            print(f"  {' '.join(h.output):50s}  cost = {h.score:.2f}")
    else:
        print("No complete translations found.")
