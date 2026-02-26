#!/usr/bin/env python3

"""
Load the phrase table from Assignment 3 into the same format used in class:
a list of (source_phrase, target_phrase, cost) tuples.

The class code used:
    PHRASETABLE = [("er", "he", 1.0), ...]

We load from the phrase_table.txt file and produce the same structure.
"""

import os

PHRASE_TABLE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "Assignment 3", "tables", "phrase_table.txt"
)


def load_phrase_table(filepath=PHRASE_TABLE_PATH):
    """Parse phrase_table.txt and return a list of (source, target, cost) tuples.

    The file has a fixed-width format:
        columns  0-39 : source phrase (Spanish)
        columns 40-79 : target phrase (English)
        columns 80+   : cost(f|e)  cost(e|f)  count
    Cost is the sum of both directional costs (combined score in bits).
    """
    phrase_table = []

    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < 5:
                continue
            line = line.rstrip()
            if not line:
                continue

            source = line[:40].strip()
            target = line[40:80].strip()
            remainder = line[80:].split()

            if len(remainder) < 3 or not source or not target:
                continue

            try:
                cost_fe = float(remainder[0])
                cost_ef = float(remainder[1])
            except ValueError:
                continue

            cost = cost_fe + cost_ef
            phrase_table.append((source, target, cost))

    return phrase_table


PHRASETABLE = load_phrase_table()

if __name__ == "__main__":
    print(f"Loaded {len(PHRASETABLE)} phrase pairs")
    print("\nTop 20 by lowest cost:")
    for src, tgt, cost in sorted(PHRASETABLE, key=lambda x: x[2])[:20]:
        print(f"  {src:30s} -> {tgt:30s}  cost={cost:.4f}")
    print("\nTop 20 by highest cost:")
    for src, tgt, cost in sorted(PHRASETABLE, key=lambda x: -x[2])[:20]:
        print(f"  {src:30s} -> {tgt:30s}  cost={cost:.4f}")
