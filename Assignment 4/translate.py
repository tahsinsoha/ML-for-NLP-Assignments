#!/usr/bin/env python3

"""
Translate Spanish sentences into English using our phrase-based decoder.

Loads the phrase table (Assignment 3) and language model (Assignment 2),
tokenizes input sentences the same way as preprocessing.py, and runs
the stack decoder on each one.
"""

import re
import os
import sys

from phrasetable import PHRASETABLE
from ngram_lm import LanguageModel
from decoder import decode, BEAM_WIDTH


def tokenize(text):
    """Same tokenizer as preprocessing.py: lowercase, separate punctuation."""
    text = text.lower()
    text = re.sub(r'([.,!?;:"\'\(\)\[\]{}])', r' \1 ', text)
    tokens = [t for t in text.split() if t]
    return tokens


TEST_SENTENCES = [
    # Short sentences built from high-frequency phrases in our table
    ("la comisión europea", "the european commission"),
    ("señor presidente , este informe es importante",
     "mr president , this report is important"),
    ("la unión europea", "the european union"),
    ("la política regional", "regional policy"),
    ("los estados miembros", "member states"),
    ("este debate es importante", "this debate is important"),
    ("el parlamento europeo", "the european parliament"),
    ("sin embargo , la comisión", "however , the commission"),
    ("mi grupo no ha", "my group has not"),
    ("la seguridad es importante", "safety is important"),
]


def main():
    print("=" * 70)
    print("Phrase-Based Machine Translation: Spanish -> English")
    print("=" * 70)
    print(f"  Phrase table : {len(PHRASETABLE)} phrase pairs (from Assignment 3)")

    lm = LanguageModel()
    print(f"  Beam width   : {BEAM_WIDTH}")
    print()

    output_lines = []
    output_lines.append("Phrase-Based MT Translation Results")
    output_lines.append("=" * 60)
    output_lines.append(f"Phrase pairs: {len(PHRASETABLE)}")
    output_lines.append(f"Beam width: {BEAM_WIDTH}")
    output_lines.append("")

    for i, (spanish, reference) in enumerate(TEST_SENTENCES, 1):
        source_tokens = tokenize(spanish)

        print("-" * 70)
        print(f"Sentence {i}:")
        print(f"  Spanish   : {spanish}")
        print(f"  Tokenized : {' '.join(source_tokens)}")
        print(f"  Reference : {reference}")
        print(f"  Decoding ...")

        results = decode(source_tokens, lm)

        output_lines.append("-" * 60)
        output_lines.append(f"Sentence {i}:")
        output_lines.append(f"  Spanish   : {spanish}")
        output_lines.append(f"  Reference : {reference}")

        if results:
            best = results[0]
            translation = " ".join(best.output)
            print(f"  Best      : {translation}")
            print(f"  Cost      : {best.score:.2f} bits")

            output_lines.append(f"  Best      : {translation}")
            output_lines.append(f"  Cost      : {best.score:.2f} bits")

            if len(results) > 1:
                print(f"  Runner-up : {' '.join(results[1].output)}  "
                      f"(cost={results[1].score:.2f})")

            match = "YES" if translation.strip() == reference.strip() else "no"
            print(f"  Match?    : {match}")
            output_lines.append(f"  Match?    : {match}")

            print(f"\n  Top 5 translations:")
            output_lines.append(f"  Top 5:")
            for j, h in enumerate(results[:5], 1):
                line = f"    {j}. {' '.join(h.output):50s}  cost={h.score:.2f}"
                print(line)
                output_lines.append(line)
        else:
            print("  ** No complete translation found **")
            output_lines.append("  ** No complete translation found **")

        print()
        output_lines.append("")

    output_path = os.path.join(os.path.dirname(__file__), "output", "translations.txt")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines) + "\n")
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
