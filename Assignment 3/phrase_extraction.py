#!/usr/bin/env python3
"""
Phrase extraction and scoring from symmetrized word alignments.
"""

import os
import math
import pickle
from collections import defaultdict
from typing import List, Tuple, Set, Dict

from config import (
    PREPROCESSED_DATA_FILE, NUM_ALIGNMENT_PAIRS, MAX_PHRASE_LENGTH,
    TABLES_DIR, ALIGNMENTS_FILE, PHRASE_TABLE_FILE, TOP_PHRASES_FILE,
    TOP_N_PHRASES
)
from training import load_preprocessed_data

def extract_phrases(
    e_sent: List[str],
    f_sent: List[str],
    alignment: Set[Tuple[int, int]],
    max_length: int = 5,
) -> List[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
    """Extract all phrase pairs consistent with the word alignment."""
    phrases: List[Tuple[Tuple[str, ...], Tuple[str, ...]]] = []
    en_len = len(e_sent)
    fn_len = len(f_sent)

    e_to_f: Dict[int, Set[int]] = defaultdict(set)
    f_to_e: Dict[int, Set[int]] = defaultdict(set)
    for e, f in alignment:
        e_to_f[e].add(f)
        f_to_e[f].add(e)

    for e_start in range(en_len):
        for e_end in range(e_start, min(e_start + max_length, en_len)):
            f_min = fn_len
            f_max = -1
            for e_pos in range(e_start, e_end + 1):
                for f_pos in e_to_f.get(e_pos, ()):
                    f_min = min(f_min, f_pos)
                    f_max = max(f_max, f_pos)

            if f_max < 0:
                continue

            consistent = True
            for f_pos in range(f_min, f_max + 1):
                for e_pos in f_to_e.get(f_pos, ()):
                    if e_pos < e_start or e_pos > e_end:
                        consistent = False
                        break
                if not consistent:
                    break
            if not consistent:
                continue

            fs = f_min
            while True:
                fe = f_max
                while True:
                    if fe - fs + 1 <= max_length:
                        e_phrase = tuple(e_sent[e_start : e_end + 1])
                        f_phrase = tuple(f_sent[fs : fe + 1])
                        phrases.append((e_phrase, f_phrase))

                    fe += 1
                    if fe >= fn_len or fe - fs + 1 > max_length:
                        break
                    if fe in f_to_e:
                        break

                fs -= 1
                if fs < 0:
                    break
                if fs in f_to_e:
                    break

    return phrases


def score_phrases(
    all_phrases: List[Tuple[Tuple[str, ...], Tuple[str, ...]]],
) -> Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], Tuple[float, float, int]]:
    """
    Score phrase pairs by relative frequency and convert to log2 costs.

        phi(f | e) = count(e, f) / sum_f' count(e, f')
        phi(e | f) = count(e, f) / sum_e' count(e', f)
        cost = -log2(phi)   (lower = more probable)
    """
    pair_count: Dict[Tuple, int] = defaultdict(int)
    e_total: Dict[Tuple, int] = defaultdict(int)
    f_total: Dict[Tuple, int] = defaultdict(int)

    for e_phrase, f_phrase in all_phrases:
        pair_count[(e_phrase, f_phrase)] += 1
        e_total[e_phrase] += 1
        f_total[f_phrase] += 1

    scored: Dict[Tuple, Tuple[float, float, int]] = {}
    for (e_phrase, f_phrase), count in pair_count.items():
        prob_f_given_e = count / e_total[e_phrase]
        prob_e_given_f = count / f_total[f_phrase]

        cost_f_given_e = -math.log2(prob_f_given_e)
        cost_e_given_f = -math.log2(prob_e_given_f)

        scored[(e_phrase, f_phrase)] = (cost_f_given_e, cost_e_given_f, count)

    return scored


def main():
    print("Extracting and scoring phrases\n")

    print(f"Loading data from {PREPROCESSED_DATA_FILE} ...")
    parallel_data = load_preprocessed_data(PREPROCESSED_DATA_FILE)
    alignment_data = parallel_data[:NUM_ALIGNMENT_PAIRS]
    print(f"  Using {len(alignment_data)} sentence pairs.\n")

    alignments_pkl = ALIGNMENTS_FILE.replace(".txt", ".pkl")
    print(f"Loading alignments from {alignments_pkl} ...")
    with open(alignments_pkl, "rb") as fin:
        symmetrized_alignments = pickle.load(fin)
    print(f"  Loaded {len(symmetrized_alignments)} alignments.\n")

    print(f"Extracting phrases (max length = {MAX_PHRASE_LENGTH}) ...")
    all_phrases: List[Tuple[Tuple[str, ...], Tuple[str, ...]]] = []

    for idx, (f_sent, e_sent) in enumerate(alignment_data):
        align = symmetrized_alignments[idx]
        phrases = extract_phrases(e_sent, f_sent, align, MAX_PHRASE_LENGTH)
        all_phrases.extend(phrases)

        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1:>5}/{len(alignment_data)} pairs  "
                  f"({len(all_phrases):,} phrases so far)")

    print(f"\n  Total phrase pair tokens extracted : {len(all_phrases):,}")

    print("Scoring phrase pairs by relative frequency ...")
    scored = score_phrases(all_phrases)
    print(f"  Unique phrase pair types           : {len(scored):,}\n")

    os.makedirs(TABLES_DIR, exist_ok=True)

    sorted_pairs = sorted(scored.items(), key=lambda x: -x[1][2])

    # Writing phrase table
    SW = 40  # source column width
    TW = 40  # target column width
    CW = 12  # cost column width

    with open(PHRASE_TABLE_FILE, "w", encoding="utf-8") as fout:
        fout.write(f"Phrase Table: {len(scored):,} unique phrase pairs "
                   f"from {len(alignment_data)} sentence pairs "
                   f"(max phrase length = {MAX_PHRASE_LENGTH})\n")
        fout.write("Costs are in bits: -log2(probability). Lower = more probable.\n\n")

        hdr = (f"{'Source (Spanish)':<{SW}} {'Target (English)':<{TW}} "
               f"{'cost(f|e)':>{CW}} {'cost(e|f)':>{CW}} {'count':>7}")
        fout.write(hdr + "\n")
        fout.write("-" * len(hdr) + "\n")

        for (e_phrase, f_phrase), (c_fe, c_ef, cnt) in sorted_pairs:
            f_str = " ".join(f_phrase)
            e_str = " ".join(e_phrase)
            fout.write(f"{f_str:<{SW}} {e_str:<{TW}} "
                       f"{c_fe:>{CW}.4f} {c_ef:>{CW}.4f} {cnt:>7}\n")

    print(f"Phrase table saved to {PHRASE_TABLE_FILE}")

    # Writing top phrases by length
    by_length: Dict[int, list] = defaultdict(list)
    for (e_phrase, f_phrase), (c_fe, c_ef, cnt) in sorted_pairs:
        length = len(f_phrase)
        if length == 0:
            continue
        by_length[length].append(((e_phrase, f_phrase), (c_fe, c_ef, cnt)))

    lines: List[str] = []
    lines.append(f"Top {TOP_N_PHRASES} Most Common Phrases by Length "
                 f"({MAX_PHRASE_LENGTH}--1), where each punctuation mark "
                 f"is treated as a single token\n")

    for length in range(MAX_PHRASE_LENGTH, 0, -1):
        lines.append(f"\nPhrase length {length}  "
                     f"({len(by_length[length])} unique pairs)\n")

        hdr = (f"{'Source (Spanish)':<{SW}} {'Target (English)':<{TW}} "
               f"{'cost(f|e)':>{CW}} {'cost(e|f)':>{CW}} {'count':>7}")
        lines.append(hdr)
        lines.append("-" * len(hdr))

        top = by_length[length][:TOP_N_PHRASES]
        if not top:
            lines.append("(no phrases of this length)")
        for (e_ph, f_ph), (c_fe, c_ef, cnt) in top:
            f_str = " ".join(f_ph)
            e_str = " ".join(e_ph)
            lines.append(f"{f_str:<{SW}} {e_str:<{TW}} "
                         f"{c_fe:>{CW}.4f} {c_ef:>{CW}.4f} {cnt:>7}")

    output_text = "\n".join(lines) + "\n"
    print("\n" + output_text)

    with open(TOP_PHRASES_FILE, "w", encoding="utf-8") as fout:
        fout.write(output_text)
    print(f"Top phrases summary saved to {TOP_PHRASES_FILE}")


if __name__ == "__main__":
    main()
