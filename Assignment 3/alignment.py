#!/usr/bin/env python3
"""Viterbi alignment extraction and grow-diag-final symmetrization."""

import os
import pickle
from typing import List, Tuple, Set

from config import (
    PREPROCESSED_DATA_FILE, FINAL_MODEL_FILE, REVERSE_MODEL_FILE,
    NUM_ALIGNMENT_PAIRS, TABLES_DIR, ALIGNMENTS_FILE
)
from training import IBMModel1, load_preprocessed_data


def t_prob(model: IBMModel1, target_word: str, source_word: str) -> float:
    """Return P(target | source) without creating defaultdict entries."""
    if target_word not in model.t:
        return 0.0
    inner = model.t[target_word]
    if source_word not in inner:
        return 0.0
    return inner[source_word]


def get_viterbi_alignment_f2e(
    model: IBMModel1,
    f_sent: List[str],
    e_sent: List[str],
) -> Set[Tuple[int, int]]:
    """
    Forward Viterbi alignment using P(e | f).
    For each English word e_j, find the Spanish word f_i that maximises P(e_j | f_i).
    Returns (e_pos, f_pos) tuples.
    """
    alignment: Set[Tuple[int, int]] = set()

    for j, e_word in enumerate(e_sent):
        best_f_pos = None
        best_prob = t_prob(model, e_word, "NULL")

        for i, f_word in enumerate(f_sent):
            prob = t_prob(model, e_word, f_word)
            if prob > best_prob:
                best_prob = prob
                best_f_pos = i

        if best_f_pos is not None:
            alignment.add((j, best_f_pos))

    return alignment


def get_viterbi_alignment_e2f(
    model: IBMModel1,
    f_sent: List[str],
    e_sent: List[str],
) -> Set[Tuple[int, int]]:
    """
    Reverse Viterbi alignment using P(f | e).
    For each Spanish word f_i, find the English word e_j that maximises P(f_i | e_j).
    Returns (e_pos, f_pos) tuples.
    """
    alignment: Set[Tuple[int, int]] = set()

    for i, f_word in enumerate(f_sent):
        best_e_pos = None
        best_prob = t_prob(model, f_word, "NULL")

        for j, e_word in enumerate(e_sent):
            prob = t_prob(model, f_word, e_word)
            if prob > best_prob:
                best_prob = prob
                best_e_pos = j

        if best_e_pos is not None:
            alignment.add((best_e_pos, i))

    return alignment


def grow_diag_final(
    e2f: Set[Tuple[int, int]],
    f2e: Set[Tuple[int, int]],
    en_len: int,
    fn_len: int,
) -> Set[Tuple[int, int]]:
    """Symmetrize two directional alignments with grow-diag-final."""
    NEIGHBORS = [(-1, 0), (0, -1), (1, 0), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

    alignment = e2f & f2e
    union_align = e2f | f2e

    aligned_e = {e for e, _ in alignment}
    aligned_f = {f for _, f in alignment}

    # GROW-DIAG
    changed = True
    while changed:
        changed = False
        for e in range(en_len):
            for f in range(fn_len):
                if (e, f) not in alignment:
                    continue
                for de, df in NEIGHBORS:
                    e_new, f_new = e + de, f + df
                    if 0 <= e_new < en_len and 0 <= f_new < fn_len:
                        if (e_new not in aligned_e or f_new not in aligned_f):
                            if (e_new, f_new) in union_align:
                                alignment.add((e_new, f_new))
                                aligned_e.add(e_new)
                                aligned_f.add(f_new)
                                changed = True

    # FINAL(e2f)
    for e_new in range(en_len):
        for f_new in range(fn_len):
            if (e_new not in aligned_e or f_new not in aligned_f):
                if (e_new, f_new) in e2f:
                    alignment.add((e_new, f_new))
                    aligned_e.add(e_new)
                    aligned_f.add(f_new)

    # FINAL(f2e)
    for e_new in range(en_len):
        for f_new in range(fn_len):
            if (e_new not in aligned_e or f_new not in aligned_f):
                if (e_new, f_new) in f2e:
                    alignment.add((e_new, f_new))
                    aligned_e.add(e_new)
                    aligned_f.add(f_new)

    return alignment


def main():
    print("Computing Viterbi alignments and symmetrizing\n")

    print(f"Loading data from {PREPROCESSED_DATA_FILE} ...")
    parallel_data = load_preprocessed_data(PREPROCESSED_DATA_FILE)
    alignment_data = parallel_data[:NUM_ALIGNMENT_PAIRS]
    print(f"  Total pairs available : {len(parallel_data)}")
    print(f"  Pairs for alignment   : {len(alignment_data)}\n")

    print(f"Loading forward model from {FINAL_MODEL_FILE} ...")
    forward_model = IBMModel1.load(FINAL_MODEL_FILE)
    print(f"  Forward model trained for {forward_model.current_iteration} iterations.\n")

    print(f"Loading reverse model from {REVERSE_MODEL_FILE} ...")
    reverse_model = IBMModel1.load(REVERSE_MODEL_FILE)
    print(f"  Reverse model trained for {reverse_model.current_iteration} iterations.\n")

    print("Computing Viterbi alignments and symmetrizing ...")
    symmetrized_alignments: List[Set[Tuple[int, int]]] = []

    for idx, (f_sent, e_sent) in enumerate(alignment_data):
        align_f2e = get_viterbi_alignment_f2e(forward_model, f_sent, e_sent)
        align_e2f = get_viterbi_alignment_e2f(reverse_model, f_sent, e_sent)
        sym = grow_diag_final(align_e2f, align_f2e, len(e_sent), len(f_sent))
        symmetrized_alignments.append(sym)

        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1:>5}/{len(alignment_data)} sentence pairs aligned")

    print(f"  Done -- {len(symmetrized_alignments)} alignments computed.\n")

    os.makedirs(TABLES_DIR, exist_ok=True)

    alignments_pkl = ALIGNMENTS_FILE.replace(".txt", ".pkl")
    with open(alignments_pkl, "wb") as fout:
        pickle.dump(symmetrized_alignments, fout)
    print(f"Alignments (pickle) saved to {alignments_pkl}")

    with open(ALIGNMENTS_FILE, "w", encoding="utf-8") as fout:
        fout.write(f"Word Alignments ({len(alignment_data)} sentence pairs, "
                    "grow-diag-final symmetrization)\n\n\n")
        for idx, (f_sent, e_sent) in enumerate(alignment_data):
            align = symmetrized_alignments[idx]
            points = " ".join(f"{e}-{f}" for e, f in sorted(align))

            # Build per-English-word alignment lines
            e_to_f: dict = {}
            for e_pos, f_pos in sorted(align):
                e_to_f.setdefault(e_pos, []).append(f_pos)

            fout.write(f"\n         Pair #{idx + 1}\n")
            fout.write(f"ES: {' '.join(f_sent)}\n")
            fout.write(f"EN: {' '.join(e_sent)}\n\n")
            fout.write(f"Word alignments: {points}\n")
            for j, e_word in enumerate(e_sent):
                if j in e_to_f:
                    targets = ", ".join(f"{f_sent[fp]}({fp})" for fp in e_to_f[j])
                    fout.write(f"    {e_word} ({j}) -> {targets}\n")
            fout.write("\n\n")
    print(f"Alignments (text)   saved to {ALIGNMENTS_FILE}")

    print("\nSample alignments (first 3 pairs):")
    for idx in range(min(3, len(alignment_data))):
        f_sent, e_sent = alignment_data[idx]
        align = symmetrized_alignments[idx]
        points = " ".join(f"{e}-{f}" for e, f in sorted(align))
        print(f"\n  Pair {idx + 1}:")
        print(f"    ES: {' '.join(f_sent)}")
        print(f"    EN: {' '.join(e_sent)}")
        print(f"    Alignment: {points}")


if __name__ == "__main__":
    main()
