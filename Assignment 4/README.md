# Assignment 4 — Phrase-Based Decoder

Tying together our phrase-based machine translation system: word alignments (Assignment 1/3), language model (Assignment 2), phrase extraction (Assignment 3), and now a stack decoder.

## How It Works

We implemented a **stack decoder** starting from the code written in class (CSE 244B, lecture 09), then expanded it following the Koehn slides on phrase-based decoding.

### Components

1. **Phrase Table** (`phrasetable.py`) — Loads the 5,575 scored phrase pairs extracted in Assignment 3 from `phrase_table.txt`. Each entry has a Spanish source phrase, English target phrase, and a combined cost in bits (sum of both directional costs: cost(f|e) + cost(e|f), following slide 26's bidirectional approach).

2. **Language Model** (`ngram_lm.py`) — Loads our trained 4-gram English language model from Assignment 2. Provides `joint_prob_of_new_words(context, new_words)` which scores new English words given the translation so far, returning cost in bits. Uses proper backoff (from Assignment 2) instead of the class code's flat 8-bit penalty.

3. **Stack Decoder** (`decoder.py`) — The core search algorithm, following the Koehn slides:
   - Maintains one "stack" per number of source words covered (0 through sentence length)
   - Starts with an empty hypothesis in stack 0
   - Expands each hypothesis by trying all applicable phrase pairs
   - Scores each new hypothesis: `phrase_cost + LM_cost + distortion_cost`
   - Recombines equivalent hypotheses (slides 51–53)
   - Prunes each stack to the top 200 hypotheses via histogram pruning (slide 58)
   - Returns all complete translations from the final stack

4. **Translation Script** (`translate.py`) — Tokenizes Spanish input sentences and runs the decoder, displaying the best translations alongside reference English.

### The Decoding Algorithm (Koehn slide 57)

```
1: place empty hypothesis into stack 0
2: for all stacks 0...n-1 do
3:   for all hypotheses in stack do
4:     for all translation options do
5:       if applicable then
6:         create new hypothesis
7:         place in stack
8:         recombine with existing hypothesis if possible
9:         prune stack if too big
10:      end if
11:    end for
12:  end for
13: end for
```

### Scoring (Koehn slides 10, 23–25)

Each hypothesis extension is scored by three components, all in bits (lower = better):

```
new_score = old_score
          + phrase_table_cost          (slide 19: φ(f|e), plus slide 26: φ(e|f))
          + language_model_cost        (slide 23: pLM for new English words)
          + distortion_cost            (slide 10: α^|distance|, α = 0.5)
```

The **distortion penalty** (slide 10) penalizes reordering: if the current phrase starts at position `start_i` and the previous phrase ended at `end_{i-1}`, the distance is `start_i - end_{i-1} - 1`. Monotone translation (distance = 0) is free; jumping costs `-log2(0.5^|distance|)` = 1 bit per position skipped.

### Improvements Over the Class Code

The class code was a minimal live-coded sketch. Following the professor's instruction to "expand this out and make it work better," we added:

| Improvement | Source | What it does |
|---|---|---|
| Distortion penalty | Koehn slide 10 | Penalizes out-of-order translation |
| Recombination | Koehn slides 51–53 | Drops duplicate hypotheses with same coverage + LM state |
| Coverage-aware phrase location | Fixes class `XXX` comment | Correctly handles duplicate source words |
| Proper LM backoff | Assignment 2 | Uses trained backoff instead of flat 8-bit penalty |
| Bidirectional phrase scores | Koehn slide 26 | Uses both φ(f\|e) and φ(e\|f) |

## Running

```bash
cd "Assignment 4"
bash run_pipeline.sh
```

Or run just the decoder:

```bash
python3 translate.py
```

## Results

| # | Spanish | Best Translation | Reference | Match? |
|---|---------|-----------------|-----------|--------|
| 1 | la comisión europea | the european commission | the european commission | YES |
| 2 | señor presidente , este informe es importante | mr president , this report is important | mr president , this report is important | YES |
| 3 | la unión europea | the european union | the european union | YES |
| 4 | la política regional | the regional policy | regional policy | no |
| 5 | los estados miembros | member states | member states | YES |
| 6 | este debate es importante | this debate is important | this debate is important | YES |
| 7 | el parlamento europeo | the european parliament | the european parliament | YES |
| 8 | sin embargo , la comisión | however , the commission | however , the commission | YES |
| 9 | mi grupo no ha | my group did not | my group has not | no |
| 10 | la seguridad es importante | the safety is important | safety is important | no |

**7 out of 10 exact matches.** The 3 mismatches are still reasonable:
- Sentence 4: added "the" — valid, the phrase table translates "la" as "the"
- Sentence 9: "did not" vs "has not" — both valid translations of "no ha"; the LM prefers "did not"
- Sentence 10: added "the" — same as sentence 4

## What Worked

- The stack decoder correctly finds good translations for sentences with vocabulary covered by our phrase table
- The language model effectively steers the decoder toward fluent English word order (e.g., "european commission" not "commission european")
- Multi-word phrase pairs like "unión europea" → "european union" and "sin embargo" → "however" translate idiomatically
- The distortion penalty prevents wild reorderings that the class code allowed

## Limitations

- Our phrase table has only 5,575 entries from 1,000 sentence pairs — longer or rarer sentences will have coverage gaps
- No future cost estimation (Koehn slides 61–65) — the decoder may prune good hypotheses that start with expensive words
- The distortion model is simple distance-based; lexicalized reordering (slides 29–31) would be better
- No weight tuning (slide 14) — we use equal weights for all components

## Files

| File | Description |
|------|-------------|
| `phrasetable.py` | Loads phrase table from Assignment 3 |
| `ngram_lm.py` | Wraps Assignment 2 language model |
| `decoder.py` | Stack decoder with distortion + recombination |
| `translate.py` | Main script — tokenizes and translates test sentences |
| `run_pipeline.sh` | Runs the full pipeline |
| `output/translations.txt` | Saved translation results |
