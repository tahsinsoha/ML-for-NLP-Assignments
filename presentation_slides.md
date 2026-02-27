# CSE 244B — Phrase-Based Machine Translation
## Team 8: Sabiha Tahsin Soha, Halleluyah Brhanemesqel, Harshith Ravi Kopparam

---

<!-- Slide 1: Title -->

# Phrase-Based Machine Translation
## Spanish → English

**Team 8**
- Sabiha Tahsin Soha
- Halleluyah Brhanemesqel
- Harshith Ravi Kopparam

CSE 244B — NLP

---

<!-- Slide 2: Background -->

# Background

- **Languages:** Spanish → English
- **Why:** Similar word order makes phrase-based MT effective; rich parallel data available
- **Data:** 10,000 sentence pairs from the **Europarl** corpus (European Parliament proceedings)
- **Approach:** Classic phrase-based statistical MT pipeline across 4 assignments

---

<!-- Slide 3: Pipeline Overview -->

# System Pipeline

```
Assignment 1: IBM Model 1 (forward P(e|f))
      ↓
Assignment 2: 4-gram English Language Model
      ↓
Assignment 3: Reverse Model + Alignments + Phrase Table
      ↓
Assignment 4: Stack Decoder → Translations!
```

Each assignment builds one component. Assignment 4 ties them all together.

---

<!-- Slide 4: Assignment 1 — IBM Model 1 -->

# Assignment 1: IBM Model 1

**Goal:** Learn word translation probabilities P(e | f)

- Expectation-Maximization (EM) algorithm, 15 iterations
- For each English word, distribute credit across Spanish words proportionally
- NULL token handles English words with no Spanish counterpart

**Example learned probabilities:**
| Spanish | English | P(e\|f) |
|---------|---------|---------|
| casa    | house   | 0.80    |
| verde   | green   | 0.85    |
| la      | the     | 0.70    |

---

<!-- Slide 5: Assignment 2 — Language Model -->

# Assignment 2: 4-gram Language Model

**Goal:** Score English fluency — P(word | previous 3 words)

- Trained on ~50,000 Europarl English sentences
- **Stupid Backoff** smoothing (α = 0.4)
- 14,617 vocabulary words, 671,649 4-gram contexts
- Handles unknown words with `<UNK>` token

**Sample generated sentences:**
- "i do not usually speak too long ."
- "this is not a european army ."
- "with regard to eurodac ."

---

<!-- Slide 6: Assignment 3 — Alignments -->

# Assignment 3: Word Alignments

**Three steps:**

1. **Train reverse model** P(f | e) — swap source/target, run EM again
2. **Viterbi alignment** — for each word, pick the highest-probability partner
3. **Grow-Diag-Final** symmetrization — combine both directions

| Phase        | What it does                              |
|--------------|-------------------------------------------|
| Intersection | Keep only points BOTH models agree on     |
| Grow-Diag    | Expand to nearby union points carefully    |
| Final        | Fill remaining gaps from either direction  |

---

<!-- Slide 7: Assignment 3 — Phrase Extraction -->

# Assignment 3: Phrase Extraction & Scoring

**Consistency rule (Koehn slide 16):**
A phrase pair is valid only if no alignment link crosses the phrase boundaries.

- Extract all consistent phrase pairs (max length 5)
- Expand over unaligned boundary words (Koehn slide 17)
- **Result:** 5,575 scored phrase pairs from 1,000 sentence pairs

**Scoring:** Relative frequency in both directions → cost in bits
```
cost = -log₂(probability)
```
Lower cost = more confident translation

---

<!-- Slide 8: Sample Phrase Table Entries -->

# Phrase Table Examples

| Spanish | English | Cost (bits) | Meaning |
|---------|---------|:-----------:|---------|
| comisión europea | european commission | 0.00 | Perfect confidence |
| unión europea | european union | 0.00 | Perfect confidence |
| sin embargo | however | 0.00 | Idiomatic phrase |
| la | the | 0.59 | Some uncertainty (also "el"→"the") |
| comisión | commission | 0.29 | High confidence |
| comisión | committee | 4.64 | Rare alternative |

Bidirectional costs: cost(f|e) + cost(e|f) following Koehn slide 26

---

<!-- Slide 9: Assignment 4 — Stack Decoder -->

# Assignment 4: Stack Decoder

**Algorithm (Koehn slide 57):**

1. Place empty hypothesis in stack 0
2. For each stack, expand hypotheses with all applicable phrase pairs
3. Score: **phrase cost + LM cost + distortion penalty**
4. Recombine equivalent hypotheses (same coverage + LM state)
5. Prune each stack to top 200 (histogram pruning)
6. Return best complete translation from final stack

---

<!-- Slide 10: Scoring Components -->

# Three Scoring Components

| Component | Source | What it measures |
|-----------|--------|-----------------|
| **Phrase table cost** | Assignment 3 | Translation quality (φ(f\|e) + φ(e\|f)) |
| **Language model cost** | Assignment 2 | English fluency |
| **Distortion penalty** | Koehn slide 10 | Reordering cost (α = 0.5) |

```
new_score = old_score + phrase_cost + LM_cost + distortion_cost
```

Distortion: monotone (distance=0) is free; each skipped position costs 1 bit

---

<!-- Slide 11: Improvements Over Class Code -->

# Improvements Over Class Code

| Improvement | Source | Effect |
|---|---|---|
| Distortion penalty | Koehn slide 10 | Prevents wild reorderings |
| Recombination | Koehn slides 51–53 | Drops duplicate hypotheses |
| Coverage-aware matching | Fixes class `XXX` bug | Handles duplicate source words |
| Proper LM backoff | Assignment 2 | Replaces flat 8-bit penalty |
| Bidirectional phrase scores | Koehn slide 26 | Uses both φ(f\|e) and φ(e\|f) |

---

<!-- Slide 12: Translation Results -->

# Results: 7/10 Exact Matches

| # | Spanish | Our Translation | Reference | ✓? |
|---|---------|----------------|-----------|:--:|
| 1 | la comisión europea | the european commission | the european commission | ✓ |
| 2 | señor presidente , este informe es importante | mr president , this report is important | mr president , this report is important | ✓ |
| 3 | la unión europea | the european union | the european union | ✓ |
| 4 | la política regional | the regional policy | regional policy | ~ |
| 5 | los estados miembros | member states | member states | ✓ |
| 6 | este debate es importante | this debate is important | this debate is important | ✓ |
| 7 | el parlamento europeo | the european parliament | the european parliament | ✓ |
| 8 | sin embargo , la comisión | however , the commission | however , the commission | ✓ |
| 9 | mi grupo no ha | my group did not | my group has not | ~ |
| 10 | la seguridad es importante | the safety is important | safety is important | ~ |

---

<!-- Slide 13: Analysis of Mismatches -->

# Analysis of Mismatches

**Sentences 4 & 10:** Added "the" — the decoder correctly translates "la" as "the"; the reference just drops the article. Both are valid English.

**Sentence 9:** "did not" vs "has not" — both are valid translations of "no ha". The LM slightly prefers "did not" in this context.

**Key insight:** All 3 mismatches produce grammatically correct, semantically reasonable English. No nonsense outputs.

---

<!-- Slide 14: What Worked -->

# What Worked

- **Multi-word phrases** translate idiomatically:
  - "unión europea" → "european union"
  - "sin embargo" → "however"
- **Language model** steers toward fluent word order:
  - "european commission" not "commission european"
- **Distortion penalty** prevents wild reorderings
- **Stack decoder** efficiently searches the hypothesis space

---

<!-- Slide 15: Struggles & Limitations -->

# Struggles & Limitations

**Data limitations:**
- Only 5,575 phrase pairs from 1,000 sentence pairs
- Longer or rarer sentences have coverage gaps
- If a word/phrase isn't in the phrase table → no translation

**Decoder limitations:**
- No future cost estimation (Koehn slides 61–65) — may prune good hypotheses early
- Simple distance-based distortion — lexicalized reordering would be better
- No weight tuning (Koehn slide 14) — equal weights for all components

**Tokenization:**
- Contractions and special punctuation need better handling

---

<!-- Slide 16: Summary -->

# Summary

Built a complete phrase-based MT system across 4 assignments:

| Component | Assignment | Key Result |
|-----------|:----------:|------------|
| Word translation probs | 1 | IBM Model 1 with EM |
| English fluency scoring | 2 | 4-gram LM with backoff |
| Phrase table | 3 | 5,575 scored phrase pairs |
| Stack decoder | 4 | **7/10 exact match translations** |

All code uses only Python standard library — no external dependencies.

---

<!-- Slide 17: Thank You -->

# Thank You!

**Team 8**
- Sabiha Tahsin Soha
- Halleluyah Brhanemesqel
- Harshith Ravi Kopparam

All code and documentation available in our repository.
