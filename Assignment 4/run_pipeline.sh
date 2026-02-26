#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "Assignment 4: Phrase-Based Decoder"
echo "=================================================="
echo ""

echo "Step 1: Loading phrase table (from Assignment 3)"
python3 -u phrasetable.py
echo ""

echo "Step 2: Testing language model (from Assignment 2)"
python3 -u ngram_lm.py
echo ""

echo "Step 3: Running stack decoder on test sentences"
python3 -u translate.py
