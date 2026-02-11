#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Step 1: Preprocessing"
python3 -u preprocessing.py
echo ""

echo "Step 2: Training (forward & reverse models)"
python3 -u training.py
echo ""

echo "Step 3: Viterbi alignment & symmetrization"
python3 -u alignment.py
echo ""

echo "Step 4: Phrase extraction & scoring"
python3 -u phrase_extraction.py
