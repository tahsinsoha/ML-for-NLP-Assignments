#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Step 1: Preprocessing"
python3 -u preprocessing.py
echo ""

echo "Step 2: Training"
python3 -u training.py
echo ""

echo "Step 3: Analysis"
python3 -u translation_tables.py
