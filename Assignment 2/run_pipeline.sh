#!/bin/bash
# Run the complete N-gram Language Model pipeline

echo "=============================================="
echo "N-gram Language Model Pipeline"
echo "=============================================="
echo ""

# Change to script directory
cd "$(dirname "$0")"

echo "Step 1: Preprocessing"
echo "----------------------------------------------"
python3 preprocessing.py
echo ""

echo "Step 2: Training"
echo "----------------------------------------------"
python3 training.py
echo ""

echo "Step 3: Evaluation"
echo "----------------------------------------------"
python3 evaluate.py
echo ""

echo "Step 4: Generation"
echo "----------------------------------------------"
python3 generate.py
echo ""

echo "=============================================="
echo "Pipeline complete!"
echo "=============================================="
echo ""
echo "Output files:"
echo "  - models/ngram_model.pkl (trained model)"
echo "  - output/generated_samples.txt (sample sentences)"
