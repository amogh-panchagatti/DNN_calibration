#!/bin/bash
# Quick script to check if datasets are ready

echo "Checking for new datasets..."
echo ""

if [ -f "Datasets/train_full_snr.npz" ]; then
    echo "✓ Training dataset found:"
    ls -lh Datasets/train_full_snr.npz
else
    echo "✗ Training dataset not ready yet (train_full_snr.npz)"
fi

echo ""

if [ -f "Datasets/test_full_snr_bcrb.npz" ]; then
    echo "✓ Test dataset found:"
    ls -lh Datasets/test_full_snr_bcrb.npz
else
    echo "✗ Test dataset not ready yet (test_full_snr_bcrb.npz)"
fi

echo ""

if [ -f "Datasets/train_full_snr.npz" ] && [ -f "Datasets/test_full_snr_bcrb.npz" ]; then
    echo "================================"
    echo "✓ All datasets ready!"
    echo "================================"
    echo ""
    echo "Next step: Train the improved model"
    echo "  $ python3 main_improved.py"
    echo ""
else
    echo "================================"
    echo "⏳ Dataset generation in progress..."
    echo "================================"
    echo ""
    echo "Check again in a few minutes with:"
    echo "  $ bash check_datasets.sh"
    echo ""
    echo "Or monitor the process:"
    echo "  $ ps aux | grep dataset_generation"
    echo ""
fi
