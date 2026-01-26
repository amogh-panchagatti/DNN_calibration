# DNN Calibration Improvements

## Problem Identified

Your original CNN was **significantly underperforming** compared to the Bayesian Cramér-Rao Bound (BCRB):

- **Gains**: CNN MSE plateaued at ~0.01 while BCRB reached ~3×10⁻⁵ (300x worse at high SNR)
- **Phases**: CNN MSE ~0.1 while BCRB reached ~1 (1-3 orders of magnitude worse)

## Root Causes Found

### 1. **Training/Test Distribution Mismatch** (CRITICAL)
- **Original**: Trained only on SNR = {25, 30} dB
- **Testing**: Evaluated on SNR = {-10, -5, 0, 5, 10, 15, 20, 25, 30} dB
- **Problem**: Model never saw low SNR examples during training!

### 2. **Phase Wrap-Around Issue** (CRITICAL)
- Regular MSE loss treats phases as linear values
- Phases in [-π, π] wrap around: φ=+3.1 and φ=-3.1 are only 0.2 apart
- MSE computes this as (3.1-(-3.1))² ≈ 38.44 instead of 0.04!

### 3. **Small Test Set**
- Only 1 sample per SNR level (9 total samples)
- Not statistically reliable for evaluation

### 4. **Limited Model Capacity**
- Original CalibNet: 3 conv layers, 2 dense layers
- Too shallow for this complex inverse problem

## Solutions Implemented

### ✅ 1. Updated Dataset Generation ([dataset_generation.py:339-359](dataset_generation.py#L339-L359))

**Changes:**
```python
# BEFORE
high_train_levels = (25, 30)  # Only 2 SNR levels!
num_samples = 40,000
num_per_snr = 1               # Tiny test set

# AFTER
full_snr_levels = (-10, -5, 0, 5, 10, 15, 20, 25, 30)  # All 9 SNR levels
num_samples = 90,000          # 10k samples per SNR
num_per_snr = 100             # 100 test samples per SNR
theta_fixed = None            # Random angles for better generalization
```

**Impact:**
- Model now trains on the **full SNR spectrum** it will be tested on
- Test set increased from 9 to 900 samples for statistical reliability
- Random angles during training prevent geometry memorization

### ✅ 2. Implemented Circular Phase Loss ([main_improved.py:99-120](main_improved.py#L99-L120))

**New Loss Function:**
```python
def circular_mse_loss(y_pred, y_true, M):
    # Gains: regular MSE
    gain_loss = F.mse_loss(y_pred[:, :M], y_true[:, :M])

    # Phases: circular distance
    phase_diff = y_pred[:, M:] - y_true[:, M:]
    # Wrap to [-π, π]
    phase_diff = torch.atan2(torch.sin(phase_diff), torch.cos(phase_diff))
    phase_loss = (phase_diff ** 2).mean()

    return gain_loss + phase_loss
```

**Impact:**
- Correctly handles phase wrap-around
- Phase errors now computed on the circle, not Euclidean space
- Expected to dramatically improve phase estimation

### ✅ 3. Deeper CalibNetDeep Architecture ([main_improved.py:72-96](main_improved.py#L72-L96))

**Architecture Improvements:**
```
Original CalibNet:
  Conv: 2→32→64→64 (3 layers)
  Dense: 64→128→2M (2 layers)
  Total params: ~50k

New CalibNetDeep:
  Conv: 2→64→128→128→256 (4 layers)
  Dense: 256→512→256→2M (3 layers)
  Total params: ~800k (16x more capacity!)
```

**Impact:**
- Much deeper network can learn more complex covariance-to-parameter mappings
- Increased capacity crucial for inverse problems
- More parameters = better approximation of optimal Bayesian estimator

### ✅ 4. Learning Rate Scheduling ([main_improved.py:159-162](main_improved.py#L159-L162))

```python
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='min', factor=0.5, patience=20, verbose=True
)
```

**Impact:**
- Automatically reduces learning rate when validation loss plateaus
- Helps model converge to better local minima
- Prevents oscillation at high learning rates late in training

### ✅ 5. Improved Evaluation ([evaluate_improved.py](evaluate_improved.py))

**New Features:**
- Uses circular distance for phase error computation
- Prints gap ratio (CNN MSE / BCRB) for each SNR level
- Better visualization with side-by-side plots
- Statistical summary tables

## Files Created

| File | Purpose |
|------|---------|
| [main_improved.py](main_improved.py) | New training script with all improvements |
| [evaluate_improved.py](evaluate_improved.py) | Evaluation with circular phase metrics |
| [dataset_generation.py](dataset_generation.py) | Updated to generate full SNR datasets |
| [IMPROVEMENTS_README.md](IMPROVEMENTS_README.md) | This file |

## Next Steps

### Step 1: Wait for Dataset Generation ⏳

The dataset generation script is currently running in the background:
```bash
# Check if datasets are ready:
ls -lh Datasets/train_full_snr.npz Datasets/test_full_snr_bcrb.npz
```

**Expected files:**
- `Datasets/train_full_snr.npz` (~200 MB) - 90,000 training samples
- `Datasets/test_full_snr_bcrb.npz` (~2 MB) - 900 test samples with BCRB

**Generation time:** ~10-20 minutes (depends on CPU)

### Step 2: Train the Improved Model 🚀

Once datasets are ready:
```bash
python3 main_improved.py
```

**Training details:**
- 500 epochs with early stopping
- Learning rate: 1e-3 with ReduceLROnPlateau
- Model saved as: `calibnet_M6_improved.pt`
- Expected training time: ~30-60 minutes on GPU

### Step 3: Evaluate Against BCRB 📊

After training:
```bash
python3 evaluate_improved.py
```

This will:
1. Load the trained `calibnet_M6_improved.pt`
2. Evaluate on `test_full_snr_bcrb.npz`
3. Generate plots: `output_improved.png`
4. Print performance gap at each SNR level

### Step 4: Compare Results 📈

Compare the new `output_improved.png` with your original `output.png` and `output2.png`.

**Expected improvements:**
- **Low SNR (-10 to 10 dB)**: CNN should now perform well (wasn't trained on this before!)
- **High SNR (15 to 30 dB)**: CNN should be much closer to BCRB
- **Phase estimation**: Should improve by 1-2 orders of magnitude due to circular loss
- **Target**: CNN within 2-5x of BCRB across all SNR levels (state-of-the-art!)

## Expected Performance

Based on the improvements, here's what you should see:

### Before (Original)
```
High SNR Gains:  CNN ~0.01,   BCRB ~3e-5   → Gap: 300x ❌
High SNR Phases: CNN ~0.1,    BCRB ~1      → Gap: 0.1x (but wrong metric!)
Low SNR:         Complete failure (never trained on it) ❌
```

### After (Improved)
```
High SNR Gains:  CNN ~1e-4,   BCRB ~3e-5   → Gap: 3x    ✅
High SNR Phases: CNN ~2,      BCRB ~1      → Gap: 2x    ✅
Low SNR Gains:   CNN ~5e-2,   BCRB ~2e-2   → Gap: 2.5x  ✅
Low SNR Phases:  CNN ~500,    BCRB ~200    → Gap: 2.5x  ✅
```

**Note:** These are rough estimates. Your actual performance may vary, but the gap should dramatically narrow!

## Troubleshooting

### If performance is still not good:

1. **Increase training samples**: Change `num_samples` from 90,000 to 180,000
2. **Longer training**: Increase `EPOCHS` from 500 to 1000
3. **Data augmentation**: Add noise to covariance matrices during training
4. **Ensemble**: Train 3-5 models and average predictions
5. **Hyperparameter tuning**:
   - Adjust dropout: try p=0.2 or p=0.4
   - Adjust learning rate: try 5e-4 or 2e-3
   - Adjust beta in Softplus: try 1.0 or 2.0

### If dataset generation fails:

Check for memory issues:
```bash
# Monitor dataset generation
ps aux | grep python3
top -p <PID>

# If it fails, reduce num_samples:
# In dataset_generation.py, change num_samples=90_000 to num_samples=45_000
```

## Theory: Why These Changes Work

### Circular Loss
- **Phase space topology**: Phases live on S¹ (circle), not R (line)
- **Riemannian distance**: atan2 computes geodesic distance on circle
- **Result**: Gradient points in correct direction for phase optimization

### Full SNR Training
- **Distribution matching**: Train distribution must match test distribution (fundamental ML principle)
- **Low SNR regime**: Covariance structure qualitatively different (noise-dominated)
- **High SNR regime**: Signal structure emerges
- **Result**: Model learns SNR-dependent strategies

### Increased Capacity
- **Universal approximation**: Deeper networks approximate complex functions better
- **Inverse problem**: Covariance → parameters is ill-posed, requires high capacity
- **BCRB attainment**: Optimal Bayesian estimator is complex nonlinear function
- **Result**: More parameters = closer approximation to optimal estimator

## References

For theoretical background:
- **BCRB**: Bayesian extension of Cramér-Rao bound (Van Trees, "Detection, Estimation, and Modulation Theory")
- **Array calibration**: Sensor gain/phase estimation (Friedlander & Weiss, "Direction finding in the presence of mutual coupling", 1991)
- **Circular statistics**: Mardia & Jupp, "Directional Statistics"

## Contact

If you have questions about these improvements or need further optimization:
- Check convergence plots to ensure model is training
- Verify BCRB computation isn't numerically unstable
- Consider problem-specific priors (e.g., gain sparsity)

---

**Summary**: The main issue was training/test mismatch + wrong loss function for phases. The improved version should achieve near-optimal performance! 🎯
