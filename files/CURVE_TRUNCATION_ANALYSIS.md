# XRD Curve Truncation Analysis - CRITICAL FINDINGS

## Date
2025-10-29

## Question
Is the curve truncation logic (`start_ML=50`) correct for ML training?

```python
# xrd.py:823-825
m1_ML = m1 - start_ML  # 700 - 50 = 650
curve_X_ML = np.linspace(0, m1_ML - 1, m1_ML)
curve_Y_ML = np.asarray(R_convolved)[start_ML:m1]  # [50:700]
```

## Physical Meaning

### What Gets Truncated?

The XRD rocking curve for GGG + YIG film consists of:
1. **Main Bragg peak (points 0-50):** High-intensity peak from substrate/film interface
2. **Interference fringes (points 50-700):** Oscillations encoding deformation profile

**Truncation rationale:**
- The first 50 points (main peak) primarily reflect the perfect crystal structure
- The tail (interference fringes) contains information about **deformation parameters** in the defected layer
- For **parameter estimation**, the fringes are more informative than the peak position

### Physical Justification

From your thesis goal:
> "Визначення параметрів тонких плівок (товщина шару, шорсткість межі, густина тощо) з експериментальних кривих дифракції (XRD)"

The deformation profile parameters (Dmax1, D01, L1, Rp1, D02, L2, Rp2) primarily affect:
- ✅ **Interference fringe spacing** → encodes L1, L2 (layer thickness)
- ✅ **Fringe intensity modulation** → encodes Dmax1, D01, D02 (deformation magnitude)
- ✅ **Fringe asymmetry** → encodes Rp1, Rp2 (position of maximum deformation)

The main peak position encodes:
- ❌ **Lattice parameter** (not a fitted parameter in your model)
- ❌ **Bragg angle** (fixed by crystal structure)

**Conclusion:** Truncating the peak to focus on fringes is **physically justified** for your specific task.

---

## CRITICAL PROBLEM #1: Training vs Experimental Data Mismatch

### The Issue

**Training data** ([xrd.py:784](xrd.py#L784)):
```python
start_ML: int = 50  # Hardcoded default
curve_Y_ML = R_convolved[50:700]  # 650 points
```

**Experimental data** ([j_analyze_experiment.ipynb](j_analyze_experiment.ipynb)):
```python
exp_data_processed = preprocess_for_model(exp_data, start_ML=90, target_length=650)
                                                    # ^^^ DIFFERENT!
```

### Impact

This is a **domain shift** problem:

| Data Type | start_ML | What Gets Saved | Alignment |
|-----------|----------|-----------------|-----------|
| Training (synthetic) | 50 | Points 50-700 | Peak starts ~index 0 of saved curve |
| Experimental | 90 | Points 90-360 | Peak starts ~index -40 (missing!) |

**Result:**
- Model learns: "Parameter L2 causes oscillations starting at index X"
- Experimental data: "Oscillations start at index X-40"
- **Model predictions will be systematically wrong!**

### Evidence

From j_analyze_experiment.ipynb output:
```
Peak position: index 90 (in experimental data)
Truncation point (start_ML): 50
Points used for ML: 310 (padded to 650 with zeros)
```

But training uses start_ML=50, meaning if the synthetic curves also have peaks at index ~50-90, **the alignment is inconsistent**.

---

## CRITICAL PROBLEM #2: Fixed vs Adaptive Truncation

### Current Approach: Fixed start_ML=50

**Assumption:** All curves (synthetic and experimental) have the main peak at approximately the same index.

**Risk:**
1. **Different deformation profiles might shift peak position slightly**
   - Large L1 → peak might shift due to thickness effects
   - Large Dmax1 → peak broadening might change effective "start of tail"

2. **Experimental data has different angular sampling**
   - Synthetic: m1=700 points over specific angular range
   - Experimental: Variable number of points (360 in notebook example)
   - **Same index ≠ same angular position!**

3. **No verification that peak is at index 50**
   - I see no code checking peak position before truncation
   - What if some curves have peak at index 30 or 70?

### Better Approach: Adaptive Truncation

Find the peak position dynamically:

```python
def find_peak_and_truncate(curve, tail_fraction=0.9):
    """
    Adaptively find peak and truncate to tail region.

    Args:
        curve: R_convolved intensity values
        tail_fraction: Start tail at this fraction of peak intensity

    Returns:
        Truncated curve starting after peak
    """
    peak_idx = np.argmax(curve)
    peak_intensity = curve[peak_idx]

    # Find where intensity drops to tail_fraction of peak
    # (searching to the right of peak)
    tail_start = peak_idx
    for i in range(peak_idx, len(curve)):
        if curve[i] < tail_fraction * peak_intensity:
            tail_start = i
            break

    return curve[tail_start:], tail_start
```

This ensures:
- ✅ Consistent alignment (always starts at same relative position after peak)
- ✅ Works for different experimental setups
- ✅ Robust to peak shifts

---

## CRITICAL PROBLEM #3: Information Loss

### What Information is Discarded?

By truncating points 0-50, you're discarding:

1. **Peak position** (angular alignment)
   - Could be useful for detecting systematic errors
   - Could help model understand absolute angular scale

2. **Peak intensity**
   - Related to film quality/coherence
   - Could correlate with deformation parameters

3. **Peak width (FWHM)**
   - Encodes mosaic spread, defect density
   - **Directly relevant to your defected layer analysis!**

4. **Peak asymmetry**
   - Asymmetric peaks indicate non-uniform deformation
   - **Highly relevant for asymmetric Gaussian profile (DDPL1)!**

### Question: Should You Keep the Peak?

**Arguments FOR truncation:**
- Peak dominates intensity (6 orders of magnitude difference)
- Log-normalization already handles this, but truncation is simpler
- Reduces input dimensionality (less overfitting risk)
- Literature precedent (Ziegler et al. also truncate?)

**Arguments AGAINST truncation:**
- Peak shape encodes defect information
- Modern CNNs with log-normalization can handle full curve
- No need to manually engineer features when using deep learning
- More data = potentially better performance

**My recommendation:**
- **Short-term:** Keep truncation BUT FIX the alignment issues (see Solutions below)
- **Long-term:** Try training on full curves with log-normalization as an experiment
  - If it works better → validates that peak contains useful information
  - If it works worse → confirms truncation is optimal

---

## Solutions

### Solution 1: Fix Training/Experimental Mismatch (URGENT)

**Option A: Make experimental match training**
```python
# In j_analyze_experiment.ipynb
exp_data_processed = preprocess_for_model(exp_data, start_ML=50, ...)  # NOT 90!
```

**Option B: Make training match experimental**
```python
# In xrd.py
def compute_curve_and_profile(..., start_ML: int = 90):  # NOT 50!
```

**Recommendation:** Use **Option A** (start_ML=50 for both) if synthetic peaks are truly at index ~50. But first **verify peak positions**!

### Solution 2: Verify Peak Positions (URGENT)

Add diagnostic code to dataset generation:

```python
def verify_peak_position_distribution(dataset_path):
    """
    Check if all curves have peaks at similar positions.
    Critical for validating fixed start_ML truncation.
    """
    import pickle
    with open(dataset_path, 'rb') as f:
        data = pickle.load(f)

    X, Y = data['X'], data['Y']  # Y is curves before truncation

    peak_positions = []
    for i in range(len(Y)):
        # Compute full curve
        curve, _ = xrd.compute_curve_and_profile(
            array=X[i],
            dl=data.get('dl', 100e-8),
            m1=700
        )
        peak_idx = np.argmax(curve.Y_R_vseZ)
        peak_positions.append(peak_idx)

    peak_positions = np.array(peak_positions)
    print(f"Peak position statistics:")
    print(f"  Mean: {peak_positions.mean():.1f}")
    print(f"  Std:  {peak_positions.std():.1f}")
    print(f"  Min:  {peak_positions.min()}")
    print(f"  Max:  {peak_positions.max()}")

    if peak_positions.std() > 5:
        print("⚠️  WARNING: Peak positions vary significantly!")
        print("   Fixed start_ML may cause misalignment.")
    else:
        print("✅ Peak positions are consistent. Fixed start_ML is safe.")

    return peak_positions
```

Run this on your 100k dataset!

### Solution 3: Implement Adaptive Truncation (RECOMMENDED)

Modify xrd.py:

```python
def compute_curve_and_profile(
    array=None,
    dl: float = 100e-8,
    m1: int = 700,
    m10: int = 20,
    ik: float = 4.671897861,
    start_ML: int = None,  # None = auto-detect
    tail_threshold: float = 0.1,  # Start tail at 10% of peak
    params_obj: DeformationProfile = None
):
    # ... existing code ...

    # Compute full curve
    DeltaTeta, R_coger, R_convolved = simulator.RunSimulation(...)

    # Auto-detect truncation point if not specified
    if start_ML is None:
        peak_idx = np.argmax(R_convolved)
        peak_intensity = R_convolved[peak_idx]

        # Find where curve drops to threshold
        for i in range(peak_idx, len(R_convolved)):
            if R_convolved[i] < tail_threshold * peak_intensity:
                start_ML = i
                break

        if start_ML is None:
            start_ML = peak_idx + 10  # Fallback

    # ML truncation
    m1_ML = m1 - start_ML
    curve_X_ML = np.linspace(0, m1_ML - 1, m1_ML)
    curve_Y_ML = np.asarray(R_convolved)[start_ML:m1]

    # Return truncation point for diagnostics
    curve = Curve(
        ML_X=curve_X_ML.copy(),
        ML_Y=curve_Y_ML.copy(),
        X_DeltaTeta=DeltaTeta.copy(),
        Y_R_vseZ=R_convolved.copy(),
        Y_R_vse=R_coger.copy(),
        start_ML=start_ML  # NEW: track actual truncation point
    )

    return curve, profile
```

This makes truncation **robust** to peak position variations.

### Solution 4: Alternative - Use Full Curve (EXPERIMENTAL)

Train a model on the **full curve** (no truncation):

```python
# In xrd.py
def compute_curve_and_profile(..., use_full_curve: bool = False):
    if use_full_curve:
        # No truncation
        curve_Y_ML = np.asarray(R_convolved)[:m1]  # All 700 points
    else:
        # Existing truncation logic
        curve_Y_ML = np.asarray(R_convolved)[start_ML:m1]
```

Advantages:
- No information loss
- No alignment issues
- Peak shape features available to model

Disadvantages:
- Larger input (700 vs 650 points)
- Potential for peak intensity to dominate (but log-normalization helps)

**Test this experimentally:** Train v3 model on full curves, compare accuracy.

---

## Recommendations (Priority Order)

### 🔴 URGENT (Do Immediately)

1. **Fix training/experimental mismatch**
   - Verify what start_ML your experimental data actually needs
   - Make sure training uses the same value
   - **Current mismatch (50 vs 90) will cause systematic errors!**

2. **Verify peak positions in synthetic data**
   - Run diagnostic on 100k dataset
   - Confirm peaks are actually at index ~50
   - If they vary, adaptive truncation is REQUIRED

### 🟡 IMPORTANT (Do Before Final Thesis Experiments)

3. **Implement adaptive truncation**
   - Makes method robust
   - Better for thesis: "We use adaptive truncation to handle varying experimental conditions"
   - Publishable improvement

4. **Document truncation choice in thesis**
   - Explain physical reasoning (fringes encode deformation)
   - Show verification that peaks are at consistent positions
   - Justify start_ML=50 value (or adaptive method)

### 🟢 OPTIONAL (For Extended Research)

5. **Experiment with full curve training**
   - Quick experiment: train v3 on full 700-point curves
   - Compare with truncated results
   - If better: major finding! "Contrary to assumptions, peak shape contains deformation information"
   - If worse: confirms truncation was correct

6. **Multi-input model** (advanced)
   - Input 1: Peak region [0:100] → extract peak width, asymmetry
   - Input 2: Tail region [50:700] → extract fringe patterns
   - Two-branch CNN merging before MLP head
   - Potential for significant accuracy improvement

---

## Current Status Assessment

| Aspect | Status | Risk Level | Action Required |
|--------|--------|------------|-----------------|
| Training uses start_ML=50 | ✅ Implemented | 🟢 Low | None |
| Experimental uses start_ML=90 | ⚠️  Mismatch | 🔴 **CRITICAL** | **Fix immediately** |
| Peak positions verified | ❌ Unknown | 🟡 Medium | Run diagnostics |
| Truncation documented | ❌ No | 🟡 Medium | Add to thesis |
| Adaptive truncation | ❌ No | 🟡 Medium | Consider implementing |

---

## Conclusion

**Is current truncation logic OK?**

**Physically:** ✅ YES - Truncating the peak to focus on fringes is **correct** for deformation parameter estimation.

**Implementationally:** ⚠️  **PARTIALLY** - The concept is sound, but there are **critical bugs**:
1. 🔴 Training/experimental mismatch (50 vs 90)
2. 🟡 No verification of peak positions
3. 🟡 Fixed truncation fragile to experimental variations

**Should you change it?**

**Minimum fix (required for correctness):**
- ✅ Fix start_ML mismatch (make both use same value after verification)
- ✅ Verify peak positions in your dataset

**Recommended improvements:**
- ✅ Implement adaptive truncation
- ✅ Document in thesis methodology section

**Optional research:**
- 🤔 Try full-curve training as comparison experiment

**Bottom line:** The current approach is conceptually sound but has implementation bugs that MUST be fixed before final results. The training/experimental mismatch is particularly critical and likely degrading your model's performance on real data.
