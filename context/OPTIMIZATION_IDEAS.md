# XRD Dataset Generation - Optimization Ideas

## Current Performance
- **100k samples in 30 minutes** (with multiprocessing)
- **Main bottleneck:** `RozrachKogerTT` function (triple nested loop)

---

## ✅ LEVEL 1: Quick Wins (Implement First)

### A. FFT Convolution in Zgortka
- **Status:** ⚠️ Needs careful testing
- **Math:** FFT convolution is mathematically equivalent to manual convolution
- **Issue:** Current code has custom indexing/alignment - need to verify mode
- **Expected speedup:** 10-20% overall (2-5x for convolution part)
- **Effort:** 1-2 hours (testing required)

### B. Reduce m1 Parameter
- **Status:** ❌ Not acceptable (lower quality curves)
- **Keep in mind:** Last resort if needed
- **m1=700 → m1=350:** 2x speedup but fewer curve points

---

## ✅ LEVEL 2: Numba JIT (Recommended) ← **IMPLEMENTED!**

### Apply @njit to RozrachKogerTT loops
- **Status:** ✅ **IMPLEMENTED in xrd_parallel.py**
- **Target:** Sublayer loop in RozrachKogerTT (hottest path)
- **Expected speedup:** 2-5x
- **File:** `xrd_parallel.py`
- **Features:**
  - JIT-compiled `_compute_sublayer_loop_jit()` function
  - Benchmarking utilities with warmup and timing
  - Step-by-step performance breakdown
  - Compatible API with original xrd.py

### Usage:
```bash
# Demo mode (single simulation with plot)
python xrd_parallel.py

# Benchmark mode (performance testing)
python xrd_parallel.py benchmark

# Benchmark with N samples
python xrd_parallel.py benchmark 50
```

---

## 📋 LEVEL 3: Parameter Tradeoffs

### dl (sublayer thickness)
- **Current:** dl = 400e-8 cm (400 Å) ← already minimal acceptable
- **Optimal:** dl = 100e-8 cm (100 Å) ← future goal
- **Note:** Fewer sublayers = faster, but less accurate physics

---

## 🔮 LEVEL 4: GPU Acceleration (Future)

### Batch processing on CUDA
- **Status:** ⏸️ Too much work for now (2-3 days)
- **Expected speedup:** 10-50x
- **Keep in mind:** For future when scaling to millions of samples

---

## 🚫 LEVEL 5: Hybrid/Interpolation (Skipped)

Pre-compute dense grid + interpolation
- Not pursuing due to accuracy concerns

---

## Implementation Priority

1. **NOW:** Numba JIT on RozrachKogerTT ← Start here
2. **THEN:** Test FFT convolution carefully
3. **FUTURE:** GPU if needed for massive scale
