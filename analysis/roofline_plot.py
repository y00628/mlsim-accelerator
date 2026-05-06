#!/usr/bin/env python3
"""
Roofline plot generator.

Usage:
  1. Run ./roofline from golden/build/ and copy the output numbers below
  2. Run: python3 roofline_plot.py
  3. Opens roofline.png showing your kernels plotted against the machine ceiling

What to measure for YOUR machine:
  - Peak GFLOP/s: run a BLAS dgemm benchmark or use `likwid-bench`
  - Peak bandwidth (GB/s): run `stream` benchmark
  For a rough estimate: bandwidth ≈ measured from roofline.cpp small-matrix runs
"""
import matplotlib.pyplot as plt
import numpy as np

# --- Fill these in after running benchmarks on your machine ---
PEAK_GFLOPS = 80.0    # your CPU's peak single-precision GFLOP/s
PEAK_BW_GBS = 40.0    # your memory bandwidth in GB/s

# Kernel measurements: (name, arithmetic_intensity, achieved_gflops)
# Copy arithmetic_intensity from roofline.cpp output, measure achieved_gflops similarly
KERNELS = [
    # ("kernel name",  AI FLOP/byte,  GFLOP/s achieved)
    # ("gemm_64",      X.XX,          X.XX),
    # ("gemm_512",     X.XX,          X.XX),
    # ("attention",    X.XX,          X.XX),
    # ("depthwise",    X.XX,          X.XX),
]

def roofline_ceiling(ai):
    """Min of compute roof and memory bandwidth roof."""
    return min(PEAK_GFLOPS, PEAK_BW_GBS * ai)

fig, ax = plt.subplots(figsize=(9, 5))

# Draw roofline
ai_range = np.logspace(-2, 4, 400)
roof = [roofline_ceiling(ai) for ai in ai_range]
ax.loglog(ai_range, roof, 'k-', linewidth=2, label='Roofline ceiling')

# Memory-bound region label
ridge_point = PEAK_GFLOPS / PEAK_BW_GBS
ax.axvline(ridge_point, color='gray', linestyle='--', alpha=0.5)
ax.text(ridge_point * 1.1, PEAK_GFLOPS * 0.6, f'ridge = {ridge_point:.1f} FLOP/B',
        fontsize=9, color='gray')

# Plot kernels
colors = plt.cm.tab10.colors
for i, (name, ai, perf) in enumerate(KERNELS):
    ax.scatter([ai], [perf], s=80, color=colors[i], zorder=5, label=name)
    efficiency = perf / roofline_ceiling(ai) * 100
    ax.annotate(f'{name}\n({efficiency:.0f}% eff.)', (ai, perf),
                textcoords='offset points', xytext=(8, 4), fontsize=8)

ax.set_xlabel('Arithmetic Intensity (FLOP / byte)', fontsize=11)
ax.set_ylabel('Performance (GFLOP/s)', fontsize=11)
ax.set_title('Roofline Model — mlsim-accelerator', fontsize=12)
ax.grid(True, which='both', alpha=0.2)
ax.legend(fontsize=9)
ax.set_xlim(1e-2, 1e4)
ax.set_ylim(0.1, PEAK_GFLOPS * 2)
plt.tight_layout()

out = 'roofline.png'
plt.savefig(out, dpi=150)
print(f'Saved {out}')
