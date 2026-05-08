#!/usr/bin/env python3
"""
Roofline plot — mlsim-accelerator golden model benchmarks.

Data source: golden/build/roofline  (run with -O3 -march=native)
AI model   : FLOPs / logical bytes (4×M×K×N per impl, /ts for tiled)
             naive → AI = 0.125 FLOP/B
             tiled → AI = ts/8  FLOP/B  (independent of matrix shape)

Machine ceilings (Apple M-series estimate — tune if needed):
  PEAK_GFLOPS  : single-core FP32 with auto-vectorisation
  PEAK_BW_GBS  : effective DRAM bandwidth for compute kernels
"""
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

# ── machine ceilings ────────────────────────────────────────────────────────
PEAK_GFLOPS = 80.0   # GFLOP/s  — adjust for your CPU
PEAK_BW_GBS = 50.0   # GB/s     — adjust for your CPU

# ── benchmark data: (label, AI FLOP/B, GFLOP/s) ────────────────────────────
# Grouped by tile strategy; matrix configs listed for each group.
CONFIGS = ["64³", "128³", "256³", "512³", "64×64×512", "512×128×512"]

# AI = 0.125 for all naive; AI = ts/8 for tiled
DATA = {
    "naive":    (0.125, [2.64, 2.84, 2.96, 2.78, 4.82, 3.60]),
    "tiled-16": (2.0,   [9.36, 11.36, 16.65, 15.95, 16.02, 16.53]),
    "tiled-32": (4.0,   [11.09, 13.17, 20.06, 19.01, 19.83, 19.68]),
    "tiled-64": (8.0,   [17.19, 21.20, 30.89, 22.98, 27.57, 24.82]),
}

STYLE = {
    "naive":    dict(color="#e15759", marker="X",  ms=9,  zorder=4),
    "tiled-16": dict(color="#4e79a7", marker="o",  ms=7,  zorder=5),
    "tiled-32": dict(color="#59a14f", marker="s",  ms=7,  zorder=5),
    "tiled-64": dict(color="#f28e2b", marker="^",  ms=9,  zorder=6),
}

# ── roofline ceiling ─────────────────────────────────────────────────────────
def ceiling(ai):
    return min(PEAK_GFLOPS, PEAK_BW_GBS * ai)

ai_range = np.logspace(-2, 4, 600)
roof     = [ceiling(ai) for ai in ai_range]
ridge    = PEAK_GFLOPS / PEAK_BW_GBS

# ── figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

ax.loglog(ai_range, roof, color="black", lw=2.5, label="Roofline ceiling")

# ridge-point marker
ax.axvline(ridge, color="gray", ls="--", lw=1.2, alpha=0.6)
ax.text(ridge * 1.08, PEAK_GFLOPS * 0.55,
        f"ridge ≈ {ridge:.1f} FLOP/B", fontsize=8.5, color="gray", va="top")

# memory-bound / compute-bound region shading
ax.fill_between(ai_range, roof, PEAK_GFLOPS * 2,
                where=[ai < ridge for ai in ai_range],
                color="#aec6e8", alpha=0.10, label="memory-bound region")
ax.fill_between(ai_range, roof, PEAK_GFLOPS * 2,
                where=[ai >= ridge for ai in ai_range],
                color="#ffcc99", alpha=0.10, label="compute-bound region")

# ── plot data points ─────────────────────────────────────────────────────────
# Add tiny horizontal jitter so points at the same AI don't perfectly overlap
rng = np.random.default_rng(42)

for group, (ai, perfs) in DATA.items():
    s = STYLE[group]
    # jitter ±2 % in log space
    jitter = rng.uniform(-0.02, 0.02, len(perfs))
    xs = ai * (1 + jitter)
    ax.scatter(xs, perfs,
               color=s["color"], marker=s["marker"], s=s["ms"]**2,
               zorder=s["zorder"], label=group, edgecolors="white", linewidths=0.4)

    # annotate min / max of each group with config label
    best_idx = int(np.argmax(perfs))
    best_x, best_y = xs[best_idx], perfs[best_idx]
    ax.annotate(CONFIGS[best_idx],
                xy=(best_x, best_y), xytext=(6, 4),
                textcoords="offset points", fontsize=7.5, color=s["color"],
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# ── efficiency iso-lines (25 %, 50 %, 75 %) ─────────────────────────────────
for frac, ls in [(0.50, ":"), (0.75, "--")]:
    eff_line = [ceiling(ai) * frac for ai in ai_range]
    ax.loglog(ai_range, eff_line, color="gray", lw=0.8, ls=ls, alpha=0.5)
    # label at AI=0.05
    label_ai = 0.04
    ax.text(label_ai, ceiling(label_ai) * frac * 1.05,
            f"{int(frac*100)}% eff.", fontsize=7, color="gray", alpha=0.7)

# ── formatting ───────────────────────────────────────────────────────────────
ax.set_xlabel("Arithmetic Intensity (FLOP / byte)", fontsize=12)
ax.set_ylabel("Performance (GFLOP/s)", fontsize=12)
ax.set_title("Roofline Model — mlsim-accelerator (Apple M-series, scalar C++)",
             fontsize=12, pad=10)
ax.set_xlim(5e-2, 1e2)
ax.set_ylim(0.5, PEAK_GFLOPS * 1.8)
ax.grid(True, which="both", alpha=0.18)
ax.legend(fontsize=9, loc="lower right", framealpha=0.85)

# annotate peak lines
ax.axhline(PEAK_GFLOPS, color="black", lw=0.8, ls=":", alpha=0.4)
ax.text(6e1, PEAK_GFLOPS * 1.04, f"peak {PEAK_GFLOPS:.0f} GFLOP/s",
        fontsize=8, color="gray", ha="right")

plt.tight_layout()
out = "roofline.png"
plt.savefig(out, dpi=150)
print(f"Saved {out}")
plt.show()
