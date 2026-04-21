"""
Generate library comparison charts (SEAL vs HELib vs OpenFHE)
from the documented benchmark results in docs/grpc-api.md.
Outputs PNG files into: poster_charts/
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

OUT = os.path.join(os.path.dirname(__file__), "..", "poster_charts")
os.makedirs(OUT, exist_ok=True)

# ── Real data from docs/grpc-api.md (20-operation run, n=4096, 128-bit) ──────
LIBRARIES = ["SEAL\n(BFV)", "HELib\n(BGV)", "OpenFHE\n(BFV)"]
LIB_LABELS = ["SEAL", "HELib", "OpenFHE"]
COLORS     = ["#3b82f6", "#c084fc", "#34d399"]   # blue, purple, green

data = {
    "Key Generation":      [5.09,  96.58, 25.01],
    "Encryption (per op)": [1.03,   1.46,  3.94],
    "Addition (per op)":   [0.02,   0.07,  0.01],
    "Multiplication (per op)": [2.67, 10.52, 13.49],
    "Decryption (per op)": [0.24,   2.17,  1.89],
    "Total (20 ops)":      [82.55, 370.42, 403.75],
}

# ── FIGURE LIB-1: Grouped bar — all operations ───────────────────────────────
ops   = list(data.keys())
x     = np.arange(len(ops))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 5))
fig.patch.set_facecolor("#f8f7f4")
ax.set_facecolor("#f8f7f4")

for i, (lib, color) in enumerate(zip(LIB_LABELS, COLORS)):
    vals = [data[op][i] for op in ops]
    bars = ax.bar(x + i*width, vals, width, label=lib, color=color, zorder=3, alpha=0.92)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.5,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax.set_xticks(x + width)
ax.set_xticklabels(ops, fontsize=9)
ax.set_ylabel("Time (ms)", fontsize=11)
ax.set_title("HE Library Micro-Benchmark Comparison\nSEAL vs HELib vs OpenFHE  ·  n = 4096  ·  128-bit security  ·  20 operations",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/lib1_grouped_bar_all_ops.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ lib1_grouped_bar_all_ops.png")

# ── FIGURE LIB-2: Split — key gen (large) vs operations (small) ──────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                gridspec_kw={"width_ratios": [1, 2]})
fig.patch.set_facecolor("#f8f7f4")
fig.suptitle("HE Library Comparison: SEAL · HELib · OpenFHE  (n=4096, 128-bit)", fontsize=12, fontweight="bold")

# Left: key gen only
ax1.set_facecolor("#f8f7f4")
bars = ax1.bar(LIB_LABELS, data["Key Generation"], color=COLORS, zorder=3, width=0.5)
for bar, v in zip(bars, data["Key Generation"]):
    ax1.text(bar.get_x() + bar.get_width()/2, v + 1.5,
             f"{v:.1f} ms", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax1.set_title("Key Generation", fontsize=11, fontweight="bold")
ax1.set_ylabel("Time (ms)", fontsize=10)
ax1.grid(axis="y", alpha=0.3, zorder=0)
ax1.spines[["top", "right"]].set_visible(False)

# Right: per-operation costs (grouped)
ax2.set_facecolor("#f8f7f4")
per_ops  = ["Encryption", "Addition", "Multiplication", "Decryption"]
per_keys = ["Encryption (per op)", "Addition (per op)",
            "Multiplication (per op)", "Decryption (per op)"]
xr = np.arange(len(per_ops))
for i, (lib, color) in enumerate(zip(LIB_LABELS, COLORS)):
    vals = [data[k][i] for k in per_keys]
    bars2 = ax2.bar(xr + i*0.25, vals, 0.25, label=lib, color=color, zorder=3, alpha=0.92)
    for bar, v in zip(bars2, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.08,
                 f"{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax2.set_xticks(xr + 0.25)
ax2.set_xticklabels(per_ops, fontsize=10)
ax2.set_title("Per-Operation Cost (ms/op)", fontsize=11, fontweight="bold")
ax2.set_ylabel("Time (ms/op)", fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(axis="y", alpha=0.3, zorder=0)
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/lib2_split_keygen_vs_ops.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ lib2_split_keygen_vs_ops.png")

# ── FIGURE LIB-3: Total time horizontal bar ──────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 2.8))
fig.patch.set_facecolor("#f8f7f4")
ax.set_facecolor("#f8f7f4")

totals = [82.55, 370.42, 403.75]
bars = ax.barh(LIB_LABELS, totals, color=COLORS, height=0.45, zorder=3)
for bar, v in zip(bars, totals):
    ax.text(v + 5, bar.get_y() + bar.get_height()/2,
            f"{v:.1f} ms", va="center", fontsize=11, fontweight="bold")

ax.set_xlabel("Total Time for 20 Operations (ms)", fontsize=10)
ax.set_title("Total Benchmark Time — 20 ops, n=4096, 128-bit Security", fontsize=11, fontweight="bold")
ax.set_xlim(0, 480)
ax.grid(axis="x", alpha=0.3, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/lib3_total_time_hbar.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ lib3_total_time_hbar.png")

# ── FIGURE LIB-4: Log-scale comparison — shows all ops without clipping ──────
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#f8f7f4")
ax.set_facecolor("#f8f7f4")

all_ops   = ["Key Gen", "Encrypt", "Add", "Multiply", "Decrypt"]
all_keys  = ["Key Generation", "Encryption (per op)", "Addition (per op)",
             "Multiplication (per op)", "Decryption (per op)"]
xa = np.arange(len(all_ops))

for i, (lib, color) in enumerate(zip(LIB_LABELS, COLORS)):
    vals = [data[k][i] for k in all_keys]
    ax.bar(xa + i*0.25, vals, 0.25, label=lib, color=color, zorder=3, alpha=0.92)

ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g} ms"))
ax.set_xticks(xa + 0.25)
ax.set_xticklabels(all_ops, fontsize=10)
ax.set_ylabel("Time — log scale (ms)", fontsize=11)
ax.set_title("Library Comparison — Log Scale\n(shows additions + multiplications without Key Gen dominating)",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3, which="both", zorder=0)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/lib4_log_scale_comparison.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ lib4_log_scale_comparison.png")

# ── FIGURE LIB-5: Radar chart ────────────────────────────────────────────────
# Normalise each metric 0→1 where 1 = fastest (lowest ms)
categories = ["Key Gen\n(fast=1)", "Encrypt\n(fast=1)", "Add\n(fast=1)",
              "Multiply\n(fast=1)", "Decrypt\n(fast=1)"]
raw_vals = {
    "SEAL":    [5.09,  1.03, 0.02, 2.67,  0.24],
    "HELib":   [96.58, 1.46, 0.07, 10.52, 2.17],
    "OpenFHE": [25.01, 3.94, 0.01, 13.49, 1.89],
}
# Normalise: score = min/value  (1.0 = best)
maxvals = [max(raw_vals[l][i] for l in raw_vals) for i in range(5)]
scores  = {lib: [1 - (raw_vals[lib][i] / maxvals[i]) + (raw_vals[lib][i] / maxvals[i]) * 0
                 for i in range(5)] for lib in raw_vals}
# Simpler: score = 1 - (v - min)/(max - min)   →  1 = fastest, 0 = slowest
minvals = [min(raw_vals[l][i] for l in raw_vals) for i in range(5)]
for lib in raw_vals:
    scores[lib] = [1 - (raw_vals[lib][i] - minvals[i]) / max(maxvals[i] - minvals[i], 1e-9)
                   for i in range(5)]

N  = len(categories)
angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(5.5, 5.5), subplot_kw=dict(polar=True))
fig.patch.set_facecolor("#f8f7f4")
ax.set_facecolor("#f0eee8")

for lib, color in zip(LIB_LABELS, COLORS):
    vals = scores[lib] + scores[lib][:1]
    ax.plot(angles, vals, color=color, linewidth=2.5, label=lib)
    ax.fill(angles, vals, color=color, alpha=0.15)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=9.5)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0 (best)"], fontsize=7, color="gray")
ax.set_title("Performance Radar\n(1.0 = fastest per operation)", fontsize=11, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=10)
ax.grid(color="gray", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/lib5_radar.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ lib5_radar.png")

# ── FIGURE LIB-6: Scheme comparison annotation table (as a figure) ───────────
fig, ax = plt.subplots(figsize=(9, 3))
fig.patch.set_facecolor("#f8f7f4")
ax.axis("off")

table_data = [
    ["Property",          "SEAL (BFV)",            "HELib (BGV)",             "OpenFHE (BFV)"],
    ["Made by",           "Microsoft Research",    "IBM Research",            "OpenFHE Consortium"],
    ["Scheme",            "BFV",                   "BGV",                     "BFV / CKKS / TFHE"],
    ["Ring dimension n",  "4,096",                 "~4,096",                  "4,096"],
    ["Security",          "128-bit",               "~128-bit",                "128-bit"],
    ["SIMD acceleration", "None",                  "Partial",                 "AVX2 / AVX512"],
    ["Plaintext slots",   "4,096",                 "1 (per ciphertext)",      "4,096"],
    ["Key Gen",           "5.1 ms  [fastest]",     "96.6 ms  [slowest]",      "25.0 ms"],
    ["Multiplication",    "2.7 ms  [fastest]",     "10.5 ms",                 "13.5 ms  [slowest]"],
    ["Addition",          "0.02 ms",               "0.07 ms",                 "0.01 ms  [fastest]"],
    ["Best use case",     "Encryption-heavy",      "Deep circuits (BGV)",     "General purpose"],
]

col_colors = [["#e8e4f4"]*4] + [["#f8f7f4", "#dbeafe", "#f3e8ff", "#dcfce7"]
                                  for _ in range(len(table_data)-1)]
tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
               cellLoc="center", loc="center",
               cellColours=col_colors[1:],
               colColours=["#e2e8f0"]*4)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1, 1.45)
ax.set_title("HE Library Comparison Summary  ·  n=4096  ·  128-bit Security",
             fontsize=11, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig(f"{OUT}/lib6_summary_table.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ lib6_summary_table.png")

print(f"\n✅  All library charts saved to: {os.path.abspath(OUT)}/")
