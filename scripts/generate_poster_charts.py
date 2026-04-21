"""
Generate all poster charts from real benchmark data.
Outputs PNG files into: poster_charts/
"""

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

OUT = os.path.join(os.path.dirname(__file__), "..", "poster_charts")
os.makedirs(OUT, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
with open(os.path.join(os.path.dirname(__file__), "..", "frontend", "public", "benchmark_data.json")) as f:
    data = json.load(f)

configs = data["configs"]          # list of 3 dicts (deg2, deg3, deg4)
raw     = data["raw_results"]      # dict: deg2/deg3/deg4 → list of per-image rows

deg2_raw = raw["deg2"]
deg3_raw = raw["deg3"]
deg4_raw = raw["deg4"]

PALETTE = {
    "deg2": "#4f46e5",   # indigo
    "deg3": "#f59e0b",   # amber
    "deg4": "#ef4444",   # red
    "bg":   "#f8f7f4",
}

# ── FIGURE 1: Accuracy bar chart ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
fig.patch.set_facecolor(PALETTE["bg"])
ax.set_facecolor(PALETTE["bg"])

labels  = ["x² (deg 2)", "x³ (deg 3)", "x⁴ (deg 4)"]
acc     = [c["accuracy_pct"] for c in configs]
colors  = [PALETTE["deg2"], PALETTE["deg3"], PALETTE["deg4"]]
bars    = ax.bar(labels, acc, color=colors, width=0.5, zorder=3)

for bar, v in zip(bars, acc):
    ax.text(bar.get_x() + bar.get_width()/2, v + 1.5,
            f"{v:.0f}%", ha="center", va="bottom", fontweight="bold", fontsize=13)

ax.set_ylim(0, 115)
ax.set_ylabel("Encrypted Accuracy (%)", fontsize=11)
ax.set_title("Accuracy by Activation Degree\n(10 images, 128-bit BFV)", fontsize=12, fontweight="bold")
ax.axhline(100, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_yticks([0, 25, 50, 75, 100])
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.spines[["top","right"]].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/fig1_accuracy_bar.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ fig1_accuracy_bar.png")

# ── FIGURE 2: Per-layer stacked bar (deg 2 only) ─────────────────────────────
cfg2 = configs[0]["layer_averages"]
layer_names  = ["Encrypt", "Conv1", "Act1", "Pool1", "Conv2", "Act2", "Pool2", "FC", "Decrypt"]
layer_keys   = ["encryption_ms","conv1_ms","act1_ms","pool1_ms","conv2_ms","act2_ms","pool2_ms","fc_ms","decryption_ms"]
layer_values = [cfg2[k] for k in layer_keys]

# group colours
group_colors = ["#06b6d4","#4f46e5","#a78bfa","#818cf8",
                "#4f46e5","#a78bfa","#818cf8","#f43f5e","#06b6d4"]

fig, ax = plt.subplots(figsize=(8, 4))
fig.patch.set_facecolor(PALETTE["bg"])
ax.set_facecolor(PALETTE["bg"])

bars = ax.bar(layer_names, layer_values, color=group_colors, zorder=3, width=0.6)
for bar, v in zip(bars, layer_values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 30,
            f"{v:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_ylabel("Average Time (ms)", fontsize=11)
ax.set_title("Per-Layer Latency Breakdown — x² Activation, 128-bit Security\n(avg over 10 images, AWS EC2 r6i.large)", fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.spines[["top","right"]].set_visible(False)
plt.xticks(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_layer_breakdown.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ fig2_layer_breakdown.png")

# ── FIGURE 3: Stacked horizontal bar showing proportion of total ─────────────
fig, ax = plt.subplots(figsize=(8, 2.2))
fig.patch.set_facecolor(PALETTE["bg"])
ax.set_facecolor(PALETTE["bg"])

total = sum(layer_values)
left  = 0
for name, val, color in zip(layer_names, layer_values, group_colors):
    pct = val / total * 100
    ax.barh(0, pct, left=left, color=color, height=0.5)
    if pct > 3:
        ax.text(left + pct/2, 0, f"{name}\n{pct:.1f}%",
                ha="center", va="center", fontsize=8, fontweight="bold", color="white")
    left += pct

ax.set_xlim(0, 100)
ax.set_yticks([])
ax.set_xlabel("% of Total Inference Time", fontsize=10)
ax.set_title("Time Composition — x² Activation (avg ~13.9 s total)", fontsize=11, fontweight="bold")
ax.spines[["top","right","left"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_time_composition.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ fig3_time_composition.png")

# ── FIGURE 4: Per-image total_ms scatter, all 3 degrees ──────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
fig.patch.set_facecolor(PALETTE["bg"])
ax.set_facecolor(PALETTE["bg"])

for key, rows, color, label in [
    ("deg2", deg2_raw, PALETTE["deg2"], "x² deg 2"),
    ("deg3", deg3_raw, PALETTE["deg3"], "x³ deg 3"),
    ("deg4", deg4_raw, PALETTE["deg4"], "x⁴ deg 4"),
]:
    xs = [r["image_index"] for r in rows]
    ys = [r["total_ms"]/1000 for r in rows]
    ax.plot(xs, ys, marker="o", color=color, label=label, linewidth=1.5)

ax.set_xlabel("Image Index (0–9)", fontsize=11)
ax.set_ylabel("Total Inference Time (s)", fontsize=11)
ax.set_title("Inference Time per Image by Activation Degree\n(128-bit BFV, AWS EC2 r6i.large)", fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_inference_scatter.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ fig4_inference_scatter.png")

# ── FIGURE 5: Confusion matrix — deg 2 (10 images, one per digit) ────────────
true2      = [r["true_label"]  for r in deg2_raw]
pred2      = [r["predicted"]   for r in deg2_raw]
conf2      = np.zeros((10, 10), dtype=int)
for t, p in zip(true2, pred2):
    conf2[t][p] += 1

fig, ax = plt.subplots(figsize=(5, 4.5))
fig.patch.set_facecolor(PALETTE["bg"])
sns.heatmap(conf2, annot=True, fmt="d", cmap="Blues",
            xticklabels=range(10), yticklabels=range(10),
            linewidths=0.5, linecolor="#ddd",
            cbar_kws={"shrink": 0.7}, ax=ax)
ax.set_xlabel("Predicted Label", fontsize=11)
ax.set_ylabel("True Label", fontsize=11)
ax.set_title("Confusion Matrix — x² Activation\n(10 images, 128-bit BFV)", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_confusion_deg2.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ fig5_confusion_deg2.png")

# ── FIGURE 6: Confusion matrix — deg 3 ───────────────────────────────────────
true3 = [r["true_label"] for r in deg3_raw]
pred3 = [r["predicted"]  for r in deg3_raw]
conf3 = np.zeros((10, 10), dtype=int)
for t, p in zip(true3, pred3):
    conf3[t][p] += 1

fig, ax = plt.subplots(figsize=(5, 4.5))
fig.patch.set_facecolor(PALETTE["bg"])
sns.heatmap(conf3, annot=True, fmt="d", cmap="Oranges",
            xticklabels=range(10), yticklabels=range(10),
            linewidths=0.5, linecolor="#ddd",
            cbar_kws={"shrink": 0.7}, ax=ax)
ax.set_xlabel("Predicted Label", fontsize=11)
ax.set_ylabel("True Label", fontsize=11)
ax.set_title("Confusion Matrix — x³ Activation\n(10 images, 128-bit BFV)", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/fig6_confusion_deg3.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ fig6_confusion_deg3.png")

# ── FIGURE 7: Confusion matrix — deg 4 ───────────────────────────────────────
true4 = [r["true_label"] for r in deg4_raw]
pred4 = [r["predicted"]  for r in deg4_raw]
conf4 = np.zeros((10, 10), dtype=int)
for t, p in zip(true4, pred4):
    conf4[t][p] += 1

fig, ax = plt.subplots(figsize=(5, 4.5))
fig.patch.set_facecolor(PALETTE["bg"])
sns.heatmap(conf4, annot=True, fmt="d", cmap="Reds",
            xticklabels=range(10), yticklabels=range(10),
            linewidths=0.5, linecolor="#ddd",
            cbar_kws={"shrink": 0.7}, ax=ax)
ax.set_xlabel("Predicted Label", fontsize=11)
ax.set_ylabel("True Label", fontsize=11)
ax.set_title("Confusion Matrix — x⁴ Activation\n(10 images, 128-bit BFV)", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/fig7_confusion_deg4.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ fig7_confusion_deg4.png")

# ── FIGURE 8: Side-by-side accuracy comparison bar ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)
fig.patch.set_facecolor(PALETTE["bg"])
fig.suptitle("Per-Digit Encrypted Accuracy by Activation Degree\n(1 image per digit, 128-bit BFV)", fontsize=12, fontweight="bold")

for ax, cfg, color, title in zip(
    axes,
    configs,
    [PALETTE["deg2"], PALETTE["deg3"], PALETTE["deg4"]],
    ["x² (deg 2)", "x³ (deg 3)", "x⁴ (deg 4)"]
):
    ax.set_facecolor(PALETTE["bg"])
    digits = list(range(10))
    pcts   = [cfg["per_digit_accuracy"][str(d)]["pct"] for d in digits]
    bar_colors = ["#22c55e" if p == 100 else "#ef4444" for p in pcts]
    ax.bar(digits, pcts, color=bar_colors, width=0.6, zorder=3)
    ax.set_xticks(digits)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Digit", fontsize=10)
    ax.set_ylim(0, 115)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top","right"]].set_visible(False)
    correct_patch = mpatches.Patch(color="#22c55e", label="Correct")
    wrong_patch   = mpatches.Patch(color="#ef4444", label="Wrong")
    ax.legend(handles=[correct_patch, wrong_patch], fontsize=8)

axes[0].set_ylabel("Accuracy (%)", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT}/fig8_per_digit_accuracy.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ fig8_per_digit_accuracy.png")

# ── FIGURE 9: Avg time comparison across degrees (grouped bar) ───────────────
fig, ax = plt.subplots(figsize=(7, 4))
fig.patch.set_facecolor(PALETTE["bg"])
ax.set_facecolor(PALETTE["bg"])

x      = np.arange(len(layer_names))
width  = 0.25

for i, (cfg, color, label) in enumerate(zip(
    configs,
    [PALETTE["deg2"], PALETTE["deg3"], PALETTE["deg4"]],
    ["x² deg 2", "x³ deg 3", "x⁴ deg 4"]
)):
    vals = [cfg["layer_averages"][k] for k in layer_keys]
    ax.bar(x + i*width, vals, width, label=label, color=color, zorder=3, alpha=0.9)

ax.set_xticks(x + width)
ax.set_xticklabels(layer_names, fontsize=9)
ax.set_ylabel("Avg Time (ms)", fontsize=11)
ax.set_title("Per-Layer Timing: All Activation Degrees\n(128-bit BFV, 10 images each)", fontsize=11, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3, zorder=0)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/fig9_layer_comparison_grouped.png", dpi=180, bbox_inches="tight")
plt.close()
print("✓ fig9_layer_comparison_grouped.png")

print(f"\n✅  All charts saved to: {os.path.abspath(OUT)}/")
