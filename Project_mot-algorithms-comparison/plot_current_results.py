import json
import matplotlib.pyplot as plt

RESULTS_FILE = "results.json"

with open(RESULTS_FILE, "r", encoding="utf-8") as f:
    results = json.load(f)

algorithms = ["JPDA", "GM-PHD", "PMBM-lite"]

precision = [results[a]["precision"] for a in algorithms]
recall    = [results[a]["recall"] for a in algorithms]
f1        = [results[a]["f1"] for a in algorithms]
symkl     = [results[a]["mean_symmetric_KL"] for a in algorithms]

# 1. F1 bar chart
plt.figure(figsize=(8, 5))
plt.bar(algorithms, f1)
plt.title("F1-score Comparison")
plt.xlabel("Algorithm")
plt.ylabel("F1-score")
plt.ylim(0, 1.0)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("plot_f1.png", dpi=200)
plt.show()

# 2. SymKL bar chart
plt.figure(figsize=(8, 5))
plt.bar(algorithms, symkl)
plt.title("Symmetric KL-Divergence Comparison")
plt.xlabel("Algorithm")
plt.ylabel("Symmetric KL (lower is better)")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("plot_symkl.png", dpi=200)
plt.show()

# 3. Grouped chart
x = range(len(algorithms))
width = 0.25

plt.figure(figsize=(10, 5))
plt.bar([i - width for i in x], precision, width=width, label="Precision")
plt.bar(x, recall, width=width, label="Recall")
plt.bar([i + width for i in x], f1, width=width, label="F1")

plt.xticks(list(x), algorithms)
plt.ylim(0, 1.1)
plt.title("Classical Tracking Metrics Comparison")
plt.xlabel("Algorithm")
plt.ylabel("Score")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("plot_classical_metrics.png", dpi=200)
plt.show()

# 4. Scatter plot
plt.figure(figsize=(8, 5))
plt.scatter(symkl, f1, s=120)

for i, name in enumerate(algorithms):
    plt.annotate(name, (symkl[i], f1[i]), xytext=(5, 5), textcoords="offset points")

plt.title("F1-score vs Symmetric KL")
plt.xlabel("Symmetric KL (lower is better)")
plt.ylabel("F1-score (higher is better)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("plot_f1_vs_symkl.png", dpi=200)
plt.show()

print("Saved plots:")
print("- plot_f1.png")
print("- plot_symkl.png")
print("- plot_classical_metrics.png")
print("- plot_f1_vs_symkl.png")