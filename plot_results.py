import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("classic")
plt.rcParams.update({
    "figure.figsize": (8, 6),
    "font.size": 12,
    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "axes.labelsize": 14,
    "axes.titlesize": 16,

    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.2,
    "axes.spines.top": False,
    "axes.spines.right": False,

    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.major.size": 5,
    "xtick.major.width": 1,
    "ytick.major.size": 5,
    "ytick.major.width": 1,

    "axes.grid": True,
    "grid.color": "gray",
    "grid.linestyle": ":",
    "grid.alpha": 0.7,
})

df = pd.read_csv("mmlu_results.csv")

os.makedirs("plots", exist_ok=True)

subjects = df["subject"].unique()

for subject in subjects:
    sub_df = (
        df[df["subject"] == subject]
        .groupby("noise_std", as_index=False)
        .agg(
            mean_accuracy=("accuracy", "mean"),
            std_accuracy=("accuracy", "std"),
            count=("accuracy", "count")
        )
    )

    sub_df["ci95"] = 1.96 * (sub_df["std_accuracy"] / np.sqrt(sub_df["count"]))

    x = sub_df["noise_std"]
    y = sub_df["mean_accuracy"]
    yerr = sub_df["ci95"]

    if subject == "ALL_FILTERED":
        subject_cleaned = "Aggregated over 5 Subjects"
        subject_for_file = "ALL_AGGREGATED"
    else:
        subject_cleaned = subject.replace("_", " ").title()
        subject_for_file = subject

    fig, ax = plt.subplots()

    ax.errorbar(
        x,
        y,
        yerr=yerr,
        fmt="-o",
        linewidth=1.5,
        markersize=5,
        capsize=4,
        capthick=1.2
    )

    ax.set_title(f"Accuracy on MMLU {subject_cleaned}\n vs Noise Standard Deviation")
    ax.set_xlabel("Noise Standard Deviation")
    ax.set_ylabel("Mean Accuracy")

    min_y = (y - yerr).min()
    max_y = (y + yerr).max()
    margin = 0.02
    lower_lim = max(0, min_y - margin)
    upper_lim = min(1, max_y + margin)
    ax.set_ylim(lower_lim, upper_lim)

    fig.tight_layout()

    fig.savefig(f"plots/{subject_for_file}_accuracy_vs_noise.png", dpi=300)
    plt.close(fig)
