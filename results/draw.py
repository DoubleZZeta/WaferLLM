"""Plot perplexity and projection speedup, matched by retained-work ratio.

Run: python3 results/draw.py
Add --show to open the figure, or --output PATH to choose the saved image.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = Path(__file__).resolve().parent


def load_results():
    accuracy = pd.read_csv(RESULTS_DIR / "accuracy.csv")
    performance = pd.read_csv(RESULTS_DIR / "performance.csv")
    accuracy = accuracy.loc[
        (accuracy["status"] == "complete") & (accuracy["after_status"] == "ok"),
        ["ratio", "after_ppl"],
    ]
    performance = performance.loc[
        performance["status"] == "ok",
        ["target_retained_work_ratio", "fused_speedup_vs_dense"],
    ].rename(columns={"target_retained_work_ratio": "ratio"})

    # CSV row order differs between files. Join on ratio instead of position.
    for frame in (accuracy, performance):
        frame["ratio"] = pd.to_numeric(frame["ratio"], errors="raise").round(8)
        if frame["ratio"].isna().any() or not frame["ratio"].between(0, 1, inclusive="right").all():
            raise ValueError("Retained-work ratios must be numeric and in (0, 1].")
        if frame["ratio"].duplicated().any():
            raise ValueError("Multiple successful rows per ratio: select one experiment before plotting.")

    data = accuracy.merge(performance, on="ratio", how="outer", validate="one_to_one", indicator=True)
    unmatched = data.loc[data["_merge"] != "both", "ratio"].tolist()
    if unmatched:
        raise ValueError(f"Missing successful accuracy or performance results for ratios: {unmatched}")
    data = data.drop(columns="_merge").sort_values("ratio")
    if data.empty:
        raise ValueError("No successful results to plot.")
    for column in ("after_ppl", "fused_speedup_vs_dense"):
        data[column] = pd.to_numeric(data[column], errors="raise")
        if not np.isfinite(data[column]).all() or (data[column] <= 0).any():
            raise ValueError(f"{column} must contain finite positive values.")
    data["negative_ppl"] = -data["after_ppl"]
   
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "accuracy_speedup.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    data = load_results()

    fig, accuracy_ax = plt.subplots(figsize=(10, 6))
    speedup_ax = accuracy_ax.twinx()
    accuracy_line, = accuracy_ax.plot(
        data["ratio"], data["negative_ppl"], marker="o", linewidth=2,
        label="Negative perplexity", color="tab:blue",
    )
    speedup_line, = speedup_ax.plot(
        data["ratio"], data["fused_speedup_vs_dense"], marker="s", linewidth=2,
        label="Projection speedup", color="tab:orange",
    )
    baseline_performance = speedup_ax.axhline(1, color="tab:orange", linestyle="--", label="Dense baseline (1×)")
    baseline_accuracy = accuracy_ax.axhline( (-5.472095653658307), color="tab:blue", linestyle="--", label="Dense baseline (-5.47 ppl)")
    accuracy_ax.set_xlabel("Target retained-work ratio")
    accuracy_ax.set_xticks(data["ratio"].tolist())
    accuracy_ax.set_ylabel("Perplexity", color="tab:blue")
    speedup_ax.set_ylabel("Speedup over dense", color="tab:orange")
    accuracy_ax.tick_params(axis="y", labelcolor="tab:blue")
    speedup_ax.tick_params(axis="y", labelcolor="tab:orange")
    accuracy_ax.grid(alpha=0.25)
    accuracy_ax.set_title("NegativePerplexity and Projection Speedup")
    fig.legend(handles=[accuracy_line, speedup_line, baseline_performance, baseline_accuracy],
               loc="upper center", bbox_to_anchor=(0.5, 0.92), ncol=3, frameon=False)
    fig.text(0.5, 0.025, "Performance uses the summed cycles of seven linear projections.",
             ha="center", fontsize=10, color="0.35")
    fig.tight_layout(rect=(0, 0.06, 1, 0.87))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(data.to_string(index=False))
    print(f"Saved {args.output}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
