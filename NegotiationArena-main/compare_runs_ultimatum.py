#!/usr/bin/env python3
"""
Generate comparative bar graphs from multiple behavior_summary CSV files.

This script reads multiple behavior_summary CSVs from different runs,
aliases "English" to "Baseline", and generates four separate bar graphs
comparing the runs side-by-side for each behavior.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Font settings
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
    }
)

# Define the inputs here as a dict: {"Label": "path/to/behavior_summary.csv"}
INPUT_FILES = {
    "Original": "./.logs/ultimatum_social_behavior_newruns1to20/behavior_summary.csv",
    "A1": "./.logs/ultimatum_promptablation/behavior_summary_promptnative.csv",
    "A2": "./.logs/ultimatum_promptablation2/behavior_summary_promptnative.csv",
    "A3": "./.logs/ultimatum_prompt_native/behavior_summary_promptnative.csv",
    # Add more runs here:
    # "Run B": "./path/to/another/behavior_summary.csv"
}


def load_data(files_dict):
    """
    Load data from multiple CSV files.
    Aliases 'Baseline' to 'English' in the Language column.
    Returns a combined DataFrame.
    """
    df_list = []
    for label, path in files_dict.items():
        if not os.path.exists(path):
            print(f"Warning: File not found: {path}")
            continue

        df = pd.read_csv(path)
        # Rename column if necessary
        if "behavior" in df.columns and "Language" not in df.columns:
            df = df.rename(columns={"behavior": "Language"})

        # Alias Baseline -> English
        df["Language"] = df["Language"].replace({"Baseline": "English"})

        df["Run"] = label
        df_list.append(df)

    if not df_list:
        raise ValueError("No valid input files found.")

    return pd.concat(df_list, ignore_index=True)


def plot_metric(
    ax, df, metric_col, title, ylabel="", is_percentage=False, runs=None, behaviors=None
):
    """
    Plots a grouped bar chart for a single metric on the given axes.
    """
    n_runs = len(runs)
    n_behaviors = len(behaviors)

    x = np.arange(n_behaviors)
    width = 0.8 / n_runs  # Total width dedicated to bars at each tick

    colors = ["#4C4CB8", "#2E8BC0", "#18A162", "#6AD34F", "#D5E86B", "#B7E06C"]

    for i, run in enumerate(runs):
        run_data = df[df["Run"] == run]
        # Ensure data is ordered by behaviors
        run_data = run_data.set_index("Language").reindex(behaviors).reset_index()

        vals = run_data[metric_col].fillna(0).values

        # Calculate bar positions
        pos = x - (0.8 / 2) + (i * width) + (width / 2)

        bars = ax.bar(
            pos,
            vals,
            width,
            label=run,
            color=colors[i % len(colors)],
            edgecolor="black",
        )

        # Add labels above bars
        for bar, val in zip(bars, vals):
            height = bar.get_height()
            if is_percentage:
                text = f"{val * 100:.1f}%"
            else:
                text = f"{val:.1f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + (ax.get_ylim()[1] * 0.02),
                text,
                ha="center",
                va="bottom",
                fontsize=10,
                rotation=45 if n_runs > 2 else 0,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(behaviors, rotation=30, ha="right")
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    if is_percentage:
        ax.set_ylim(0, 1.1)
    else:
        # Give some headroom for labels
        ymax = df[metric_col].max()
        ax.set_ylim(0, ymax * 1.15)

    ax.legend(title="Runs", bbox_to_anchor=(1.05, 1), loc="upper left")


def main():
    parser = argparse.ArgumentParser(
        description="Generate comparative bar graphs from run CSVs."
    )
    parser.add_argument(
        "--outdir", default="./comparisons", help="Output directory for the graphs"
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Loading data...")
    df = load_data(INPUT_FILES)

    runs = list(INPUT_FILES.keys())
    # Identify common behaviors across all runs, or just unique behaviors
    behaviors = sorted(df["Language"].unique())

    # 1. Average Acceptance Rate
    print("Generating Acceptance Rate graph...")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_metric(
        ax,
        df,
        "acceptance_rate_mean",
        "Average Acceptance Rate by Behavior",
        is_percentage=True,
        runs=runs,
        behaviors=behaviors,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(args.outdir, "acceptance_rate_comparison.png"), dpi=600)
    plt.close()

    # 2. Average Initial Offer
    print("Generating Initial Offer graph...")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_metric(
        ax,
        df,
        "initial_offer_mean",
        "Average Initial Offer by Behavior",
        runs=runs,
        behaviors=behaviors,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(args.outdir, "initial_offer_comparison.png"), dpi=600)
    plt.close()

    # 3. Win Rate (Player 1)
    print("Generating Win Rate graph...")
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_metric(
        ax,
        df,
        "p1_win_rate_mean",
        "Average Win Rate (Player 1) by Behavior",
        is_percentage=True,
        runs=runs,
        behaviors=behaviors,
    )
    plt.tight_layout()
    fig.savefig(os.path.join(args.outdir, "win_rate_comparison.png"), dpi=600)
    plt.close()

    # 4. Average Payoffs
    # Payoffs is slightly more complex as it plots P1 and P2 side-by-side for each run inside each behavior
    print("Generating Payoffs graph...")
    n_runs = len(runs)
    n_behaviors = len(behaviors)
    x = np.arange(n_behaviors)

    # We need 2 bars (P1, P2) per run per behavior
    total_bars_per_behavior = n_runs * 2
    width = 0.8 / total_bars_per_behavior

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, run in enumerate(runs):
        run_data = (
            df[df["Run"] == run].set_index("Language").reindex(behaviors).reset_index()
        )
        p1_vals = run_data["p1_payoff_mean"].fillna(0).values
        p2_vals = run_data["p2_payoff_mean"].fillna(0).values

        # Positions
        # Center of the cluster for this run
        run_center = x - (0.8 / 2) + (i * 2 * width) + width

        pos_p1 = run_center - (width / 2)
        pos_p2 = run_center + (width / 2)

        # Different colors for P1 and P2, maybe varied slightly by run if desired.
        # For simplicity, use standard colors, but label them with run names
        color_p1 = "#2E86AB" if i % 2 == 0 else "#1F5B73"
        color_p2 = "#A23E48" if i % 2 == 0 else "#702A31"

        bars1 = ax.bar(
            pos_p1,
            p1_vals,
            width,
            label=f"P1 ({run})",
            color=color_p1,
            edgecolor="black",
        )
        bars2 = ax.bar(
            pos_p2,
            p2_vals,
            width,
            label=f"P2 ({run})",
            color=color_p2,
            edgecolor="black",
        )

        for bar, val in zip(bars1, p1_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 1,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

        for bar, val in zip(bars2, p2_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 1,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(behaviors, rotation=30, ha="right")
    ax.set_title("Average Payoffs by Behavior")
    ax.set_ylim(0, df[["p1_payoff_mean", "p2_payoff_mean"]].max().max() * 1.25)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    fig.savefig(os.path.join(args.outdir, "payoffs_comparison.png"), dpi=600)
    plt.close()

    print(f"Done! Graphs saved to {args.outdir}")


if __name__ == "__main__":
    main()
