#!/usr/bin/env python3
"""
Custom Heatmap Generator for Social Behavior Test Results

This script creates customizable heatmaps similar to academic research papers,
allowing for fine-tuned control over visualization appearance and style.
"""

import json
import sys
from pathlib import Path

# Add current directory to Python path for module imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle


class CustomHeatmapGenerator:
    def __init__(self, results_file, output_dir=None):
        self.results_file = Path(results_file)
        self.output_dir = Path(output_dir) if output_dir else self.results_file.parent
        self.results = []
        self.metrics = {}

    def load_results(self):
        """Load results from JSON file"""
        with open(self.results_file, "r") as f:
            self.results = json.load(f)

        # Filter completed games only
        self.results = [r for r in self.results if r.get("game_completed", False)]
        print(f"Loaded {len(self.results)} completed games")

    def calculate_metrics(self):
        """Calculate comprehensive metrics"""
        df = pd.DataFrame(self.results)

        models = sorted(df["model1"].unique())
        behaviors = sorted(df["behavior"].unique())

        self.metrics = {}

        for behavior in behaviors:
            behavior_df = df[df["behavior"] == behavior]

            # Initialize matrices
            win_rates = pd.DataFrame(index=models, columns=models, dtype=float)
            payoffs_p1 = pd.DataFrame(index=models, columns=models, dtype=float)
            payoffs_p2 = pd.DataFrame(index=models, columns=models, dtype=float)
            acceptance_rates = pd.DataFrame(index=models, columns=models, dtype=float)

            # Initialize all cells to NaN (for same model vs same model)
            win_rates[:] = np.nan
            payoffs_p1[:] = np.nan
            payoffs_p2[:] = np.nan
            acceptance_rates[:] = np.nan

            for m1 in models:
                for m2 in models:
                    if m1 == m2:
                        # Skip same model vs same model - leave as NaN
                        continue

                    subset = behavior_df[
                        (behavior_df["model1"] == m1) & (behavior_df["model2"] == m2)
                    ]

                    if len(subset) > 0:
                        p1_res = pd.to_numeric(
                            subset["player1_final_resources"], errors="coerce"
                        ).fillna(0)
                        p2_res = pd.to_numeric(
                            subset["player2_final_resources"], errors="coerce"
                        ).fillna(0)

                        # Win rate (including ties as 0.5)
                        wins = (p1_res > p2_res).sum()
                        ties = (p1_res == p2_res).sum()
                        win_rate = (wins + 0.5 * ties) / len(subset)

                        win_rates.loc[m1, m2] = win_rate
                        payoffs_p1.loc[m1, m2] = p1_res.mean()
                        payoffs_p2.loc[m1, m2] = p2_res.mean()

                        # Calculate acceptance rate (games where both players get something)
                        accepted = ((p1_res > 0) & (p2_res > 0)).sum()
                        acceptance_rates.loc[m1, m2] = (
                            accepted / len(subset) if len(subset) > 0 else 0
                        )
                    else:
                        win_rates.loc[m1, m2] = 0.0
                        payoffs_p1.loc[m1, m2] = 0.0
                        payoffs_p2.loc[m1, m2] = 0.0
                        acceptance_rates.loc[m1, m2] = 0.0

            self.metrics[behavior] = {
                "win_rates": win_rates,
                "payoffs_p1": payoffs_p1,
                "payoffs_p2": payoffs_p2,
                "acceptance_rates": acceptance_rates,
            }

    def create_academic_style_heatmap(self, metric="win_rates", title_prefix=""):
        """Create publication-ready heatmap"""
        n_behaviors = len(self.metrics)

        # Set up the figure with academic styling
        plt.style.use("default")  # Clean style
        fig, axes = plt.subplots(1, n_behaviors, figsize=(5 * n_behaviors, 4))

        if n_behaviors == 1:
            axes = [axes]

        # Color schemes for different metrics
        color_schemes = {
            "win_rates": {"cmap": "Blues", "vmin": 0, "vmax": 1, "format": ".2f"},
            "payoffs_p1": {
                "cmap": "Blues",
                "vmin": None,
                "vmax": None,
                "format": ".1f",
            },
            "payoffs_p2": {
                "cmap": "Blues",
                "vmin": None,
                "vmax": None,
                "format": ".1f",
            },
            "acceptance_rates": {
                "cmap": "Greens",
                "vmin": 0,
                "vmax": 1,
                "format": ".2f",
            },
        }

        scheme = color_schemes.get(metric, color_schemes["win_rates"])

        for idx, (behavior, behavior_metrics) in enumerate(self.metrics.items()):
            data = behavior_metrics[metric]
            ax = axes[idx]

            # Create masked array for visualization
            masked_data = np.ma.masked_invalid(data.values)
            im = ax.imshow(
                masked_data,
                cmap=scheme["cmap"],
                vmin=scheme["vmin"],
                vmax=scheme["vmax"],
                aspect="auto",
            )

            # Add text annotations
            for i in range(len(data.index)):
                for j in range(len(data.columns)):
                    value = data.iloc[i, j]

                    if np.isnan(value):
                        # Gray out diagonal (same model vs same model)
                        ax.add_patch(
                            Rectangle(
                                (j - 0.5, i - 0.5),
                                1,
                                1,
                                facecolor="lightgray",
                                alpha=0.5,
                            )
                        )
                        ax.text(
                            j,
                            i,
                            "N/A",
                            ha="center",
                            va="center",
                            color="gray",
                            fontweight="bold",
                        )
                    else:
                        # Choose text color based on background
                        if scheme["vmin"] is not None and scheme["vmax"] is not None:
                            norm_value = (value - scheme["vmin"]) / (
                                scheme["vmax"] - scheme["vmin"]
                            )
                        else:
                            valid_values = data.values[~np.isnan(data.values)]
                            if len(valid_values) > 0:
                                data_min, data_max = (
                                    valid_values.min(),
                                    valid_values.max(),
                                )
                                norm_value = (
                                    (value - data_min) / (data_max - data_min)
                                    if data_max > data_min
                                    else 0
                                )
                            else:
                                norm_value = 0

                        text_color = "white" if norm_value > 0.5 else "black"

                        ax.text(
                            j,
                            i,
                            f"{value:{scheme['format']}}",
                            ha="center",
                            va="center",
                            color=text_color,
                            fontsize=12,
                            fontweight="bold",
                        )

            # Styling
            ax.set_title(
                f"{title_prefix}{behavior}", fontsize=14, fontweight="bold", pad=15
            )
            ax.set_xlabel("Player 2", fontsize=12, fontweight="bold")
            if idx == 0:
                ax.set_ylabel("Player 1", fontsize=12, fontweight="bold")

            # Set ticks and labels
            ax.set_xticks(range(len(data.columns)))
            ax.set_yticks(range(len(data.index)))
            ax.set_xticklabels(data.columns, fontsize=10)
            ax.set_yticklabels(data.index, fontsize=10)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.ax.tick_params(labelsize=10)

            # Set colorbar label
            metric_labels = {
                "win_rates": "Win Rate",
                "payoffs_p1": "Avg Payoff (P1)",
                "payoffs_p2": "Avg Payoff (P2)",
                "acceptance_rates": "Acceptance Rate",
            }
            cbar.set_label(
                metric_labels.get(metric, metric), fontsize=11, fontweight="bold"
            )

        plt.tight_layout()

        # Save with high DPI for publication
        output_file = self.output_dir / f"academic_{metric}_heatmap.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved: {output_file}")

        return fig

    def create_combined_academic_figure(self):
        """Create a combined figure similar to the reference image"""
        behaviors = list(self.metrics.keys())

        # Create figure with subplots
        fig = plt.figure(figsize=(12, 4 * len(behaviors)))

        for idx, behavior in enumerate(behaviors):
            # Win rates subplot
            ax1 = plt.subplot2grid((len(behaviors), 2), (idx, 0))
            win_data = self.metrics[behavior]["win_rates"]

            masked_win_data = np.ma.masked_invalid(win_data.values)
            im1 = ax1.imshow(
                masked_win_data, cmap="Blues", vmin=0, vmax=1, aspect="auto"
            )

            # Add annotations
            for i in range(len(win_data.index)):
                for j in range(len(win_data.columns)):
                    value = win_data.iloc[i, j]
                    if np.isnan(value):
                        ax1.add_patch(
                            Rectangle(
                                (j - 0.5, i - 0.5),
                                1,
                                1,
                                facecolor="lightgray",
                                alpha=0.5,
                            )
                        )
                        ax1.text(
                            j,
                            i,
                            "N/A",
                            ha="center",
                            va="center",
                            color="gray",
                            fontsize=11,
                            fontweight="bold",
                        )
                    else:
                        color = "white" if value > 0.5 else "black"
                        ax1.text(
                            j,
                            i,
                            f"{value:.2f}",
                            ha="center",
                            va="center",
                            color=color,
                            fontsize=11,
                            fontweight="bold",
                        )

            ax1.set_title(f"{behavior} - Win Rate", fontsize=13, fontweight="bold")
            ax1.set_xlabel("Player 2", fontsize=11, fontweight="bold")
            ax1.set_ylabel("Player 1", fontsize=11, fontweight="bold")
            ax1.set_xticks(range(len(win_data.columns)))
            ax1.set_yticks(range(len(win_data.index)))
            ax1.set_xticklabels(win_data.columns, fontsize=9)
            ax1.set_yticklabels(win_data.index, fontsize=9)

            # Payoffs subplot
            ax2 = plt.subplot2grid((len(behaviors), 2), (idx, 1))
            payoff_data = self.metrics[behavior]["payoffs_p1"]

            masked_payoff_data = np.ma.masked_invalid(payoff_data.values)
            im2 = ax2.imshow(masked_payoff_data, cmap="Blues", aspect="auto")

            # Add annotations
            for i in range(len(payoff_data.index)):
                for j in range(len(payoff_data.columns)):
                    value = payoff_data.iloc[i, j]
                    if np.isnan(value):
                        ax2.add_patch(
                            Rectangle(
                                (j - 0.5, i - 0.5),
                                1,
                                1,
                                facecolor="lightgray",
                                alpha=0.5,
                            )
                        )
                        ax2.text(
                            j,
                            i,
                            "N/A",
                            ha="center",
                            va="center",
                            color="gray",
                            fontsize=11,
                            fontweight="bold",
                        )
                    else:
                        # Normalize for color selection
                        valid_values = payoff_data.values[~np.isnan(payoff_data.values)]
                        if len(valid_values) > 0:
                            data_min, data_max = valid_values.min(), valid_values.max()
                            norm_val = (
                                (value - data_min) / (data_max - data_min)
                                if data_max > data_min
                                else 0
                            )
                        else:
                            norm_val = 0
                        color = "white" if norm_val > 0.5 else "black"
                        ax2.text(
                            j,
                            i,
                            f"{value:.1f}",
                            ha="center",
                            va="center",
                            color=color,
                            fontsize=11,
                            fontweight="bold",
                        )

            ax2.set_title(f"{behavior} - Payoff", fontsize=13, fontweight="bold")
            ax2.set_xlabel("Player 2", fontsize=11, fontweight="bold")
            if idx == 0:
                ax2.set_ylabel("Player 1", fontsize=11, fontweight="bold")
            ax2.set_xticks(range(len(payoff_data.columns)))
            ax2.set_yticks(range(len(payoff_data.index)))
            ax2.set_xticklabels(payoff_data.columns, fontsize=9)
            ax2.set_yticklabels(payoff_data.index, fontsize=9)

            # Add colorbars
            if idx == 0:  # Only add colorbars to top row
                cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
                cbar1.set_label("Win Rate", fontsize=10, fontweight="bold")
                cbar1.ax.tick_params(labelsize=9)

                cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
                cbar2.set_label("Average Payoff", fontsize=10, fontweight="bold")
                cbar2.ax.tick_params(labelsize=9)

        plt.tight_layout()

        # Save the combined figure
        output_file = self.output_dir / "combined_academic_heatmap.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved combined figure: {output_file}")

        return fig

    def create_difference_heatmap(self, model1, model2):
        """Create heatmap showing performance differences between two models"""
        fig, axes = plt.subplots(
            1, len(self.metrics), figsize=(5 * len(self.metrics), 4)
        )

        if len(self.metrics) == 1:
            axes = [axes]

        for idx, (behavior, behavior_metrics) in enumerate(self.metrics.items()):
            # Calculate difference in win rates when model1 is P1 vs model2 is P1
            win_rates = behavior_metrics["win_rates"]

            # Get performance of model1 vs model2 and vice versa
            perf_diff = pd.DataFrame(
                index=[f"{model1} adv", f"{model2} adv"], columns=["As P1", "As P2"]
            )

            # Model1 as P1 vs Model2 as P2
            if model1 in win_rates.index and model2 in win_rates.columns:
                perf_diff.loc[f"{model1} adv", "As P1"] = (
                    win_rates.loc[model1, model2] - 0.5
                )

            # Model1 as P2 vs Model2 as P1
            if model2 in win_rates.index and model1 in win_rates.columns:
                perf_diff.loc[f"{model1} adv", "As P2"] = (
                    1 - win_rates.loc[model2, model1]
                ) - 0.5

            # Model2 advantages (negative of model1)
            perf_diff.loc[f"{model2} adv", "As P1"] = -perf_diff.loc[
                f"{model1} adv", "As P1"
            ]
            perf_diff.loc[f"{model2} adv", "As P2"] = -perf_diff.loc[
                f"{model1} adv", "As P2"
            ]

            # Create heatmap
            ax = axes[idx]
            im = ax.imshow(
                perf_diff.values, cmap="RdBu_r", vmin=-0.5, vmax=0.5, aspect="auto"
            )

            # Add annotations
            for i in range(len(perf_diff.index)):
                for j in range(len(perf_diff.columns)):
                    value = perf_diff.iloc[i, j]
                    color = "white" if abs(value) > 0.25 else "black"
                    ax.text(
                        j,
                        i,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontsize=11,
                        fontweight="bold",
                    )

            ax.set_title(
                f"{behavior}\n{model1} vs {model2} Advantage",
                fontsize=12,
                fontweight="bold",
            )
            ax.set_xticks(range(len(perf_diff.columns)))
            ax.set_yticks(range(len(perf_diff.index)))
            ax.set_xticklabels(perf_diff.columns, fontsize=10)
            ax.set_yticklabels(perf_diff.index, fontsize=10)

            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label("Performance Advantage", fontsize=10, fontweight="bold")

        plt.tight_layout()

        output_file = self.output_dir / f"difference_{model1}_vs_{model2}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Saved difference heatmap: {output_file}")

        return fig

    def generate_all_visualizations(self):
        """Generate comprehensive visualization suite"""
        self.load_results()
        self.calculate_metrics()

        print("Generating visualizations...")

        # Main academic-style heatmaps
        self.create_academic_style_heatmap("win_rates", "Win Rate - ")
        self.create_academic_style_heatmap("payoffs_p1", "Payoff (P1) - ")
        self.create_academic_style_heatmap("acceptance_rates", "Acceptance Rate - ")

        # Combined figure
        self.create_combined_academic_figure()

        # Model comparisons (if we have the models)
        models = list(self.metrics[list(self.metrics.keys())[0]]["win_rates"].index)
        if "GPT-4o" in models and "GPT-3.5" in models:
            self.create_difference_heatmap("GPT-4o", "GPT-3.5")
        if "GPT-4o" in models and "GPT-5-nano" in models:
            self.create_difference_heatmap("GPT-4o", "GPT-5-nano")
        if "GPT-3.5" in models and "GPT-5-nano" in models:
            self.create_difference_heatmap("GPT-3.5", "GPT-5-nano")

        print(f"\nAll visualizations saved to: {self.output_dir}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python create_custom_heatmap.py <results_file> [output_dir]")
        print(
            "Example: python create_custom_heatmap.py .logs/ultimatum_social_behavior_20241201_143022/all_results.json"
        )
        return

    results_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(results_file).exists():
        print(f"Error: Results file not found: {results_file}")
        return

    generator = CustomHeatmapGenerator(results_file, output_dir)
    generator.generate_all_visualizations()

    print("\n✓ Custom heatmap generation completed!")


if __name__ == "__main__":
    main()
