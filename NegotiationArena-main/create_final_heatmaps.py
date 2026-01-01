#!/usr/bin/env python3
"""
Create Final Heatmaps for Ultimatum Game Social Behavior Results

This script creates publication-quality heatmaps from the analyzed results,
similar to academic research papers.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle

# Set publication-quality matplotlib parameters
plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "serif",
        "axes.linewidth": 1.2,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
        "figure.dpi": 300,
    }
)


class UltimatumHeatmapGenerator:
    # Known behaviors to help with processing
    KNOWN_BEHAVIORS = [
        "Hindi",
        "Gujarati",
        "Marwadi",
        #"Marwadi_Forced",
        "Punjabi",
        "Baseline",
    ]

    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.summary_file = self.results_dir / "summary.json"
        self.summary_data = {}

    def load_data(self):
        """Load the summary data"""
        if not self.summary_file.exists():
            raise FileNotFoundError(f"Summary file not found: {self.summary_file}")

        with open(self.summary_file, "r", encoding="utf-8") as f:
            self.summary_data = json.load(f)

        print(f"Loaded data for {len(self.summary_data)} combinations")

    def extract_matrices(self):
        """Extract matrices for visualization"""
        # Get unique models and behaviors
        models = set()
        behaviors = set()

        for combo_key, metrics in self.summary_data.items():
            models.add(metrics["model1"])
            models.add(metrics["model2"])
            behaviors.add(metrics["behavior"])

        models = sorted(list(models))
        behaviors = sorted(list(behaviors))

        print(f"Models: {models}")
        print(f"Behaviors: {behaviors}")

        # Create matrices for each metric
        matrices = {}

        for behavior in behaviors:
            matrices[behavior] = {
                "win_rates": pd.DataFrame(index=models, columns=models, dtype=float),
                "payoffs": pd.DataFrame(index=models, columns=models, dtype=float),
                "initial_offers": pd.DataFrame(
                    index=models, columns=models, dtype=float
                ),
                "acceptance_rates": pd.DataFrame(
                    index=models, columns=models, dtype=float
                ),
                "game_counts": pd.DataFrame(index=models, columns=models, dtype=int),
            }

            # Fill matrices with NaN initially
            for matrix_name in matrices[behavior]:
                matrices[behavior][matrix_name][:] = np.nan

        # Fill in the data
        for combo_key, metrics in self.summary_data.items():
            model1 = metrics["model1"]
            model2 = metrics["model2"]
            behavior = metrics["behavior"]

            if behavior in matrices:
                matrices[behavior]["win_rates"].loc[model1, model2] = metrics[
                    "win_rate_player1"
                ]
                matrices[behavior]["payoffs"].loc[model1, model2] = metrics[
                    "player1_payoff_avg"
                ]
                matrices[behavior]["initial_offers"].loc[model1, model2] = metrics[
                    "initial_offer_avg"
                ]
                matrices[behavior]["acceptance_rates"].loc[model1, model2] = metrics[
                    "acceptance_rate"
                ]
                matrices[behavior]["game_counts"].loc[model1, model2] = metrics[
                    "total_games"
                ]

        return matrices, models, behaviors

    def create_academic_heatmaps(self, matrices, models, behaviors):
        """Create academic-style heatmaps"""
        n_behaviors = len(behaviors)

        # Create figure with 2 columns (win rates and payoffs)
        fig, axes = plt.subplots(n_behaviors, 2, figsize=(12, 4 * n_behaviors))

        if n_behaviors == 1:
            axes = axes.reshape(1, -1)

        for idx, behavior in enumerate(behaviors):
            # Format behavior name for display (replace underscores with spaces)
            display_behavior = behavior.replace("_", " ")

            # Win Rate heatmap (left column)
            ax1 = axes[idx, 0]
            win_data = matrices[behavior]["win_rates"]

            # Create custom colormap for win rates
            win_cmap = plt.cm.RdBu_r

            # Create masked array for missing data
            win_masked = np.ma.masked_invalid(win_data.values)

            im1 = ax1.imshow(win_masked, cmap=win_cmap, vmin=0, vmax=1, aspect="equal")

            # Add text annotations for win rates
            for i in range(len(win_data.index)):
                for j in range(len(win_data.columns)):
                    value = win_data.iloc[i, j]
                    if np.isnan(value):
                        # Gray out diagonal (same model comparisons)
                        rect = Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            facecolor="lightgray",
                            alpha=0.3,
                            edgecolor="gray",
                        )
                        ax1.add_patch(rect)
                        ax1.text(
                            j,
                            i,
                            "N/A",
                            ha="center",
                            va="center",
                            color="gray",
                            fontweight="bold",
                            fontsize=10,
                        )
                    else:
                        # Choose text color based on value
                        text_color = "white" if value > 0.5 else "black"
                        ax1.text(
                            j,
                            i,
                            f"{value:.2f}",
                            ha="center",
                            va="center",
                            color=text_color,
                            fontweight="bold",
                            fontsize=11,
                        )

            ax1.set_title(
                f"{display_behavior} - Win Rate (Player 1)", fontweight="bold", pad=15
            )
            ax1.set_xlabel("Player 2", fontweight="bold")
            ax1.set_ylabel("Player 1", fontweight="bold")
            ax1.set_xticks(range(len(models)))
            ax1.set_yticks(range(len(models)))
            ax1.set_xticklabels(models, rotation=45, ha="right")
            ax1.set_yticklabels(models)

            # Add colorbar
            cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8, aspect=20)
            cbar1.set_label("Win Rate", fontweight="bold")
            cbar1.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

            # Payoff heatmap (right column)
            ax2 = axes[idx, 1]
            payoff_data = matrices[behavior]["payoffs"]

            # Create custom colormap for payoffs
            payoff_cmap = plt.cm.Blues

            # Create masked array
            payoff_masked = np.ma.masked_invalid(payoff_data.values)

            # Get valid range for normalization
            valid_payoffs = payoff_data.values[~np.isnan(payoff_data.values)]
            if len(valid_payoffs) > 0:
                vmin, vmax = valid_payoffs.min(), valid_payoffs.max()
            else:
                vmin, vmax = 0, 100

            im2 = ax2.imshow(
                payoff_masked, cmap=payoff_cmap, vmin=vmin, vmax=vmax, aspect="equal"
            )

            # Add text annotations for payoffs
            for i in range(len(payoff_data.index)):
                for j in range(len(payoff_data.columns)):
                    value = payoff_data.iloc[i, j]
                    if np.isnan(value):
                        # Gray out diagonal
                        rect = Rectangle(
                            (j - 0.5, i - 0.5),
                            1,
                            1,
                            facecolor="lightgray",
                            alpha=0.3,
                            edgecolor="gray",
                        )
                        ax2.add_patch(rect)
                        ax2.text(
                            j,
                            i,
                            "N/A",
                            ha="center",
                            va="center",
                            color="gray",
                            fontweight="bold",
                            fontsize=10,
                        )
                    else:
                        # Normalize for text color
                        norm_value = (
                            (value - vmin) / (vmax - vmin) if vmax > vmin else 0
                        )
                        text_color = "white" if norm_value > 0.6 else "black"
                        ax2.text(
                            j,
                            i,
                            f"{value:.0f}",
                            ha="center",
                            va="center",
                            color=text_color,
                            fontweight="bold",
                            fontsize=11,
                        )

            ax2.set_title(
                f"{display_behavior} - Average Payoff (Player 1)",
                fontweight="bold",
                pad=15,
            )
            ax2.set_xlabel("Player 2", fontweight="bold")
            ax2.set_ylabel("Player 1", fontweight="bold")
            ax2.set_xticks(range(len(models)))
            ax2.set_yticks(range(len(models)))
            ax2.set_xticklabels(models, rotation=45, ha="right")
            ax2.set_yticklabels(models)

            # Add colorbar
            cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, aspect=20)
            cbar2.set_label("Average Payoff ($)", fontweight="bold")

        plt.tight_layout()

        # Save the plot
        output_file = self.results_dir / "final_heatmaps.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
        plt.show()

        print(f"Heatmaps saved to: {output_file}")
        return output_file

    def create_summary_table(self, matrices, models, behaviors):
        """Create a summary table of key metrics"""
        summary_rows = []

        for behavior in behaviors:
            for model1 in models:
                for model2 in models:
                    if model1 != model2:  # Skip same model comparisons
                        combo_key = f"{model1}_vs_{model2}_{behavior}"
                        if combo_key in self.summary_data:
                            metrics = self.summary_data[combo_key]
                            summary_rows.append(
                                {
                                    "Player 1": model1,
                                    "Player 2": model2,
                                    "Behavior": behavior,
                                    "Games": metrics["total_games"],
                                    "Win Rate": f"{metrics['win_rate_player1']:.3f}",
                                    "Acceptance Rate": f"{metrics['acceptance_rate']:.3f}",
                                    "Avg Payoff P1": f"{metrics['player1_payoff_avg']:.1f}",
                                    "Avg Payoff P2": f"{metrics['player2_payoff_avg']:.1f}",
                                    "Avg Initial Offer": f"{metrics['initial_offer_avg']:.1f}",
                                }
                            )

        # Create DataFrame and save
        df = pd.DataFrame(summary_rows)
        csv_file = self.results_dir / "summary_table.csv"
        df.to_csv(csv_file, index=False)

        print(f"Summary table saved to: {csv_file}")
        return df

    def create_behavior_comparison(self, matrices, models, behaviors):
        """Create a comparison plot across behaviors"""
        # Calculate grid size based on number of behaviors
        n_behaviors = len(behaviors)
        n_cols = 3  # Use 3 columns
        n_rows = (n_behaviors + n_cols - 1) // n_cols  # Ceiling division

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

        # Flatten axes for easier indexing (handle single row case)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for idx, behavior in enumerate(behaviors):
            ax = axes[idx]

            # Get win rate data for this behavior
            win_data = matrices[behavior]["win_rates"]

            # Create a simple bar plot showing win rates for each model combination
            combinations = []
            win_rates = []

            for i in range(len(models)):
                for j in range(len(models)):
                    if i != j and not np.isnan(win_data.iloc[i, j]):
                        combinations.append(f"{models[i]} vs {models[j]}")
                        win_rates.append(win_data.iloc[i, j])

            if combinations:
                bars = ax.bar(
                    range(len(combinations)),
                    win_rates,
                    color=[
                        "#1f77b4",
                        "#ff7f0e",
                        "#2ca02c",
                        "#d62728",
                        "#9467bd",
                        "#8c564b",
                    ][: len(combinations)],
                )

                # Format behavior name for display
                display_behavior = behavior.replace("_", " ")
                ax.set_title(f"{display_behavior}", fontweight="bold")
                ax.set_ylabel("Win Rate (Player 1)", fontweight="bold")
                ax.set_ylim(0, 1)
                ax.set_xticks(range(len(combinations)))
                ax.set_xticklabels(combinations, rotation=45, ha="right")
                ax.grid(axis="y", alpha=0.3)

                # Add value labels on bars
                for bar, rate in zip(bars, win_rates):
                    height = bar.get_height()
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.01,
                        f"{rate:.2f}",
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                    )

        # Remove any unused subplots
        for idx in range(len(behaviors), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()

        # Save the plot
        comparison_file = self.results_dir / "behavior_comparison.png"
        plt.savefig(comparison_file, dpi=300, bbox_inches="tight", facecolor="white")
        plt.show()

        print(f"Behavior comparison saved to: {comparison_file}")
        return comparison_file

    def create_language_comparison(self, matrices, models, behaviors):
        """Create a comprehensive comparison across behaviors/languages"""
        print("\n" + "=" * 60)
        print("CREATING LANGUAGE/BEHAVIOR COMPARISON")
        print("=" * 60 + "\n")

        # Collect aggregate statistics for each behavior
        behavior_stats = {}

        for behavior in behaviors:
            # Collect raw data and counts
            all_player1_payoffs = []
            all_player2_payoffs = []
            all_initial_offers = []
            total_accepts = 0
            total_rejects = 0
            total_player1_wins = 0
            total_player2_wins = 0
            total_draws = 0
            total_games = 0

            # Aggregate data from summary
            for combo_key, metrics in self.summary_data.items():
                if metrics["behavior"] == behavior:
                    # Collect raw payoff and offer data
                    all_player1_payoffs.extend(metrics.get("player1_payoffs", []))
                    all_player2_payoffs.extend(metrics.get("player2_payoffs", []))
                    all_initial_offers.extend(metrics.get("initial_offers", []))

                    # Sum up counts
                    total_accepts += metrics.get("accepts", 0)
                    total_rejects += metrics.get("rejects", 0)
                    total_player1_wins += metrics.get("player1_wins", 0)
                    total_player2_wins += metrics.get("player2_wins", 0)
                    total_draws += metrics.get("draws", 0)
                    total_games += metrics.get("total_games", 0)

            # Calculate aggregate statistics
            total_non_draw_games = total_games - total_draws

            behavior_stats[behavior] = {
                # Acceptance rate calculated from total accepts/rejects
                "avg_acceptance": total_accepts / total_games if total_games > 0 else 0,
                "std_acceptance": np.sqrt(
                    (total_accepts / total_games)
                    * (total_rejects / total_games)
                    / total_games
                )
                if total_games > 0
                else 0,
                # Win rate calculated from total wins / non-draw games
                "avg_win_rate": total_player1_wins / total_non_draw_games
                if total_non_draw_games > 0
                else 0,
                "std_win_rate": np.sqrt(
                    (total_player1_wins / total_non_draw_games)
                    * (total_player2_wins / total_non_draw_games)
                    / total_non_draw_games
                )
                if total_non_draw_games > 0
                else 0,
                # Payoffs calculated from raw data
                "avg_payoff_p1": np.mean(all_player1_payoffs)
                if all_player1_payoffs
                else 0,
                "std_payoff_p1": np.std(all_player1_payoffs)
                if all_player1_payoffs
                else 0,
                "avg_payoff_p2": np.mean(all_player2_payoffs)
                if all_player2_payoffs
                else 0,
                "std_payoff_p2": np.std(all_player2_payoffs)
                if all_player2_payoffs
                else 0,
                # Initial offers calculated from raw data
                "avg_initial_offer": np.mean(all_initial_offers)
                if all_initial_offers
                else 0,
                "std_initial_offer": np.std(all_initial_offers)
                if all_initial_offers
                else 0,
                # Counts
                "total_games": total_games,
                "total_non_draw_games": total_non_draw_games,
            }

        # Create figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Format behavior names for display
        display_behaviors = [b.replace("_", " ") for b in behaviors]

        # 1. Acceptance Rate Comparison
        ax1 = axes[0, 0]
        acceptance_rates = [behavior_stats[b]["avg_acceptance"] for b in behaviors]
        acceptance_stds = [behavior_stats[b]["std_acceptance"] for b in behaviors]
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(behaviors)))
        bars1 = ax1.bar(
            display_behaviors,
            acceptance_rates,
            yerr=acceptance_stds,
            color=colors,
            edgecolor="black",
            linewidth=1.5,
            capsize=5,
            error_kw={"linewidth": 2, "ecolor": "black", "alpha": 0.6},
        )
        ax1.set_title(
            "Average Acceptance Rate by Behavior", fontweight="bold", fontsize=14
        )
        ax1.set_ylabel("Acceptance Rate", fontweight="bold")
        ax1.set_ylim(0, 1.1)
        ax1.grid(axis="y", alpha=0.3, linestyle="--")

        # Add value labels
        for bar, rate, std in zip(bars1, acceptance_rates, acceptance_stds):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + std + 0.02,
                f"{rate:.2%}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # 2. Average Initial Offer Comparison
        ax2 = axes[0, 1]
        initial_offers = [behavior_stats[b]["avg_initial_offer"] for b in behaviors]
        initial_offer_stds = [behavior_stats[b]["std_initial_offer"] for b in behaviors]
        bars2 = ax2.bar(
            display_behaviors,
            initial_offers,
            yerr=initial_offer_stds,
            color=colors,
            edgecolor="black",
            linewidth=1.5,
            capsize=5,
            error_kw={"linewidth": 2, "ecolor": "black", "alpha": 0.6},
        )
        ax2.set_title(
            "Average Initial Offer by Behavior", fontweight="bold", fontsize=14
        )
        ax2.set_ylabel("Initial Offer ($)", fontweight="bold")
        max_val = (
            max([o + s for o, s in zip(initial_offers, initial_offer_stds)])
            if initial_offers
            else 100
        )
        ax2.set_ylim(0, max_val * 1.2)
        ax2.grid(axis="y", alpha=0.3, linestyle="--")

        # Add value labels
        for bar, offer, std in zip(bars2, initial_offers, initial_offer_stds):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + std + 1,
                f"${offer:.1f}±{std:.1f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # 3. Average Payoffs Comparison (P1 vs P2)
        ax3 = axes[1, 0]
        x = np.arange(len(behaviors))
        width = 0.35

        payoffs_p1 = [behavior_stats[b]["avg_payoff_p1"] for b in behaviors]
        payoffs_p2 = [behavior_stats[b]["avg_payoff_p2"] for b in behaviors]
        payoffs_p1_std = [behavior_stats[b]["std_payoff_p1"] for b in behaviors]
        payoffs_p2_std = [behavior_stats[b]["std_payoff_p2"] for b in behaviors]

        bars3a = ax3.bar(
            x - width / 2,
            payoffs_p1,
            width,
            yerr=payoffs_p1_std,
            label="Player 1",
            color="#2E86AB",
            edgecolor="black",
            linewidth=1.2,
            capsize=5,
            error_kw={"linewidth": 2, "ecolor": "black", "alpha": 0.6},
        )
        bars3b = ax3.bar(
            x + width / 2,
            payoffs_p2,
            width,
            yerr=payoffs_p2_std,
            label="Player 2",
            color="#A23B72",
            edgecolor="black",
            linewidth=1.2,
            capsize=5,
            error_kw={"linewidth": 2, "ecolor": "black", "alpha": 0.6},
        )

        ax3.set_title("Average Payoffs by Behavior", fontweight="bold", fontsize=14)
        ax3.set_ylabel("Average Payoff ($)", fontweight="bold")
        ax3.set_xticks(x)
        ax3.set_xticklabels(display_behaviors)
        ax3.legend(fontsize=11, loc="upper right")
        ax3.grid(axis="y", alpha=0.3, linestyle="--")

        # Add value labels
        for bar, std in zip(bars3a, payoffs_p1_std):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + std + 1,
                f"${height:.1f}±{std:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
        for bar, std in zip(bars3b, payoffs_p2_std):
            height = bar.get_height()
            ax3.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + std + 1,
                f"${height:.1f}±{std:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # 4. Win Rate Comparison
        ax4 = axes[1, 1]
        win_rates = [behavior_stats[b]["avg_win_rate"] for b in behaviors]
        win_rate_stds = [behavior_stats[b]["std_win_rate"] for b in behaviors]
        bars4 = ax4.bar(
            display_behaviors,
            win_rates,
            yerr=win_rate_stds,
            color=colors,
            edgecolor="black",
            linewidth=1.5,
            capsize=5,
            error_kw={"linewidth": 2, "ecolor": "black", "alpha": 0.6},
        )
        ax4.set_title(
            "Average Win Rate (Player 1) by Behavior", fontweight="bold", fontsize=14
        )
        ax4.set_ylabel("Win Rate", fontweight="bold")
        ax4.set_ylim(0, 1.1)
        ax4.grid(axis="y", alpha=0.3, linestyle="--")

        # Add value labels
        for bar, rate, std in zip(bars4, win_rates, win_rate_stds):
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + std + 0.02,
                f"{rate:.2%}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=10,
            )

        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right")

        plt.tight_layout()

        # Save the plot
        comparison_file = self.results_dir / "language_behavior_comparison.png"
        plt.savefig(comparison_file, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"✓ Language/Behavior comparison saved to: {comparison_file}")
        plt.close()

        # Create a summary table
        summary_data = []
        for behavior in behaviors:
            stats = behavior_stats[behavior]
            summary_data.append(
                {
                    "Behavior": behavior.replace("_", " "),
                    "Total Games": stats["total_games"],
                    "Non-Draw Games": stats["total_non_draw_games"],
                    "Acceptance Rate": f"{stats['avg_acceptance']:.2%} ± {stats['std_acceptance']:.2%}",
                    "Initial Offer": f"${stats['avg_initial_offer']:.2f} ± {stats['std_initial_offer']:.2f}",
                    "Payoff P1": f"${stats['avg_payoff_p1']:.2f} ± {stats['std_payoff_p1']:.2f}",
                    "Payoff P2": f"${stats['avg_payoff_p2']:.2f} ± {stats['std_payoff_p2']:.2f}",
                    "Win Rate P1": f"{stats['avg_win_rate']:.2%} ± {stats['std_win_rate']:.2%}",
                }
            )

        df = pd.DataFrame(summary_data)
        csv_file = self.results_dir / "behavior_comparison_summary.csv"
        df.to_csv(csv_file, index=False)
        print(f"✓ Behavior comparison summary saved to: {csv_file}")

        # Print ranking
        print("\n" + "-" * 60)
        print("BEHAVIOR RANKINGS:")
        print("-" * 60)

        # Sort by acceptance rate
        sorted_by_acceptance = sorted(
            behaviors, key=lambda b: behavior_stats[b]["avg_acceptance"], reverse=True
        )
        print("\nBy Acceptance Rate (highest to lowest):")
        for i, behavior in enumerate(sorted_by_acceptance, 1):
            stats = behavior_stats[behavior]
            print(f"  {i}. {behavior.replace('_', ' ')}: {stats['avg_acceptance']:.2%}")

        # Sort by initial offer (fairness)
        sorted_by_offer = sorted(
            behaviors,
            key=lambda b: behavior_stats[b]["avg_initial_offer"],
            reverse=True,
        )
        print("\nBy Initial Offer - Fairness (highest to lowest):")
        for i, behavior in enumerate(sorted_by_offer, 1):
            stats = behavior_stats[behavior]
            print(
                f"  {i}. {behavior.replace('_', ' ')}: ${stats['avg_initial_offer']:.2f}"
            )

        # Sort by P1 payoff
        sorted_by_payoff = sorted(
            behaviors, key=lambda b: behavior_stats[b]["avg_payoff_p1"], reverse=True
        )
        print("\nBy Player 1 Payoff (highest to lowest):")
        for i, behavior in enumerate(sorted_by_payoff, 1):
            stats = behavior_stats[behavior]
            print(f"  {i}. {behavior.replace('_', ' ')}: ${stats['avg_payoff_p1']:.2f}")

        print()

        return comparison_file, csv_file

    def generate_all_visualizations(self):
        """Generate all visualizations"""
        print("Creating Final Heatmaps and Visualizations")
        print("=" * 50)

        # Load data
        self.load_data()

        # Extract matrices
        matrices, models, behaviors = self.extract_matrices()

        # Create heatmaps
        heatmap_file = self.create_academic_heatmaps(matrices, models, behaviors)

        # Create summary table
        summary_df = self.create_summary_table(matrices, models, behaviors)

        # Create behavior comparison
        comparison_file = self.create_behavior_comparison(matrices, models, behaviors)

        # Create language/behavior comparison
        language_comparison_file, behavior_summary_csv = (
            self.create_language_comparison(matrices, models, behaviors)
        )

        print("\n" + "=" * 50)
        print("VISUALIZATION COMPLETE!")
        print("=" * 50)
        print(f"Files created:")
        print(f"  - {heatmap_file}")
        print(f"  - {comparison_file}")
        print(f"  - {language_comparison_file}")
        print(f"  - {self.results_dir}/summary_table.csv")
        print(f"  - {behavior_summary_csv}")

        return heatmap_file, comparison_file, language_comparison_file


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python create_final_heatmaps.py <results_directory>")
        print(
            "Example: python create_final_heatmaps.py .logs/ultimatum_social_behavior_20251130_180851"
        )
        return

    results_dir = sys.argv[1]

    if not Path(results_dir).exists():
        print(f"Error: Results directory does not exist: {results_dir}")
        return

    try:
        generator = UltimatumHeatmapGenerator(results_dir)
        generator.generate_all_visualizations()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
