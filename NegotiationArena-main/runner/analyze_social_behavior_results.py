import json
import os
import sys
from pathlib import Path

# Add current directory to Python path for module imports
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Rectangle


class SocialBehaviorResultsAnalyzer:
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.results_file = self.results_dir / "all_results.json"
        self.results = []

    def load_results(self):
        """Load results from JSON file"""
        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {self.results_file}")

        with open(self.results_file, "r") as f:
            self.results = json.load(f)

        print(f"Loaded {len(self.results)} game results")
        return self.results

    def calculate_metrics(self):
        """Calculate win rates, payoffs, and other metrics"""
        df = pd.DataFrame(self.results)

        # Filter only completed games
        completed_df = df[df["game_completed"] == True].copy()

        if len(completed_df) == 0:
            print("WARNING: No completed games found!")
            return {}

        print(
            f"Analyzing {len(completed_df)} completed games out of {len(df)} total games"
        )

        # Get unique values
        models = sorted(completed_df["model1"].unique())
        behaviors = sorted(completed_df["behavior"].unique())

        metrics = {}

        for behavior in behaviors:
            behavior_df = completed_df[completed_df["behavior"] == behavior]

            # Get actual model combinations that exist (no same vs same)
            actual_combinations = behavior_df[["model1", "model2"]].drop_duplicates()

            # Initialize result matrices
            win_rates = pd.DataFrame(index=models, columns=models, dtype=float)
            avg_payoffs_p1 = pd.DataFrame(index=models, columns=models, dtype=float)
            avg_payoffs_p2 = pd.DataFrame(index=models, columns=models, dtype=float)
            game_counts = pd.DataFrame(index=models, columns=models, dtype=int)

            # Initialize all cells to NaN (for same model vs same model)
            win_rates[:] = np.nan
            avg_payoffs_p1[:] = np.nan
            avg_payoffs_p2[:] = np.nan
            game_counts[:] = 0

            for model1 in models:
                for model2 in models:
                    if model1 == model2:
                        # Skip same model vs same model - leave as NaN
                        continue

                    subset = behavior_df[
                        (behavior_df["model1"] == model1)
                        & (behavior_df["model2"] == model2)
                    ]

                    if len(subset) > 0:
                        # Win rate calculation - Player 1 wins if they get more resources
                        p1_resources = pd.to_numeric(
                            subset["player1_final_resources"], errors="coerce"
                        ).fillna(0)
                        p2_resources = pd.to_numeric(
                            subset["player2_final_resources"], errors="coerce"
                        ).fillna(0)

                        wins = (p1_resources > p2_resources).sum()
                        ties = (p1_resources == p2_resources).sum()

                        # Win rate includes half points for ties
                        win_rate = (wins + 0.5 * ties) / len(subset)
                        win_rates.loc[model1, model2] = win_rate

                        # Average payoffs
                        avg_payoffs_p1.loc[model1, model2] = p1_resources.mean()
                        avg_payoffs_p2.loc[model1, model2] = p2_resources.mean()

                        # Game count
                        game_counts.loc[model1, model2] = len(subset)
                    else:
                        win_rates.loc[model1, model2] = 0.0
                        avg_payoffs_p1.loc[model1, model2] = 0.0
                        avg_payoffs_p2.loc[model1, model2] = 0.0
                        game_counts.loc[model1, model2] = 0

            metrics[behavior] = {
                "win_rates": win_rates,
                "avg_payoffs_p1": avg_payoffs_p1,
                "avg_payoffs_p2": avg_payoffs_p2,
                "game_counts": game_counts,
            }

        return metrics

    def create_combined_heatmap(self, metrics):
        """Create a combined heatmap similar to the reference image"""
        behaviors = list(metrics.keys())
        n_behaviors = len(behaviors)

        if n_behaviors == 0:
            print("No behaviors to analyze!")
            return

        # Create figure with appropriate size
        fig, axes = plt.subplots(n_behaviors, 2, figsize=(12, 4 * n_behaviors))

        # Handle single behavior case
        if n_behaviors == 1:
            axes = axes.reshape(1, -1)

        for idx, behavior in enumerate(behaviors):
            win_rates = metrics[behavior]["win_rates"]
            payoffs = metrics[behavior]["avg_payoffs_p1"]

            # Win Rate heatmap
            ax1 = axes[idx, 0] if n_behaviors > 1 else axes[0]
            im1 = ax1.imshow(
                win_rates.values, cmap="Blues", vmin=0, vmax=1, aspect="auto"
            )

            # Add text annotations
            for i in range(len(win_rates.index)):
                for j in range(len(win_rates.columns)):
                    value = win_rates.iloc[i, j]
                    color = "white" if value > 0.5 else "black"
                    ax1.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        color=color,
                        fontweight="bold",
                    )

            ax1.set_title(f"{behavior} - Win Rate", fontsize=12, fontweight="bold")
            ax1.set_xlabel("Player 2", fontweight="bold")
            ax1.set_ylabel("Player 1", fontweight="bold")
            ax1.set_xticks(range(len(win_rates.columns)))
            ax1.set_yticks(range(len(win_rates.index)))
            ax1.set_xticklabels(win_rates.columns)
            ax1.set_yticklabels(win_rates.index)

            # Add colorbar for win rates
            cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
            cbar1.set_label("Win Rate", fontweight="bold")

            # Payoff heatmap
            ax2 = axes[idx, 1] if n_behaviors > 1 else axes[1]

            # Normalize payoffs for better visualization
            payoff_min = payoffs.values.min()
            payoff_max = payoffs.values.max()

            # Create masked array for payoffs
            masked_payoffs = np.ma.masked_invalid(payoffs.values)
            im2 = ax2.imshow(masked_payoffs, cmap="Blues", aspect="auto")

            # Add text annotations
            for i in range(len(payoffs.index)):
                for j in range(len(payoffs.columns)):
                    value = payoffs.iloc[i, j]
                    if np.isnan(value):
                        # Gray out diagonal (same model vs same model)
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
                            fontweight="bold",
                        )
                    else:
                        # Choose text color based on value relative to range
                        valid_values = payoffs.values[~np.isnan(payoffs.values)]
                        if len(valid_values) > 0:
                            payoff_min, payoff_max = (
                                valid_values.min(),
                                valid_values.max(),
                            )
                            norm_value = (
                                (value - payoff_min) / (payoff_max - payoff_min)
                                if payoff_max > payoff_min
                                else 0
                            )
                        else:
                            norm_value = 0
                        color = "white" if norm_value > 0.5 else "black"
                        ax2.text(
                            j,
                            i,
                            f"{value:.1f}",
                            ha="center",
                            va="center",
                            color=color,
                            fontweight="bold",
                        )

            ax2.set_title(
                f"{behavior} - Average Payoff (Player 1)",
                fontsize=12,
                fontweight="bold",
            )
            ax2.set_xlabel("Player 2", fontweight="bold")
            ax2.set_ylabel("Player 1", fontweight="bold")
            ax2.set_xticks(range(len(payoffs.columns)))
            ax2.set_yticks(range(len(payoffs.index)))
            ax2.set_xticklabels(payoffs.columns)
            ax2.set_yticklabels(payoffs.index)

            # Add colorbar for payoffs
            cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
            cbar2.set_label("Average Payoff", fontweight="bold")

        plt.tight_layout()

        # Save the plot
        output_file = self.results_dir / "social_behavior_heatmap.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"Heatmap saved to: {output_file}")

        plt.show()

    def create_summary_statistics(self, metrics):
        """Create summary statistics table"""
        summary_data = []

        for behavior, behavior_metrics in metrics.items():
            win_rates = behavior_metrics["win_rates"]
            payoffs = behavior_metrics["avg_payoffs_p1"]
            game_counts = behavior_metrics["game_counts"]

            # Calculate overall statistics for this behavior
            total_games = game_counts.values.sum()
            avg_win_rate = win_rates.values.mean()
            avg_payoff = payoffs.values.mean()

            # Model-specific performance (as Player 1)
            for model in win_rates.index:
                # Calculate mean excluding NaN values (same model comparisons)
                model_win_rate = win_rates.loc[model].mean(skipna=True)
                model_avg_payoff = payoffs.loc[model].mean(skipna=True)
                model_games = game_counts.loc[model].sum()

                summary_data.append(
                    {
                        "Behavior": behavior,
                        "Model": model,
                        "Role": "Player 1",
                        "Avg_Win_Rate": model_win_rate,
                        "Avg_Payoff": model_avg_payoff,
                        "Games_Played": model_games,
                    }
                )

            # Model-specific performance (as Player 2) - calculate from transpose
            for model in win_rates.columns:
                # When this model is Player 2, win rate is 1 - opponent's win rate
                opponent_win_rates = win_rates[model].dropna()  # Remove NaN values
                if len(opponent_win_rates) > 0:
                    model_win_rate = 1 - opponent_win_rates.mean()
                else:
                    model_win_rate = np.nan

                model_avg_payoff = behavior_metrics["avg_payoffs_p2"][model].mean(
                    skipna=True
                )
                model_games = game_counts[model].sum()

                summary_data.append(
                    {
                        "Behavior": behavior,
                        "Model": model,
                        "Role": "Player 2",
                        "Avg_Win_Rate": model_win_rate,
                        "Avg_Payoff": model_avg_payoff,
                        "Games_Played": model_games,
                    }
                )

        summary_df = pd.DataFrame(summary_data)

        # Save summary to CSV
        summary_file = self.results_dir / "summary_statistics.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary statistics saved to: {summary_file}")

        return summary_df

    def print_detailed_results(self, metrics):
        """Print detailed results to console"""
        print("\n" + "=" * 80)
        print("DETAILED RESULTS ANALYSIS")
        print("=" * 80)

        for behavior, behavior_metrics in metrics.items():
            print(f"\n{behavior.upper()} BEHAVIOR:")
            print("-" * 50)

            print("\nWin Rates Matrix (Player 1 perspective):")
            print(behavior_metrics["win_rates"].round(3))

            print(f"\nAverage Payoffs Matrix (Player 1):")
            print(behavior_metrics["avg_payoffs_p1"].round(2))

            print(f"\nAverage Payoffs Matrix (Player 2):")
            print(behavior_metrics["avg_payoffs_p2"].round(2))

            print(f"\nGame Counts Matrix:")
            print(behavior_metrics["game_counts"])

    def run_full_analysis(self):
        """Run complete analysis pipeline"""
        print("Starting social behavior results analysis...")

        # Load results
        self.load_results()

        # Calculate metrics
        metrics = self.calculate_metrics()

        if not metrics:
            print("No metrics calculated. Exiting.")
            return

        # Create visualizations
        self.create_combined_heatmap(metrics)

        # Create summary statistics
        summary_df = self.create_summary_statistics(metrics)

        # Print detailed results
        self.print_detailed_results(metrics)

        print(f"\nAnalysis complete! All outputs saved to: {self.results_dir}")

        return metrics, summary_df


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python analyze_social_behavior_results.py <results_directory>")
        print(
            "Example: python analyze_social_behavior_results.py .logs/ultimatum_social_behavior_20241201_143022"
        )
        return

    results_dir = sys.argv[1]

    if not os.path.exists(results_dir):
        print(f"Error: Results directory does not exist: {results_dir}")
        return

    analyzer = SocialBehaviorResultsAnalyzer(results_dir)

    try:
        analyzer.run_full_analysis()
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
