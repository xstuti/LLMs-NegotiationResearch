#!/usr/bin/env python3
"""
Analyze Buy-Sell Game Results

This script analyzes the buy-sell game results from the log files,
extracting key metrics from the all_results.json file.
"""

import json
import os
import statistics
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import defaultdict
from pathlib import Path

# Add current directory to Python path for module imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


class BuySellResultsAnalyzer:
    # Known behaviors to help with parsing
    KNOWN_BEHAVIORS = [
        "Hindi",
        "Gujarati",
        "Marwadi",
        "Punjabi",
        "English",
    ]

    # Known model patterns
    KNOWN_MODELS = [
        "GPT-4o",
        "GPT-3.5",
        "Claude-3-Haiku",
        "Claude-3.5-Haiku",
    ]

    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.raw_data = []
        self.summary = {}
        self.behavior_summary = {}

    def analyze_all_games(self):
        """Load all game results from all_results.json"""
        results_file = self.results_dir / "all_results.json"
        if not results_file.exists():
            print(f"Error: {results_file} not found")
            return []

        with open(results_file, "r", encoding="utf-8") as f:
            self.raw_data = json.load(f)

        print(f"Loaded {len(self.raw_data)} game results")
        return self.raw_data

    def calculate_metrics(self):
        """Calculate summary metrics for each model combination and behavior"""
        # Group data by model combination and behavior
        groups = defaultdict(list)

        for game in self.raw_data:
            key = (game["seller_model"], game["buyer_model"], game["behavior"])
            groups[key].append(game)

        summary = {}

        for (seller_model, buyer_model, behavior), games in groups.items():
            combo_key = f"{seller_model}_vs_{buyer_model}_{behavior}"

            # Filter valid games (exclude errors)
            valid_games = [g for g in games if g.get("game_completed", False)]

            if not valid_games:
                print(f"Warning: No valid games found for {combo_key}")
                continue

            # Accepted games for metrics that should only include successful trades
            accepted_games = [g for g in valid_games if g.get("trade_occurred", False)]

            # Calculate requested metrics only
            total_games = len(valid_games)
            accepts = [1 if g.get("trade_occurred", False) else 0 for g in valid_games]

            # Negotiation rounds (keep for all valid games)
            negotiation_rounds = [g.get("negotiation_rounds", 0) for g in valid_games]

            # Seller and buyer advantages (only from accepted trades)
            seller_advantages = [g.get("seller_profit", 0) for g in accepted_games]
            buyer_advantages = [g.get("buyer_savings", 0) for g in accepted_games]

            # Win counts for seller (player 1), only for accepted trades
            p1_wins = 0
            p2_wins = 0
            for g in accepted_games:
                seller_profit = g.get("seller_profit", 0)
                buyer_savings = g.get("buyer_savings", 0)
                if seller_profit > buyer_savings:
                    p1_wins += 1
                elif buyer_savings > seller_profit:
                    p2_wins += 1

            non_draws = p1_wins + p2_wins
            win_rate_p1 = p1_wins / non_draws if non_draws > 0 else 0.0

            # Statistics (means and stds)
            def mean_std(lst):
                m = statistics.mean(lst) if lst else 0.0
                s = statistics.stdev(lst) if len(lst) > 1 else 0.0
                return m, s

            accepts_mean, accepts_std = mean_std(accepts)
            seller_mean, seller_std = mean_std(seller_advantages)
            buyer_mean, buyer_std = mean_std(buyer_advantages)

            metrics = {
                "total_games": total_games,
                "acceptance_rate_mean": accepts_mean,
                "acceptance_rate_std": accepts_std,
                "avg_negotiation_rounds": statistics.mean(negotiation_rounds) if negotiation_rounds else 0.0,
                "negotiation_rounds_std": statistics.stdev(negotiation_rounds) if len(negotiation_rounds) > 1 else 0.0,
                "seller_advantage_mean": seller_mean,
                "seller_advantage_std": seller_std,
                "buyer_advantage_mean": buyer_mean,
                "buyer_advantage_std": buyer_std,
                "player1_wins": p1_wins,
                "player2_wins": p2_wins,
                "non_draws": non_draws,
                "win_rate_player1": win_rate_p1,
                "seller_model": seller_model,
                "buyer_model": buyer_model,
                "behavior": behavior,
            }

            summary[combo_key] = metrics

            # Print summary for this combination
            print(f"\n{combo_key}:")
            num_accepts = sum(accepts) if accepts else 0
            num_rejects = total_games - num_accepts
            draws = total_games - (p1_wins + p2_wins)
            draw_rate = draws / total_games if total_games > 0 else 0.0
            print(f"  Games: {total_games} | Accepts: {num_accepts} | Rejects: {num_rejects}")
            print(f"  Win rate (Seller, excluding ties): {win_rate_p1:.3f} | Draw rate: {draw_rate:.3f}")
            print(f"  Avg advantages - Seller: {metrics['seller_advantage_mean']:.1f}, Buyer: {metrics['buyer_advantage_mean']:.1f}")
            print(f"  Avg negotiation rounds: {metrics['avg_negotiation_rounds']:.1f}")

        self.summary = summary
        return summary

    def calculate_behavior_summary(self):
        """Aggregate metrics across all model combinations for each behavior."""
        behavior_groups = defaultdict(list)
        for game in self.raw_data:
            behavior_groups[game["behavior"]].append(game)

        behavior_summary = {}

        for behavior, games in behavior_groups.items():
            if not games:
                continue

            # Filter valid games
            valid_games = [g for g in games if g.get("game_completed", False)]
            accepted_games = [g for g in valid_games if g.get("trade_occurred", False)]

            # Compute per-game metrics
            accepts = [1 if g.get("trade_occurred", False) else 0 for g in valid_games]
            negotiation_rounds = [g.get("negotiation_rounds", 0) for g in valid_games]
            seller_advantages = [g.get("seller_profit", 0) for g in accepted_games]
            buyer_advantages = [g.get("buyer_savings", 0) for g in accepted_games]

            # Wins for seller
            p1_wins = sum(1 for g in accepted_games if g.get("seller_profit", 0) > g.get("buyer_savings", 0))
            p2_wins = sum(1 for g in accepted_games if g.get("buyer_savings", 0) > g.get("seller_profit", 0))
            non_draws = p1_wins + p2_wins
            win_rate_p1 = p1_wins / non_draws if non_draws > 0 else 0.0

            def mean_std(lst):
                m = statistics.mean(lst) if lst else 0.0
                s = statistics.stdev(lst) if len(lst) > 1 else 0.0
                return m, s

            acc_m, acc_s = mean_std(accepts)
            seller_m, seller_s = mean_std(seller_advantages)
            buyer_m, buyer_s = mean_std(buyer_advantages)

            behavior_metrics = {
                "behavior": behavior,
                "total_games": len(valid_games),
                "acceptance_rate_mean": acc_m,
                "acceptance_rate_std": acc_s,
                "avg_negotiation_rounds": statistics.mean(negotiation_rounds) if negotiation_rounds else 0.0,
                "negotiation_rounds_std": statistics.stdev(negotiation_rounds) if len(negotiation_rounds) > 1 else 0.0,
                "seller_advantage_mean": seller_m,
                "seller_advantage_std": seller_s,
                "buyer_advantage_mean": buyer_m,
                "buyer_advantage_std": buyer_s,
                "player1_wins": p1_wins,
                "player2_wins": p2_wins,
                "non_draws": non_draws,
                "win_rate_player1": win_rate_p1,
            }

            behavior_summary[behavior] = behavior_metrics

        self.behavior_summary = behavior_summary
        return behavior_summary

    def create_bar_plots_and_tables(self):
        """Create bar plots for acceptance rate, seller advantage, and buyer advantage by behavior.
        Also save CSV tables with mean and std for each behavior."""
        out_dir = Path(self.results_dir) / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)

        behaviors = sorted(self.behavior_summary.keys())
        if not behaviors:
            print("No behavior summary to plot")
            return

        # Gather arrays
        acc_means = [self.behavior_summary[b]["acceptance_rate_mean"] for b in behaviors]
        acc_stds = [self.behavior_summary[b]["acceptance_rate_std"] for b in behaviors]
        seller_means = [self.behavior_summary[b]["seller_advantage_mean"] for b in behaviors]
        seller_stds = [self.behavior_summary[b]["seller_advantage_std"] for b in behaviors]
        buyer_means = [self.behavior_summary[b]["buyer_advantage_mean"] for b in behaviors]
        buyer_stds = [self.behavior_summary[b]["buyer_advantage_std"] for b in behaviors]
        win_means = [self.behavior_summary[b]["win_rate_player1"] for b in behaviors]

        # Styling similar to provided figure
        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        })

        colors = ["#4C4CB8", "#2E8BC0", "#18A162", "#6AD34F", "#D5E86B", "#B7E06C"]
        bar_colors = [colors[i % len(colors)] for i in range(len(behaviors))]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Acceptance rate
        ax = axes[0, 0]
        x = np.arange(len(behaviors))
        ax.bar(x, acc_means, color=bar_colors, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(behaviors, rotation=30, ha='right')
        ax.set_ylim(0, 1.05)
        ax.set_title('Average Acceptance Rate by Behavior')
        for i, v in enumerate(acc_means):
            ax.text(i, v + 0.02, f"{v*100:.2f}%", ha='center', fontsize=9)

        # Seller advantage
        ax = axes[0, 1]
        ax.bar(x, seller_means, color=bar_colors, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(behaviors, rotation=30, ha='right')
        ax.set_title('Average Seller Advantage by Behavior')
        for i, v in enumerate(seller_means):
            ax.text(i, v + (max(seller_means) * 0.02 if seller_means else 0.1), f"{v:.1f}", ha='center', fontsize=9)

        # Buyer advantage
        ax = axes[1, 0]
        ax.bar(x, buyer_means, color=bar_colors, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(behaviors, rotation=30, ha='right')
        ax.set_title('Average Buyer Advantage by Behavior')
        for i, v in enumerate(buyer_means):
            ax.text(i, v + (max(buyer_means) * 0.02 if buyer_means else 0.1), f"{v:.1f}", ha='center', fontsize=9)

        # Win rate (seller)
        ax = axes[1, 1]
        ax.bar(x, win_means, color=bar_colors, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(behaviors, rotation=30, ha='right')
        ax.set_ylim(0, 1.05)
        ax.set_title('Average Win Rate (Seller) by Behavior')
        for i, v in enumerate(win_means):
            ax.text(i, v + 0.02, f"{v*100:.2f}%", ha='center', fontsize=9)

        plt.tight_layout()
        fig_file = out_dir / "behavior_bar_summary.png"
        fig.savefig(fig_file, dpi=150)
        plt.close(fig)

        # Save CSV table
        csv_file = Path(self.results_dir) / "behavior_metrics_summary.csv"
        with open(csv_file, "w", encoding="utf-8") as f:
            headers = [
                "behavior",
                "total_games",
                "acceptance_rate_mean",
                "acceptance_rate_std",
                "avg_negotiation_rounds",
                "negotiation_rounds_std",
                "seller_advantage_mean",
                "seller_advantage_std",
                "buyer_advantage_mean",
                "buyer_advantage_std",
                "win_rate_player1",
            ]
            f.write(",".join(headers) + "\n")
            for b in behaviors:
                metrics = self.behavior_summary[b]
                row = [
                    b,
                    str(metrics["total_games"]),
                    f"{metrics['acceptance_rate_mean']:.4f}",
                    f"{metrics['acceptance_rate_std']:.4f}",
                    f"{metrics['avg_negotiation_rounds']:.4f}",
                    f"{metrics['negotiation_rounds_std']:.4f}",
                    f"{metrics['seller_advantage_mean']:.4f}",
                    f"{metrics['seller_advantage_std']:.4f}",
                    f"{metrics['buyer_advantage_mean']:.4f}",
                    f"{metrics['buyer_advantage_std']:.4f}",
                    f"{metrics['win_rate_player1']:.4f}",
                ]
                f.write(",".join(row) + "\n")

        print(f"Saved behavior bar plot: {fig_file}")
        print(f"Saved behavior CSV table: {csv_file}")

    def save_results(self):
        """Save results to JSON files"""
        # Save raw data (already loaded)
        raw_data_file = self.results_dir / "raw_game_data.json"
        with open(raw_data_file, "w", encoding="utf-8") as f:
            json.dump(self.raw_data, f, indent=2, ensure_ascii=False)

        # Save summary
        summary_file = self.results_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2, ensure_ascii=False)

        # Save behavior-level summary (averaged across model combinations for each behavior)
        behavior_summary_file = self.results_dir / "behavior_summary.json"
        with open(behavior_summary_file, "w", encoding="utf-8") as f:
            json.dump(self.behavior_summary, f, indent=2, ensure_ascii=False)

        # Also save behavior-level summary as CSV
        behavior_csv_file = self.results_dir / "behavior_summary.csv"
        with open(behavior_csv_file, "w", encoding="utf-8") as f:
            headers = [
                "behavior",
                "total_games",
                "acceptance_rate_mean",
                "acceptance_rate_std",
                "avg_negotiation_rounds",
                "negotiation_rounds_std",
                "seller_advantage_mean",
                "seller_advantage_std",
                "buyer_advantage_mean",
                "buyer_advantage_std",
                "win_rate_player1",
            ]
            f.write(",".join(headers) + "\n")
            for behavior, metrics in self.behavior_summary.items():
                row = [
                    behavior,
                    str(metrics["total_games"]),
                    f"{metrics['acceptance_rate_mean']:.4f}",
                    f"{metrics['acceptance_rate_std']:.4f}",
                    f"{metrics['avg_negotiation_rounds']:.4f}",
                    f"{metrics['negotiation_rounds_std']:.4f}",
                    f"{metrics['seller_advantage_mean']:.4f}",
                    f"{metrics['seller_advantage_std']:.4f}",
                    f"{metrics['buyer_advantage_mean']:.4f}",
                    f"{metrics['buyer_advantage_std']:.4f}",
                    f"{metrics['win_rate_player1']:.4f}",
                ]
                f.write(",".join(row) + "\n")

        # Create readable summary
        readable_file = self.results_dir / "readable_summary.txt"
        with open(readable_file, "w", encoding="utf-8") as f:
            f.write("BUY-SELL GAME RESULTS SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            for combo_key, metrics in self.summary.items():
                f.write(f"{combo_key}:\n")
                f.write(f"  Total games: {metrics['total_games']}\n")
                f.write(f"  Acceptance rate: {metrics['acceptance_rate_mean']:.3f} ± {metrics['acceptance_rate_std']:.3f}\n")
                f.write(f"  Avg negotiation rounds: {metrics['avg_negotiation_rounds']:.1f} ± {metrics['negotiation_rounds_std']:.1f}\n")
                f.write(f"  Seller advantage: {metrics['seller_advantage_mean']:.1f} ± {metrics['seller_advantage_std']:.1f}\n")
                f.write(f"  Buyer advantage: {metrics['buyer_advantage_mean']:.1f} ± {metrics['buyer_advantage_std']:.1f}\n")
                f.write(f"  Win rate (Seller): {metrics['win_rate_player1']:.3f}\n")
                f.write("\n")

            # Behavior-level aggregated metrics
            f.write("\nBEHAVIOR-LEVEL AVERAGES\n")
            f.write("=" * 50 + "\n")
            for behavior, metrics in self.behavior_summary.items():
                f.write(f"{behavior}:\n")
                f.write(f"  Total games: {metrics['total_games']}\n")
                f.write(f"  Acceptance rate: {metrics['acceptance_rate_mean']:.3f} ± {metrics['acceptance_rate_std']:.3f}\n")
                f.write(f"  Avg negotiation rounds: {metrics['avg_negotiation_rounds']:.1f} ± {metrics['negotiation_rounds_std']:.1f}\n")
                f.write(f"  Seller advantage: {metrics['seller_advantage_mean']:.1f} ± {metrics['seller_advantage_std']:.1f}\n")
                f.write(f"  Buyer advantage: {metrics['buyer_advantage_mean']:.1f} ± {metrics['buyer_advantage_std']:.1f}\n")
                f.write(f"  Win rate (Seller): {metrics['win_rate_player1']:.3f}\n")
                f.write("\n")

        print(f"\nResults saved:")
        print(f"  Raw data: {raw_data_file}")
        print(f"  Summary: {summary_file}")
        print(f"  Readable: {readable_file}")
        print(f"  Behavior summary (JSON): {behavior_summary_file}")
        print(f"  Behavior summary (CSV): {behavior_csv_file}")
        plots_dir = Path(self.results_dir) / "plots"
        print(f"  Plots and heatmaps: {plots_dir}")

    def create_heatmap_data(self):
        """Create data for heatmap visualization"""
        # Get unique models and behaviors
        models = set()
        behaviors = set()

        for combo_key, metrics in self.summary.items():
            models.add(metrics["seller_model"])
            models.add(metrics["buyer_model"])
            behaviors.add(metrics["behavior"])

        models = sorted(list(models))
        behaviors = sorted(list(behaviors))

        # Create matrices for each behavior
        heatmap_data = {}

        for behavior in behaviors:
            win_rates = {}
            seller_advantages = {}
            buyer_advantages = {}
            for model1 in models:
                win_rates[model1] = {}
                seller_advantages[model1] = {}
                buyer_advantages[model1] = {}

                for model2 in models:
                    combo_key = f"{model1}_vs_{model2}_{behavior}"

                    if combo_key in self.summary:
                        metrics = self.summary[combo_key]
                        win_rates[model1][model2] = metrics.get("win_rate_player1", None)
                        seller_advantages[model1][model2] = metrics.get("seller_advantage_mean", None)
                        buyer_advantages[model1][model2] = metrics.get("buyer_advantage_mean", None)
                    else:
                        win_rates[model1][model2] = None
                        seller_advantages[model1][model2] = None
                        buyer_advantages[model1][model2] = None

            heatmap_data[behavior] = {
                "win_rates": win_rates,
                "seller_advantages": seller_advantages,
                "buyer_advantages": buyer_advantages,
                "models": models,
            }

        # Save heatmap data
        heatmap_file = self.results_dir / "heatmap_data.json"
        with open(heatmap_file, "w", encoding="utf-8") as f:
            json.dump(heatmap_data, f, indent=2, ensure_ascii=False)

        # Render heatmap images per behavior
        plots_dir = Path(self.results_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        for behavior, data in heatmap_data.items():
            models = data["models"]
            m = len(models)

            # Render heatmaps for each type
            for heatmap_type, title_suffix, value_format, cmap_name, vmin, vmax in [
                ("win_rates", "Win Rate (Seller)", lambda v: f"{v*100:.1f}%", 'YlGn', 0.0, 1.0),
                ("seller_advantages", "Average Seller Advantage", lambda v: f"{v:.1f}", 'Blues', None, None),
                ("buyer_advantages", "Average Buyer Advantage", lambda v: f"{v:.1f}", 'Oranges', None, None),
            ]:
                matrix = np.full((m, m), np.nan, dtype=float)
                for i, a in enumerate(models):
                    for j, b in enumerate(models):
                        val = data[heatmap_type].get(a, {}).get(b, None)
                        if val is None:
                            matrix[i, j] = np.nan
                        else:
                            matrix[i, j] = float(val)

                fig, ax = plt.subplots(figsize=(8, 6))
                cmap = plt.cm.get_cmap(cmap_name)
                im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap=cmap)
                ax.set_xticks(range(m))
                ax.set_yticks(range(m))
                ax.set_xticklabels(models, rotation=45, ha='right')
                ax.set_yticklabels(models)
                ax.set_title(f'{title_suffix} heatmap — {behavior}')
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ylabel = 'Win Rate' if 'win' in heatmap_type else 'Average Advantage'
                cbar.ax.set_ylabel(ylabel, rotation=270, labelpad=15)

                # Annotate heatmap cells with numeric values in large readable font.
                for i in range(m):
                    for j in range(m):
                        val = matrix[i, j]
                        if np.isnan(val):
                            txt = "-"
                            txt_color = 'gray'
                            fontsize = 10
                        else:
                            txt = value_format(val)
                            txt_color = 'white' if (val > 0.55 and "win" in heatmap_type) or (not np.isnan(matrix.max()) and val > matrix.max() * 0.7) else 'black'
                            fontsize = 12
                        ax.text(j, i, txt, ha='center', va='center', color=txt_color, fontsize=fontsize, fontweight='bold')

                heatmap_file_img = plots_dir / f'heatmap_{heatmap_type}_{behavior}.png'
                fig.tight_layout()
                fig.savefig(heatmap_file_img, dpi=150)
                plt.close(fig)

        print(f"  Heatmap data JSON: {heatmap_file}")
        print(f"  Heatmap images saved in: {plots_dir}")

        return heatmap_data


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python analyze_buysell_results.py <results_directory>")
        print("Example: python analyze_buysell_results.py .logs/final_buysell")
        return

    results_dir = sys.argv[1]

    if not os.path.exists(results_dir):
        print(f"Error: Results directory does not exist: {results_dir}")
        return

    print(f"Analyzing results in: {results_dir}")
    print("=" * 60)

    analyzer = BuySellResultsAnalyzer(results_dir)

    # Analyze all games
    raw_data = analyzer.analyze_all_games()

    if not raw_data:
        print("No valid game data found!")
        return

    # Calculate metrics
    summary = analyzer.calculate_metrics()

    # Aggregate behavior-level summaries
    behavior_summary = analyzer.calculate_behavior_summary()

    # Create bar plots and tables for behavior metrics
    analyzer.create_bar_plots_and_tables()

    # Save results
    analyzer.save_results()

    # Create heatmap data (and images)
    analyzer.create_heatmap_data()

    print(f"\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"Processed {len(raw_data)} games")
    print(f"Generated {len(summary)} combination summaries")
    print(f"Check the generated files in: {results_dir}")


if __name__ == "__main__":
    main()
