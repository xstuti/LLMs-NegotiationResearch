#!/usr/bin/env python3
"""
Comprehensive Ultimatum Game Analysis and Visualization Script

This script combines:
1. Results analysis from game directories
2. Heatmap generation
3. Final report generation

Properly handles behavior names with underscores like "Marwadi_Forced"
"""

import json
import os
import sys
from datetime import datetime
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


class UltimatumComprehensiveAnalyzer:
    """Combined analyzer for ultimatum game results"""

    # Known behaviors to help with parsing
    KNOWN_BEHAVIORS = [
        "Hindi",
        "Gujarati",
        "Marwadi",
        "Marwadi_Forced",
        "Punjabi",
        "Baseline",
    ]

    # Known model patterns
    KNOWN_MODELS = [
        "GPT-4o",
        "GPT-3.5",
        "GPT-oss",
        "Claude-3-Haiku",
        "Claude-3.5-Haiku",
        "Google-2.0-Flash",
    ]

    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.raw_data = []
        self.summary_data = {}
        self.models = set()
        self.behaviors = set()

    # ============= PART 1: RESULTS ANALYSIS =============

    def find_game_directories(self):
        """Find all game directories in the results folder"""
        game_dirs = []
        for item in self.results_dir.iterdir():
            if item.is_dir() and "_vs_" in item.name:
                game_dirs.append(item)

        print(f"Found {len(game_dirs)} game directories")
        return game_dirs

    def parse_directory_name(self, dir_name):
        """
        Parse directory name to extract model1, model2, behavior, and iteration
        Handles behavior names with underscores like "Marwadi_Forced"

        Format: Model1_vs_Model2_Behavior_iter_N
        Examples:
        - GPT-3.5_vs_GPT-4o_Gujarati_iter_1
        - GPT-4o_vs_Claude-3.5-Haiku_Marwadi_Forced_iter_2
        - GPT-oss_vs_GPT-4o_Baseline_iter_3
        """
        print(f"Parsing: {dir_name}")

        try:
            parts = dir_name.split("_")
            print(f"  Parts: {parts}")

            # Find key indices
            vs_index = parts.index("vs")
            iter_index = parts.index("iter")

            # Model 1: everything before 'vs', rejoin with underscores/dashes
            model1_parts = parts[:vs_index]
            model1 = "_".join(model1_parts)
            # Fix known model patterns
            for known_model in self.KNOWN_MODELS:
                if model1.replace("_", "-") == known_model or model1.replace(
                    "-", "_"
                ) == known_model.replace("-", "_"):
                    model1 = known_model
                    break

            # Everything between 'vs' and 'iter'
            middle_parts = parts[vs_index + 1 : iter_index]
            print(f"  Middle parts: {middle_parts}")

            # Try to identify model2 by matching known patterns
            model2 = None
            behavior_start_idx = 0

            # Check each known model pattern
            for known_model in self.KNOWN_MODELS:
                model_parts = known_model.replace("-", "_").split("_")
                if len(middle_parts) >= len(model_parts):
                    # Check if the first parts match this model
                    candidate = "_".join(middle_parts[: len(model_parts)])
                    if candidate.replace(
                        "_", "-"
                    ) == known_model or candidate == known_model.replace("-", "_"):
                        model2 = known_model
                        behavior_start_idx = len(model_parts)
                        break

            if model2 is None:
                # Fallback: assume first part is model2
                model2 = middle_parts[0]
                behavior_start_idx = 1

            # Remaining parts form the behavior name
            behavior_parts = middle_parts[behavior_start_idx:]
            behavior_candidate = (
                "_".join(behavior_parts) if behavior_parts else "Unknown"
            )

            # Check against known behaviors (prefer exact match)
            behavior = behavior_candidate
            for known_behavior in self.KNOWN_BEHAVIORS:
                if behavior_candidate == known_behavior:
                    behavior = known_behavior
                    break
                elif (
                    behavior_candidate.replace("_", "").lower()
                    == known_behavior.replace("_", "").lower()
                ):
                    behavior = known_behavior
                    break

            # Iteration number
            iteration = int(parts[-1])

            print(f"  ✓ Parsed: {model1} vs {model2} | {behavior} | iter {iteration}")
            return model1, model2, behavior, iteration

        except Exception as e:
            print(f"  ✗ Error parsing {dir_name}: {e}")
            # Try fallback parsing
            return self._fallback_parse(dir_name)

    def _fallback_parse(self, dir_name):
        """Fallback parsing method"""
        try:
            # Remove iteration part
            base = dir_name.rsplit("_iter_", 1)[0]
            iteration = int(dir_name.rsplit("_iter_", 1)[1])

            # Split by _vs_
            parts = base.split("_vs_")
            if len(parts) != 2:
                return "Unknown", "Unknown", "Unknown", 1

            model1_str = parts[0]
            rest_str = parts[1]

            # Match model1 against known models
            model1 = model1_str
            for known in self.KNOWN_MODELS:
                if model1_str.replace("_", "-") == known:
                    model1 = known
                    break

            # For rest_str, try to match against known models from the start
            model2 = None
            behavior = None
            rest_parts = rest_str.split("_")

            for known_model in self.KNOWN_MODELS:
                model_parts = known_model.replace("-", "_").split("_")
                if len(rest_parts) >= len(model_parts):
                    candidate = "_".join(rest_parts[: len(model_parts)])
                    if candidate.replace("_", "-") == known_model:
                        model2 = known_model
                        behavior_parts = rest_parts[len(model_parts) :]
                        behavior = (
                            "_".join(behavior_parts) if behavior_parts else "Unknown"
                        )
                        break

            if model2 is None:
                # Last resort: split at last known behavior
                for known_behavior in self.KNOWN_BEHAVIORS:
                    if known_behavior in rest_str:
                        idx = rest_str.rfind(known_behavior)
                        model2 = rest_str[:idx].rstrip("_")
                        behavior = known_behavior
                        break

            if model2 is None or behavior is None:
                model2 = rest_parts[0]
                behavior = (
                    "_".join(rest_parts[1:]) if len(rest_parts) > 1 else "Unknown"
                )

            print(
                f"  ⚠ Fallback parsed: {model1} vs {model2} | {behavior} | iter {iteration}"
            )
            return model1, model2, behavior, iteration

        except Exception as e:
            print(f"  ✗ Fallback parsing failed: {e}")
            return "Unknown", "Unknown", "Unknown", 1

    def find_game_state_file(self, game_dir):
        """Find the game state JSON file in the game directory"""
        for file in game_dir.iterdir():
            if file.name.startswith("game_state_") and file.suffix == ".json":
                return file
        return None

    def extract_game_data(self, game_dir):
        """Extract data from a single game directory"""
        game_state_file = self.find_game_state_file(game_dir)

        if not game_state_file:
            print(f"  No game state file found in {game_dir.name}")
            return None

        try:
            with open(game_state_file, "r", encoding="utf-8") as f:
                game_state = json.load(f)

            # Extract key information
            game_data = {
                "directory": game_dir.name,
                "player1_final": game_state.get("player_resources", {})
                .get("AgentOne", {})
                .get("Dollars", 0),
                "player2_final": game_state.get("player_resources", {})
                .get("AgentTwo", {})
                .get("Dollars", 0),
                "completed": True,
            }

            # Extract turn history
            turns = game_state.get("turn_history", [])
            game_data["total_turns"] = len(turns)

            # Extract offers and responses
            offers = []
            responses = []

            for turn in turns:
                action = turn.get("action", {})
                if action.get("type") == "Offer":
                    offer_amount = action.get("parameters", {}).get("dollars", 0)
                    offers.append(offer_amount)
                elif action.get("type") == "Response":
                    response = action.get("parameters", {}).get("accept", False)
                    responses.append(response)

            game_data["offers"] = offers
            game_data["responses"] = responses
            game_data["initial_offer"] = offers[0] if offers else None
            game_data["accepted"] = any(responses) if responses else False
            game_data["rejected"] = not any(responses) if responses else False

            return game_data

        except Exception as e:
            print(f"  Error extracting data from {game_dir.name}: {e}")
            return None

    def analyze_all_games(self):
        """Analyze all game directories"""
        print("\n" + "=" * 60)
        print("ANALYZING GAME RESULTS")
        print("=" * 60 + "\n")

        game_dirs = self.find_game_directories()

        for game_dir in game_dirs:
            print(f"\nProcessing: {game_dir.name}")

            # Parse directory name
            model1, model2, behavior, iteration = self.parse_directory_name(
                game_dir.name
            )

            # Extract game data
            game_data = self.extract_game_data(game_dir)

            if game_data:
                game_data.update(
                    {
                        "model1": model1,
                        "model2": model2,
                        "behavior": behavior,
                        "iteration": iteration,
                    }
                )
                self.raw_data.append(game_data)
                self.models.add(model1)
                self.models.add(model2)
                self.behaviors.add(behavior)

        print(f"\n✓ Successfully analyzed {len(self.raw_data)} games")
        print(f"✓ Models found: {sorted(self.models)}")
        print(f"✓ Behaviors found: {sorted(self.behaviors)}")

        return self.raw_data

    def calculate_metrics(self):
        """Calculate summary metrics for each model-behavior combination"""
        print("\n" + "=" * 60)
        print("CALCULATING METRICS")
        print("=" * 60 + "\n")

        # Group by model1, model2, and behavior
        grouped = {}

        for game in self.raw_data:
            key = (game["model1"], game["model2"], game["behavior"])
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(game)

        # Calculate metrics for each group
        for (model1, model2, behavior), games in grouped.items():
            combo_key = f"{model1}_vs_{model2}_{behavior}"

            total_games = len(games)
            completed_games = sum(1 for g in games if g["completed"])

            # Player 1 metrics
            player1_payoffs = [g["player1_final"] for g in games]
            player2_payoffs = [g["player2_final"] for g in games]

            player1_wins = sum(
                1 for g in games if g["player1_final"] > g["player2_final"]
            )
            player2_wins = sum(
                1 for g in games if g["player2_final"] > g["player1_final"]
            )
            ties = sum(1 for g in games if g["player1_final"] == g["player2_final"])

            # Offer and acceptance metrics
            initial_offers = [
                g["initial_offer"] for g in games if g["initial_offer"] is not None
            ]
            accepts = sum(1 for g in games if g.get("accepted", False))
            rejects = sum(1 for g in games if g.get("rejected", False))

            metrics = {
                "model1": model1,
                "model2": model2,
                "behavior": behavior,
                "total_games": total_games,
                "completed_games": completed_games,
                "player1_wins": player1_wins,
                "player2_wins": player2_wins,
                "ties": ties,
                "win_rate_player1": player1_wins / total_games
                if total_games > 0
                else 0,
                "win_rate_player2": player2_wins / total_games
                if total_games > 0
                else 0,
                "player1_payoff_avg": (
                    sum(player1_payoffs) / len(player1_payoffs)
                    if player1_payoffs
                    else 0
                ),
                "player1_payoff_std": (
                    float(np.std(player1_payoffs)) if player1_payoffs else 0
                ),
                "player2_payoff_avg": (
                    sum(player2_payoffs) / len(player2_payoffs)
                    if player2_payoffs
                    else 0
                ),
                "player2_payoff_std": (
                    float(np.std(player2_payoffs)) if player2_payoffs else 0
                ),
                "initial_offer_avg": (
                    sum(initial_offers) / len(initial_offers) if initial_offers else 0
                ),
                "initial_offer_std": (
                    float(np.std(initial_offers)) if initial_offers else 0
                ),
                "accepts": accepts,
                "rejects": rejects,
                "acceptance_rate": accepts / total_games if total_games > 0 else 0,
            }

            self.summary_data[combo_key] = metrics

            print(f"✓ {combo_key}")
            print(
                f"   Games: {total_games}, Win Rate P1: {metrics['win_rate_player1']:.2f}, "
                f"Acceptance: {metrics['acceptance_rate']:.2f}"
            )

        return self.summary_data

    def save_results(self):
        """Save analysis results"""
        # Save raw data
        raw_data_file = self.results_dir / "raw_game_data.json"
        with open(raw_data_file, "w", encoding="utf-8") as f:
            json.dump(self.raw_data, f, indent=2)

        # Save summary
        summary_file = self.results_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(self.summary_data, f, indent=2)

        print(f"\n✓ Raw data saved to: {raw_data_file}")
        print(f"✓ Summary saved to: {summary_file}")

    # ============= PART 2: HEATMAP GENERATION =============

    def extract_matrices(self):
        """Extract matrices for visualization"""
        models = sorted(list(self.models))
        behaviors = sorted(list(self.behaviors))

        print(f"\nCreating matrices for:")
        print(f"  Models: {models}")
        print(f"  Behaviors: {behaviors}")

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

    def create_heatmaps(self, matrices, models, behaviors):
        """Create academic-style heatmaps"""
        print("\n" + "=" * 60)
        print("CREATING HEATMAPS")
        print("=" * 60 + "\n")

        n_behaviors = len(behaviors)

        # Create figure with 2 columns (win rates and payoffs)
        fig, axes = plt.subplots(n_behaviors, 2, figsize=(12, 4 * n_behaviors))

        if n_behaviors == 1:
            axes = axes.reshape(1, -1)

        for idx, behavior in enumerate(behaviors):
            # Format behavior name for display
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
        print(f"✓ Heatmaps saved to: {output_file}")
        plt.close()

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

        print(f"✓ Summary table saved to: {csv_file}")
        return df

    # ============= PART 3: REPORT GENERATION =============

    def generate_report(self):
        """Generate comprehensive text report"""
        print("\n" + "=" * 60)
        print("GENERATING REPORT")
        print("=" * 60 + "\n")

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(
            "ULTIMATUM GAME SOCIAL BEHAVIOR ANALYSIS - COMPREHENSIVE REPORT"
        )
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        report_lines.append(f"Results Directory: {self.results_dir}")
        report_lines.append("")

        # Overview
        report_lines.append("-" * 80)
        report_lines.append("OVERVIEW")
        report_lines.append("-" * 80)
        report_lines.append(f"Total Games Analyzed: {len(self.raw_data)}")
        report_lines.append(f"Models Tested: {len(self.models)}")
        report_lines.append(f"  - {', '.join(sorted(self.models))}")
        report_lines.append(f"Behaviors Tested: {len(self.behaviors)}")
        report_lines.append(f"  - {', '.join(sorted(self.behaviors))}")
        report_lines.append(f"Unique Combinations: {len(self.summary_data)}")
        report_lines.append("")

        # Completion rate
        completed = sum(1 for g in self.raw_data if g["completed"])
        completion_rate = (completed / len(self.raw_data) * 100) if self.raw_data else 0
        report_lines.append(f"Game Completion Rate: {completion_rate:.1f}%")
        report_lines.append("")

        # Behavior-specific analysis
        for behavior in sorted(self.behaviors):
            report_lines.append("")
            report_lines.append("=" * 80)
            report_lines.append(f"BEHAVIOR: {behavior.upper().replace('_', ' ')}")
            report_lines.append("=" * 80)
            report_lines.append("")

            # Get all combinations for this behavior
            behavior_combos = {
                k: v for k, v in self.summary_data.items() if v["behavior"] == behavior
            }

            if not behavior_combos:
                report_lines.append("No data available for this behavior.")
                continue

            # Summary statistics
            total_games_behavior = sum(
                v["total_games"] for v in behavior_combos.values()
            )
            avg_acceptance = (
                sum(
                    v["acceptance_rate"] * v["total_games"]
                    for v in behavior_combos.values()
                )
                / total_games_behavior
            )
            avg_initial_offer = (
                sum(
                    v["initial_offer_avg"] * v["total_games"]
                    for v in behavior_combos.values()
                )
                / total_games_behavior
            )

            report_lines.append(f"Total Games: {total_games_behavior}")
            report_lines.append(f"Average Acceptance Rate: {avg_acceptance:.2%}")
            report_lines.append(f"Average Initial Offer: ${avg_initial_offer:.2f}")
            report_lines.append("")

            # Model performance table
            report_lines.append("-" * 80)
            report_lines.append("Model Combinations Performance")
            report_lines.append("-" * 80)
            report_lines.append(
                f"{'Player 1':<20} {'Player 2':<20} {'Games':<8} {'Win Rate':<10} {'Accept Rate':<12} {'Avg Payoff P1':<15}"
            )
            report_lines.append("-" * 80)

            for combo_key in sorted(behavior_combos.keys()):
                metrics = behavior_combos[combo_key]
                report_lines.append(
                    f"{metrics['model1']:<20} {metrics['model2']:<20} "
                    f"{metrics['total_games']:<8} {metrics['win_rate_player1']:<10.2f} "
                    f"{metrics['acceptance_rate']:<12.2f} ${metrics['player1_payoff_avg']:<14.2f}"
                )

            report_lines.append("")

            # Key findings for this behavior
            report_lines.append("-" * 80)
            report_lines.append("Key Findings")
            report_lines.append("-" * 80)

            # Best and worst performers
            if behavior_combos:
                best_win_rate = max(
                    behavior_combos.items(), key=lambda x: x[1]["win_rate_player1"]
                )
                worst_win_rate = min(
                    behavior_combos.items(), key=lambda x: x[1]["win_rate_player1"]
                )
                best_payoff = max(
                    behavior_combos.items(), key=lambda x: x[1]["player1_payoff_avg"]
                )
                highest_acceptance = max(
                    behavior_combos.items(), key=lambda x: x[1]["acceptance_rate"]
                )

                report_lines.append(
                    f"• Highest Win Rate: {best_win_rate[1]['model1']} vs {best_win_rate[1]['model2']} "
                    f"({best_win_rate[1]['win_rate_player1']:.2%})"
                )
                report_lines.append(
                    f"• Lowest Win Rate: {worst_win_rate[1]['model1']} vs {worst_win_rate[1]['model2']} "
                    f"({worst_win_rate[1]['win_rate_player1']:.2%})"
                )
                report_lines.append(
                    f"• Highest Average Payoff: {best_payoff[1]['model1']} vs {best_payoff[1]['model2']} "
                    f"(${best_payoff[1]['player1_payoff_avg']:.2f})"
                )
                report_lines.append(
                    f"• Highest Acceptance Rate: {highest_acceptance[1]['model1']} vs {highest_acceptance[1]['model2']} "
                    f"({highest_acceptance[1]['acceptance_rate']:.2%})"
                )

            report_lines.append("")

        # Cross-behavior comparison
        if len(self.behaviors) > 1:
            report_lines.append("")
            report_lines.append("=" * 80)
            report_lines.append("CROSS-BEHAVIOR COMPARISON")
            report_lines.append("=" * 80)
            report_lines.append("")

            for behavior in sorted(self.behaviors):
                behavior_combos = {
                    k: v
                    for k, v in self.summary_data.items()
                    if v["behavior"] == behavior
                }
                if behavior_combos:
                    total_games = sum(
                        v["total_games"] for v in behavior_combos.values()
                    )
                    avg_acceptance = (
                        sum(
                            v["acceptance_rate"] * v["total_games"]
                            for v in behavior_combos.values()
                        )
                        / total_games
                    )
                    avg_offer = (
                        sum(
                            v["initial_offer_avg"] * v["total_games"]
                            for v in behavior_combos.values()
                        )
                        / total_games
                    )

                    report_lines.append(f"{behavior.replace('_', ' ')}:")
                    report_lines.append(f"  Acceptance Rate: {avg_acceptance:.2%}")
                    report_lines.append(f"  Average Initial Offer: ${avg_offer:.2f}")
                    report_lines.append("")

        # Final summary
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("CONCLUSIONS")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append(
            "This analysis reveals patterns in how different AI models negotiate under"
        )
        report_lines.append("various social behavior contexts. Key insights include:")
        report_lines.append("")

        # Overall stats
        all_acceptance_rates = [
            v["acceptance_rate"] for v in self.summary_data.values()
        ]
        all_initial_offers = [
            v["initial_offer_avg"] for v in self.summary_data.values()
        ]

        if all_acceptance_rates:
            report_lines.append(
                f"• Overall acceptance rate across all combinations: {np.mean(all_acceptance_rates):.2%}"
            )
        if all_initial_offers:
            report_lines.append(
                f"• Overall average initial offer: ${np.mean(all_initial_offers):.2f}"
            )

        report_lines.append("")
        report_lines.append("=" * 80)

        # Save report
        report_file = self.results_dir / "comprehensive_report.txt"
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

        print(f"✓ Report saved to: {report_file}")
        return report_file

    # ============= MAIN EXECUTION =============

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("\n" + "=" * 80)
        print(" " * 20 + "ULTIMATUM GAME COMPREHENSIVE ANALYSIS")
        print("=" * 80)

        # Part 1: Analyze results
        self.analyze_all_games()
        self.calculate_metrics()
        self.save_results()

        # Part 2: Create visualizations
        matrices, models, behaviors = self.extract_matrices()
        self.create_heatmaps(matrices, models, behaviors)
        self.create_summary_table(matrices, models, behaviors)

        # Part 3: Generate report
        self.generate_report()

        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"\nAll results saved in: {self.results_dir}")
        print("\nGenerated files:")
        print("  ✓ raw_game_data.json - Raw game data")
        print("  ✓ summary.json - Summary metrics")
        print("  ✓ final_heatmaps.png - Heatmap visualizations")
        print("  ✓ summary_table.csv - Summary table")
        print("  ✓ comprehensive_report.txt - Detailed text report")
        print("\n")


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python analyze_and_visualize.py <results_directory>")
        print(
            "Example: python analyze_and_visualize.py .logs/ultimatum_social_behavior_20251130_180851"
        )
        return

    results_dir = sys.argv[1]

    if not Path(results_dir).exists():
        print(f"Error: Results directory does not exist: {results_dir}")
        return

    try:
        analyzer = UltimatumComprehensiveAnalyzer(results_dir)
        analyzer.run_complete_analysis()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
