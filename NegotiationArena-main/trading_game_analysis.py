#!/usr/bin/env python3
"""
Comprehensive analysis script for Trading Game Negotiation Arena results.
Generates heatmaps, graphs, and statistical summaries.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("default")

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "axes.labelweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.titlesize": 20,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BAR_COLORS = [
    "#3b4cc0",  # deep blue
    "#2c728e",  # blue-green
    "#21918c",  # teal
    "#28ae80",  # green
    "#5ec962",  # light green
    "#b8de29",  # yellow-green
]


class TradingGameAnalyzer:
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.data = defaultdict(lambda: defaultdict(list))
        self.models = ["Claude-3.5-Haiku", "Claude-3-Haiku", "GPT-4o", "GPT-3.5"]
        self.languages = ["English", "Hindi", "Gujarati", "Marwadi", "Punjabi"]
        self.all_results = []

    def extract_game_data(self, game_state_path: Path) -> Dict[str, Any]:
        """Extract relevant data from a single game_state.json file."""
        try:
            with open(game_state_path, "r") as f:
                data = json.load(f)

            # Find the END state in game_state array
            game_states = data.get("game_state", [])
            end_state = None
            for state in game_states:
                if state.get("current_iteration") == "END":
                    end_state = state
                    break

            if not end_state:
                print(f"No END state found in {game_state_path}")
                return None

            # Extract from summary
            summary = end_state.get("summary", {})
            final_response = summary.get("final_response", "")
            player_outcome = summary.get("player_outcome", [False, False])
            proposed_trade = summary.get("proposed_trade", {})
            initial_resources = summary.get("initial_resources", [])
            final_resources = summary.get("final_resources", [])
            player_goals = summary.get("player_goals", [])

            # Helper function to extract resource values
            def get_resource_values(resource_dict):
                if isinstance(resource_dict, dict) and "_value" in resource_dict:
                    vals = resource_dict["_value"]
                    return vals.get("X", 0), vals.get("Y", 0)
                return 0, 0

            # Calculate resource distances from goals
            goal_distances = []
            for i in range(2):
                if i < len(player_goals) and i < len(final_resources):
                    goal = player_goals[i]
                    final = final_resources[i]

                    # Extract goal values
                    if isinstance(goal, dict) and "_value" in goal:
                        goal_val = goal["_value"]
                        if isinstance(goal_val, dict) and "_value" in goal_val:
                            goal_x = goal_val["_value"].get("X", 0)
                            goal_y = goal_val["_value"].get("Y", 0)
                        else:
                            goal_x, goal_y = 0, 0
                    else:
                        goal_x, goal_y = 0, 0

                    final_x, final_y = get_resource_values(final)

                    # Calculate Euclidean distance from goal
                    distance = np.sqrt(
                        (final_x - goal_x) ** 2 + (final_y - goal_y) ** 2
                    )
                    goal_distances.append(distance)
                else:
                    goal_distances.append(float("inf"))

            # Calculate resource balance (how balanced are X and Y)
            resource_balances = []
            for i in range(2):
                if i < len(final_resources):
                    x, y = get_resource_values(final_resources[i])
                    # Balance is measured as the ratio of min to max (closer to 1 is more balanced)
                    if max(x, y) > 0:
                        balance = min(x, y) / max(x, y)
                    else:
                        balance = 0
                    resource_balances.append(balance)
                else:
                    resource_balances.append(0)

            # Calculate total resources gained/lost
            resource_changes = []
            for i in range(2):
                if i < len(initial_resources) and i < len(final_resources):
                    init_x, init_y = get_resource_values(initial_resources[i])
                    final_x, final_y = get_resource_values(final_resources[i])
                    total_change = (final_x + final_y) - (init_x + init_y)
                    resource_changes.append(total_change)
                else:
                    resource_changes.append(0)

            # Extract trade volume if trade occurred
            trade_volume = 0
            if (
                proposed_trade
                and isinstance(proposed_trade, dict)
                and "_value" in proposed_trade
            ):
                trade_val = proposed_trade["_value"]
                if isinstance(trade_val, dict):
                    # Sum up all resources being traded
                    for player_key in ["RED", "BLUE"]:
                        if player_key in trade_val:
                            player_trade = trade_val[player_key]
                            if (
                                isinstance(player_trade, dict)
                                and "_value" in player_trade
                            ):
                                for resource, amount in player_trade["_value"].items():
                                    trade_volume += (
                                        abs(amount)
                                        if isinstance(amount, (int, float))
                                        else 0
                                    )

            # Calculate fairness metric (difference in outcomes)
            fairness = (
                abs(goal_distances[0] - goal_distances[1])
                if len(goal_distances) == 2
                else 0
            )

            # Calculate negotiation efficiency (rounds used)
            num_rounds = len(
                [
                    s
                    for s in game_states
                    if s.get("current_iteration") not in ["START", "END"]
                ]
            )

            return {
                "final_response": final_response,
                "player1_outcome": player_outcome[0]
                if len(player_outcome) > 0
                else False,
                "player2_outcome": player_outcome[1]
                if len(player_outcome) > 1
                else False,
                "both_goals_reached": all(player_outcome)
                if len(player_outcome) == 2
                else False,
                "at_least_one_goal": any(player_outcome)
                if len(player_outcome) == 2
                else False,
                "accepted": final_response == "ACCEPT",
                "player1_goal_distance": goal_distances[0]
                if len(goal_distances) > 0
                else float("inf"),
                "player2_goal_distance": goal_distances[1]
                if len(goal_distances) > 1
                else float("inf"),
                "avg_goal_distance": np.mean(goal_distances)
                if goal_distances
                else float("inf"),
                "player1_balance": resource_balances[0]
                if len(resource_balances) > 0
                else 0,
                "player2_balance": resource_balances[1]
                if len(resource_balances) > 1
                else 0,
                "avg_balance": np.mean(resource_balances) if resource_balances else 0,
                "player1_resource_change": resource_changes[0]
                if len(resource_changes) > 0
                else 0,
                "player2_resource_change": resource_changes[1]
                if len(resource_changes) > 1
                else 0,
                "trade_volume": trade_volume,
                "fairness": fairness,
                "num_rounds": num_rounds,
                "summary": summary,
            }
        except Exception as e:
            print(f"Error processing {game_state_path}: {e}")
            import traceback

            traceback.print_exc()
            return None

    def load_all_results(self):
        """Load all game results from the directory structure."""
        print("Loading results from:", self.results_dir)

        # Iterate through all subdirectories
        for subdir in self.results_dir.iterdir():
            if not subdir.is_dir():
                continue

            # Parse directory name: Model1_Model2_Language_iterN
            parts = subdir.name.split("_")

            # Find iter position
            iter_idx = -1
            iteration = None
            for i, p in enumerate(parts):
                if p.startswith("iter"):
                    iter_idx = i
                    try:
                        iteration = int(p[4:])
                    except:
                        continue
                    break

            if iter_idx == -1:
                continue

            try:
                # Language is right before iter
                language = parts[iter_idx - 1]

                # Models are everything before language
                model_parts = parts[: iter_idx - 1]

                # Split models - typically model1_model2
                # We need to figure out where one model ends and another begins
                # Try to match known models
                model1 = None
                model2 = None

                # Try different split points
                for split in range(1, len(model_parts)):
                    potential_model1 = "_".join(model_parts[:split]).replace("_", "-")
                    potential_model2 = "_".join(model_parts[split:]).replace("_", "-")

                    if (
                        potential_model1 in self.models
                        and potential_model2 in self.models
                    ):
                        model1 = potential_model1
                        model2 = potential_model2
                        break

                if model1 is None or model2 is None:
                    # Fallback: assume equal split
                    mid = len(model_parts) // 2
                    model1 = "_".join(model_parts[:mid]).replace("_", "-")
                    model2 = "_".join(model_parts[mid:]).replace("_", "-")

                # Find game_state.json in subdirectories
                game_state_files = list(subdir.glob("*/game_state.json"))

                for game_state_file in game_state_files:
                    game_data = self.extract_game_data(game_state_file)

                    if game_data:
                        result = {
                            "model1": model1,
                            "model2": model2,
                            "language": language,
                            "iteration": iteration,
                            **game_data,
                        }
                        self.all_results.append(result)

                        # Store in structured format
                        key = f"{model1}_vs_{model2}_{language}"
                        self.data[key]["accepted"].append(game_data["accepted"])
                        self.data[key]["both_goals_reached"].append(
                            game_data["both_goals_reached"]
                        )
                        self.data[key]["at_least_one_goal"].append(
                            game_data["at_least_one_goal"]
                        )
                        self.data[key]["player1_goal_distance"].append(
                            game_data["player1_goal_distance"]
                        )
                        self.data[key]["player2_goal_distance"].append(
                            game_data["player2_goal_distance"]
                        )
                        self.data[key]["avg_goal_distance"].append(
                            game_data["avg_goal_distance"]
                        )
                        self.data[key]["avg_balance"].append(game_data["avg_balance"])
                        self.data[key]["trade_volume"].append(game_data["trade_volume"])
                        self.data[key]["fairness"].append(game_data["fairness"])
                        self.data[key]["num_rounds"].append(game_data["num_rounds"])
                        self.data[key]["model1"] = model1
                        self.data[key]["model2"] = model2
                        self.data[key]["language"] = language

            except (ValueError, IndexError) as e:
                print(f"Skipping directory {subdir.name}: {e}")
                continue

        print(f"Loaded {len(self.all_results)} game results")

    def compute_statistics(self) -> Dict:
        """Compute comprehensive statistics for all model-language combinations."""
        stats = {}

        for key, data in self.data.items():
            accepted = data["accepted"]
            both_goals = [x for x in data["both_goals_reached"]]
            at_least_one = [x for x in data["at_least_one_goal"]]
            goal_distances = [x for x in data["avg_goal_distance"] if x != float("inf")]
            balances = [x for x in data["avg_balance"] if x > 0]
            trade_volumes = [x for x in data["trade_volume"] if x > 0]
            fairness_vals = [x for x in data["fairness"] if x != float("inf")]
            rounds = [x for x in data["num_rounds"]]

            stats[key] = {
                "model1": data["model1"],
                "model2": data["model2"],
                "language": data["language"],
                "num_games": len(accepted),
                # Acceptance and success rates
                "acceptance_rate": sum(accepted) / len(accepted) if accepted else 0,
                "both_goals_rate": sum(both_goals) / len(both_goals)
                if both_goals
                else 0,
                "at_least_one_goal_rate": sum(at_least_one) / len(at_least_one)
                if at_least_one
                else 0,
                # Goal distance metrics (lower is better)
                "avg_goal_distance": np.mean(goal_distances) if goal_distances else 0,
                "std_goal_distance": np.std(goal_distances) if goal_distances else 0,
                "min_goal_distance": np.min(goal_distances) if goal_distances else 0,
                "max_goal_distance": np.max(goal_distances) if goal_distances else 0,
                # Resource balance metrics (higher is better)
                "avg_balance": np.mean(balances) if balances else 0,
                "std_balance": np.std(balances) if balances else 0,
                # Trade volume metrics
                "avg_trade_volume": np.mean(trade_volumes) if trade_volumes else 0,
                "std_trade_volume": np.std(trade_volumes) if trade_volumes else 0,
                # Fairness metrics (lower is better)
                "avg_fairness": np.mean(fairness_vals) if fairness_vals else 0,
                "std_fairness": np.std(fairness_vals) if fairness_vals else 0,
                # Efficiency metrics
                "avg_rounds": np.mean(rounds) if rounds else 0,
                "std_rounds": np.std(rounds) if rounds else 0,
                # Raw values for detailed analysis
                "values": {
                    "goal_distances": goal_distances,
                    "balances": balances,
                    "trade_volumes": trade_volumes,
                    "fairness": fairness_vals,
                    "rounds": rounds,
                },
            }

        return stats

    def create_consolidated_heatmaps(
        self, language: str, stats: Dict, output_dir: Path
    ):
        """Create a single image with 6 heatmaps (3x2) for a language."""
        metrics = [
            ("acceptance_rate", "Acceptance Rate", "YlGn"),
            ("both_goals_rate", "Both Goals Reached Rate", "YlGn"),
            ("avg_goal_distance", "Avg Distance from Goal (lower=better)", "YlOrRd_r"),
            ("avg_balance", "Avg Resource Balance (higher=better)", "YlGn"),
            ("avg_trade_volume", "Average Trade Volume", "Blues"),
            ("avg_fairness", "Fairness (lower=better)", "YlOrRd_r"),
        ]

        fig, axes = plt.subplots(3, 2, figsize=(18, 24))
        fig.suptitle(
            f"{language} - Trading Game Model Comparison Heatmaps",
            fontsize=20,
            fontweight="bold",
            y=0.995,
        )

        for idx, (metric, title, cmap) in enumerate(metrics):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]

            # Create matrix for the heatmap
            matrix = np.zeros((len(self.models), len(self.models)))

            for i, model1 in enumerate(self.models):
                for j, model2 in enumerate(self.models):
                    key = f"{model1}_vs_{model2}_{language}"
                    if key in stats:
                        matrix[i, j] = stats[key][metric]
                    else:
                        matrix[i, j] = np.nan

            # Create mask for NaN values
            mask = np.isnan(matrix)

            # Determine if we need a center point
            center = None
            if "rate" in metric or "balance" in metric or "volume" in metric:
                vmin, vmax = None, None
            else:
                vmin, vmax = None, None

            sns.heatmap(
                matrix,
                annot=True,
                fmt=".3f",
                cmap=cmap,
                xticklabels=self.models,
                yticklabels=self.models,
                center=center,
                mask=mask,
                cbar_kws={"label": title},
                ax=ax,
                vmin=vmin,
                vmax=vmax,
            )

            ax.set_xlabel("Player 2 Model", fontsize=11, fontweight="bold")
            ax.set_ylabel("Player 1 Model", fontsize=11, fontweight="bold")
            ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

        plt.tight_layout()
        filename = f"heatmap_{language}_all_metrics.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Created consolidated heatmap: {filename}")

    def create_all_heatmaps(self, stats: Dict, output_dir: Path):
        """Create consolidated heatmaps for all languages."""
        for language in self.languages:
            self.create_consolidated_heatmaps(language, stats, output_dir)

    def create_language_comparison_graphs(self, stats: Dict, output_dir: Path):
        """Create comprehensive language-wise comparison graphs."""

        # For each model combination, compare across languages
        model_combos = set()
        for key in stats.keys():
            parts = key.rsplit("_", 1)
            if len(parts) == 2:
                model_combos.add(parts[0])

        for combo in sorted(model_combos):
            # Collect data for this combo across all languages
            lang_data = {
                "acceptance": [],
                "both_goals": [],
                "at_least_one": [],
                "goal_distance": [],
                "balance": [],
                "trade_volume": [],
                "fairness": [],
                "rounds": [],
            }
            languages_present = []

            for lang in self.languages:
                key = f"{combo}_{lang}"
                if key in stats:
                    languages_present.append(lang)
                    lang_data["acceptance"].append(stats[key]["acceptance_rate"])
                    lang_data["both_goals"].append(stats[key]["both_goals_rate"])
                    lang_data["at_least_one"].append(
                        stats[key]["at_least_one_goal_rate"]
                    )
                    lang_data["goal_distance"].append(stats[key]["avg_goal_distance"])
                    lang_data["balance"].append(stats[key]["avg_balance"])
                    lang_data["trade_volume"].append(stats[key]["avg_trade_volume"])
                    lang_data["fairness"].append(stats[key]["avg_fairness"])
                    lang_data["rounds"].append(stats[key]["avg_rounds"])

            if not languages_present:
                continue

            # Create figure with subplots (4x2 grid)
            fig, axes = plt.subplots(4, 2, figsize=(16, 20))
            fig.suptitle(
                f"{combo} - Language Comparison", fontsize=18, fontweight="bold"
            )

            x = np.arange(len(languages_present))
            width = 0.6

            # Acceptance rate
            axes[0, 0].bar(
                x, lang_data["acceptance"], width, color="lightgreen", alpha=0.8
            )
            axes[0, 0].set_ylabel("Rate", fontweight="bold")
            axes[0, 0].set_title("Acceptance Rate by Language")
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(languages_present, rotation=45, ha="right")
            axes[0, 0].set_ylim([0, 1.1])
            axes[0, 0].grid(axis="y", alpha=0.3)
            for i, v in enumerate(lang_data["acceptance"]):
                axes[0, 0].text(i, v + 0.02, f"{v:.2%}", ha="center", fontsize=9)

            # Both goals reached
            axes[0, 1].bar(
                x, lang_data["both_goals"], width, color="mediumseagreen", alpha=0.8
            )
            axes[0, 1].set_ylabel("Rate", fontweight="bold")
            axes[0, 1].set_title("Both Goals Reached Rate by Language")
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(languages_present, rotation=45, ha="right")
            axes[0, 1].set_ylim([0, 1.1])
            axes[0, 1].grid(axis="y", alpha=0.3)
            for i, v in enumerate(lang_data["both_goals"]):
                axes[0, 1].text(i, v + 0.02, f"{v:.2%}", ha="center", fontsize=9)

            # At least one goal
            axes[1, 0].bar(
                x, lang_data["at_least_one"], width, color="darkseagreen", alpha=0.8
            )
            axes[1, 0].set_ylabel("Rate", fontweight="bold")
            axes[1, 0].set_title("At Least One Goal Reached Rate by Language")
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(languages_present, rotation=45, ha="right")
            axes[1, 0].set_ylim([0, 1.1])
            axes[1, 0].grid(axis="y", alpha=0.3)
            for i, v in enumerate(lang_data["at_least_one"]):
                axes[1, 0].text(i, v + 0.02, f"{v:.2%}", ha="center", fontsize=9)

            # Goal distance (lower is better)
            axes[1, 1].bar(
                x, lang_data["goal_distance"], width, color="coral", alpha=0.8
            )
            axes[1, 1].set_ylabel("Distance", fontweight="bold")
            axes[1, 1].set_title("Avg Distance from Goal by Language (lower=better)")
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(languages_present, rotation=45, ha="right")
            axes[1, 1].grid(axis="y", alpha=0.3)
            for i, v in enumerate(lang_data["goal_distance"]):
                axes[1, 1].text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=9)

            # Resource balance (higher is better)
            axes[2, 0].bar(x, lang_data["balance"], width, color="skyblue", alpha=0.8)
            axes[2, 0].set_ylabel("Balance Score", fontweight="bold")
            axes[2, 0].set_title("Avg Resource Balance by Language (higher=better)")
            axes[2, 0].set_xticks(x)
            axes[2, 0].set_xticklabels(languages_present, rotation=45, ha="right")
            axes[2, 0].set_ylim([0, 1.1])
            axes[2, 0].grid(axis="y", alpha=0.3)
            for i, v in enumerate(lang_data["balance"]):
                axes[2, 0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

            # Trade volume
            axes[2, 1].bar(x, lang_data["trade_volume"], width, color="plum", alpha=0.8)
            axes[2, 1].set_ylabel("Volume", fontweight="bold")
            axes[2, 1].set_title("Avg Trade Volume by Language")
            axes[2, 1].set_xticks(x)
            axes[2, 1].set_xticklabels(languages_present, rotation=45, ha="right")
            axes[2, 1].grid(axis="y", alpha=0.3)
            for i, v in enumerate(lang_data["trade_volume"]):
                axes[2, 1].text(i, v + 0.2, f"{v:.1f}", ha="center", fontsize=9)

            # Fairness (lower is better)
            axes[3, 0].bar(
                x, lang_data["fairness"], width, color="lightsalmon", alpha=0.8
            )
            axes[3, 0].set_ylabel("Fairness Score", fontweight="bold")
            axes[3, 0].set_title("Fairness by Language (lower=better)")
            axes[3, 0].set_xticks(x)
            axes[3, 0].set_xticklabels(languages_present, rotation=45, ha="right")
            axes[3, 0].grid(axis="y", alpha=0.3)
            for i, v in enumerate(lang_data["fairness"]):
                axes[3, 0].text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=9)

            # Average rounds
            axes[3, 1].bar(
                x, lang_data["rounds"], width, color="mediumpurple", alpha=0.8
            )
            axes[3, 1].set_ylabel("Number of Rounds", fontweight="bold")
            axes[3, 1].set_title("Avg Negotiation Rounds by Language")
            axes[3, 1].set_xticks(x)
            axes[3, 1].set_xticklabels(languages_present, rotation=45, ha="right")
            axes[3, 1].grid(axis="y", alpha=0.3)
            for i, v in enumerate(lang_data["rounds"]):
                axes[3, 1].text(i, v + 0.1, f"{v:.1f}", ha="center", fontsize=9)

            plt.tight_layout()
            safe_combo = combo.replace("/", "_")
            filename = f"language_comparison_{safe_combo}.png"
            plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Created language comparison graph: {filename}")

    def create_cross_model_graphs(self, stats: Dict, output_dir: Path):
        """Create bar graphs combining values across models for same language."""
        for language in self.languages:
            # Collect all model combinations for this language
            combos = []
            data_points = {
                "acceptance": [],
                "both_goals": [],
                "goal_distance": [],
                "balance": [],
                "trade_volume": [],
                "fairness": [],
            }

            for key, stat in stats.items():
                if stat["language"] == language:
                    combo = f"{stat['model1']}_vs_{stat['model2']}"
                    combos.append(combo)
                    data_points["acceptance"].append(stat["acceptance_rate"])
                    data_points["both_goals"].append(stat["both_goals_rate"])
                    data_points["goal_distance"].append(stat["avg_goal_distance"])
                    data_points["balance"].append(stat["avg_balance"])
                    data_points["trade_volume"].append(stat["avg_trade_volume"])
                    data_points["fairness"].append(stat["avg_fairness"])

            if not combos:
                continue

            # Sort by combo name for consistency
            sorted_indices = sorted(range(len(combos)), key=lambda i: combos[i])
            combos = [combos[i] for i in sorted_indices]
            for key in data_points:
                data_points[key] = [data_points[key][i] for i in sorted_indices]

            # Create figure with subplots (3x2 grid)
            fig, axes = plt.subplots(3, 2, figsize=(20, 16))
            fig.suptitle(
                f"{language} - Cross-Model Comparison", fontsize=18, fontweight="bold"
            )

            x = np.arange(len(combos))
            width = 0.6

            # Acceptance rate
            axes[0, 0].bar(
                x, data_points["acceptance"], width, color="lightgreen", alpha=0.8
            )
            axes[0, 0].set_ylabel("Rate", fontweight="bold")
            axes[0, 0].set_title("Acceptance Rate by Model Combination")
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(combos, rotation=90, ha="right", fontsize=7)
            axes[0, 0].set_ylim([0, 1.1])
            axes[0, 0].grid(axis="y", alpha=0.3)

            # Both goals reached
            axes[0, 1].bar(
                x, data_points["both_goals"], width, color="mediumseagreen", alpha=0.8
            )
            axes[0, 1].set_ylabel("Rate", fontweight="bold")
            axes[0, 1].set_title("Both Goals Reached Rate by Model Combination")
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(combos, rotation=90, ha="right", fontsize=7)
            axes[0, 1].set_ylim([0, 1.1])
            axes[0, 1].grid(axis="y", alpha=0.3)

            # Goal distance
            axes[1, 0].bar(
                x, data_points["goal_distance"], width, color="coral", alpha=0.8
            )
            axes[1, 0].set_ylabel("Distance", fontweight="bold")
            axes[1, 0].set_title("Avg Distance from Goal by Model Combination")
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(combos, rotation=90, ha="right", fontsize=7)
            axes[1, 0].grid(axis="y", alpha=0.3)

            # Balance
            axes[1, 1].bar(x, data_points["balance"], width, color="skyblue", alpha=0.8)
            axes[1, 1].set_ylabel("Balance Score", fontweight="bold")
            axes[1, 1].set_title("Avg Resource Balance by Model Combination")
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(combos, rotation=90, ha="right", fontsize=7)
            axes[1, 1].set_ylim([0, 1.1])
            axes[1, 1].grid(axis="y", alpha=0.3)

            # Trade volume
            axes[2, 0].bar(
                x, data_points["trade_volume"], width, color="plum", alpha=0.8
            )
            axes[2, 0].set_ylabel("Volume", fontweight="bold")
            axes[2, 0].set_title("Avg Trade Volume by Model Combination")
            axes[2, 0].set_xticks(x)
            axes[2, 0].set_xticklabels(combos, rotation=90, ha="right", fontsize=7)
            axes[2, 0].grid(axis="y", alpha=0.3)

            # Fairness
            axes[2, 1].bar(
                x, data_points["fairness"], width, color="lightsalmon", alpha=0.8
            )
            axes[2, 1].set_ylabel("Fairness Score", fontweight="bold")
            axes[2, 1].set_title("Fairness by Model Combination")
            axes[2, 1].set_xticks(x)
            axes[2, 1].set_xticklabels(combos, rotation=90, ha="right", fontsize=7)
            axes[2, 1].grid(axis="y", alpha=0.3)

            plt.tight_layout()
            filename = f"cross_model_{language}.png"
            plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Created cross-model graph: {filename}")

    def create_overall_language_comparison(self, stats: Dict, output_dir: Path):
        """Create aggregated comparison across all languages and all model combinations."""

        # Aggregate data by language
        lang_aggregate = {
            lang: {
                "acceptance": [],
                "both_goals": [],
                "at_least_one": [],
                "goal_distance": [],
                "balance": [],
                "trade_volume": [],
                "fairness": [],
                "rounds": [],
            }
            for lang in self.languages
        }

        for key, stat in stats.items():
            lang = stat["language"]
            if lang in lang_aggregate:
                lang_aggregate[lang]["acceptance"].append(stat["acceptance_rate"])
                lang_aggregate[lang]["both_goals"].append(stat["both_goals_rate"])
                lang_aggregate[lang]["at_least_one"].append(
                    stat["at_least_one_goal_rate"]
                )
                lang_aggregate[lang]["goal_distance"].append(stat["avg_goal_distance"])
                lang_aggregate[lang]["balance"].append(stat["avg_balance"])
                lang_aggregate[lang]["trade_volume"].append(stat["avg_trade_volume"])
                lang_aggregate[lang]["fairness"].append(stat["avg_fairness"])
                lang_aggregate[lang]["rounds"].append(stat["avg_rounds"])

        # Calculate means for each language
        lang_means = {lang: {} for lang in self.languages}
        for lang in self.languages:
            for metric in lang_aggregate[lang]:
                values = lang_aggregate[lang][metric]
                lang_means[lang][metric] = np.mean(values) if values else 0

        # Create visualization
        fig, axes = plt.subplots(4, 2, figsize=(16, 20))
        fig.suptitle(
            "Overall Language Performance Comparison (All Model Combinations)",
            fontsize=18,
            fontweight="bold",
        )

        languages = [lang for lang in self.languages if any(lang_means[lang].values())]
        x = np.arange(len(languages))
        width = 0.6

        metrics_to_plot = [
            ("acceptance", "Overall Acceptance Rate", "lightgreen", [0, 1.1]),
            (
                "both_goals",
                "Overall Both Goals Reached Rate",
                "mediumseagreen",
                [0, 1.1],
            ),
            (
                "at_least_one",
                "Overall At Least One Goal Rate",
                "darkseagreen",
                [0, 1.1],
            ),
            ("goal_distance", "Overall Avg Distance from Goal", "coral", None),
            ("balance", "Overall Avg Resource Balance", "skyblue", [0, 1.1]),
            ("trade_volume", "Overall Avg Trade Volume", "plum", None),
            ("fairness", "Overall Fairness", "lightsalmon", None),
            ("rounds", "Overall Avg Negotiation Rounds", "mediumpurple", None),
        ]

        for idx, (metric, title, color, ylim) in enumerate(metrics_to_plot):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]

            values = [lang_means[lang][metric] for lang in languages]

            ax.bar(x, values, width, color=color, alpha=0.8)
            ax.set_ylabel("Value", fontweight="bold")
            ax.set_title(title)
            ax.set_xticks(x)
            ax.set_xticklabels(languages, rotation=45, ha="right")
            if ylim:
                ax.set_ylim(ylim)
            ax.grid(axis="y", alpha=0.3)

            # Add value labels on bars
            for i, v in enumerate(values):
                if "rate" in metric or "balance" in metric:
                    label = f"{v:.2%}" if v < 10 else f"{v:.3f}"
                else:
                    label = f"{v:.2f}"
                ax.text(
                    i, v + (v * 0.02 if v > 0 else 0.02), label, ha="center", fontsize=9
                )

        plt.tight_layout()
        filename = "overall_language_comparison.png"
        plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Created overall language comparison graph: {filename}")

    def save_summary_json(self, stats: Dict, output_dir: Path):
        """Save comprehensive summary to JSON file."""
        # Convert numpy types to native Python types for JSON serialization
        summary = {}
        for key, stat in stats.items():
            summary[key] = {
                "model1": stat["model1"],
                "model2": stat["model2"],
                "language": stat["language"],
                "num_games": int(stat["num_games"]),
                "acceptance_rate": float(stat["acceptance_rate"]),
                "both_goals_reached_rate": float(stat["both_goals_rate"]),
                "at_least_one_goal_rate": float(stat["at_least_one_goal_rate"]),
                "goal_distance_statistics": {
                    "average": float(stat["avg_goal_distance"]),
                    "std_dev": float(stat["std_goal_distance"]),
                    "min": float(stat["min_goal_distance"]),
                    "max": float(stat["max_goal_distance"]),
                    "all_values": [float(x) for x in stat["values"]["goal_distances"]],
                },
                "balance_statistics": {
                    "average": float(stat["avg_balance"]),
                    "std_dev": float(stat["std_balance"]),
                    "all_values": [float(x) for x in stat["values"]["balances"]],
                },
                "trade_volume_statistics": {
                    "average": float(stat["avg_trade_volume"]),
                    "std_dev": float(stat["std_trade_volume"]),
                    "all_values": [float(x) for x in stat["values"]["trade_volumes"]],
                },
                "fairness_statistics": {
                    "average": float(stat["avg_fairness"]),
                    "std_dev": float(stat["std_fairness"]),
                    "all_values": [float(x) for x in stat["values"]["fairness"]],
                },
                "rounds_statistics": {
                    "average": float(stat["avg_rounds"]),
                    "std_dev": float(stat["std_rounds"]),
                    "all_values": [float(x) for x in stat["values"]["rounds"]],
                },
            }

        # Add overall statistics
        all_acceptance = []
        all_both_goals = []
        all_goal_distances = []
        all_balances = []
        all_trade_volumes = []
        all_fairness = []
        all_rounds = []

        for stat in stats.values():
            all_acceptance.append(stat["acceptance_rate"])
            all_both_goals.append(stat["both_goals_rate"])
            all_goal_distances.extend(stat["values"]["goal_distances"])
            all_balances.extend(stat["values"]["balances"])
            all_trade_volumes.extend(stat["values"]["trade_volumes"])
            all_fairness.extend(stat["values"]["fairness"])
            all_rounds.extend(stat["values"]["rounds"])

        summary["overall_statistics"] = {
            "total_games": len(self.all_results),
            "total_combinations": len(stats),
            "acceptance_rate": {
                "mean": float(np.mean(all_acceptance)) if all_acceptance else 0,
                "std": float(np.std(all_acceptance)) if all_acceptance else 0,
            },
            "both_goals_rate": {
                "mean": float(np.mean(all_both_goals)) if all_both_goals else 0,
                "std": float(np.std(all_both_goals)) if all_both_goals else 0,
            },
            "goal_distance": {
                "mean": float(np.mean(all_goal_distances)) if all_goal_distances else 0,
                "std": float(np.std(all_goal_distances)) if all_goal_distances else 0,
            },
            "balance": {
                "mean": float(np.mean(all_balances)) if all_balances else 0,
                "std": float(np.std(all_balances)) if all_balances else 0,
            },
            "trade_volume": {
                "mean": float(np.mean(all_trade_volumes)) if all_trade_volumes else 0,
                "std": float(np.std(all_trade_volumes)) if all_trade_volumes else 0,
            },
            "fairness": {
                "mean": float(np.mean(all_fairness)) if all_fairness else 0,
                "std": float(np.std(all_fairness)) if all_fairness else 0,
            },
            "rounds": {
                "mean": float(np.mean(all_rounds)) if all_rounds else 0,
                "std": float(np.std(all_rounds)) if all_rounds else 0,
            },
        }

        # Add language-specific aggregates
        summary["language_aggregates"] = {}
        for language in self.languages:
            lang_stats = [s for k, s in stats.items() if s["language"] == language]
            if lang_stats:
                summary["language_aggregates"][language] = {
                    "num_combinations": len(lang_stats),
                    "avg_acceptance_rate": float(
                        np.mean([s["acceptance_rate"] for s in lang_stats])
                    ),
                    "avg_both_goals_rate": float(
                        np.mean([s["both_goals_rate"] for s in lang_stats])
                    ),
                    "avg_goal_distance": float(
                        np.mean([s["avg_goal_distance"] for s in lang_stats])
                    ),
                    "avg_balance": float(
                        np.mean([s["avg_balance"] for s in lang_stats])
                    ),
                    "avg_trade_volume": float(
                        np.mean([s["avg_trade_volume"] for s in lang_stats])
                    ),
                    "avg_fairness": float(
                        np.mean([s["avg_fairness"] for s in lang_stats])
                    ),
                    "avg_rounds": float(np.mean([s["avg_rounds"] for s in lang_stats])),
                }

        output_file = output_dir / "summary.json"
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Saved summary to: {output_file}")

    def run_analysis(self):
        """Run complete analysis pipeline."""
        print("=" * 80)
        print("Trading Game Negotiation Arena - Comprehensive Analysis")
        print("=" * 80)

        # Create output directory
        output_dir = self.results_dir / "analysis_output"
        output_dir.mkdir(exist_ok=True)
        print(f"Output directory: {output_dir}")

        # Load all results
        self.load_all_results()

        if not self.all_results:
            print("ERROR: No results loaded!")
            return

        # Compute statistics
        print("\nComputing statistics...")
        stats = self.compute_statistics()

        # Print some sample statistics
        print("\nSample statistics:")
        for key in list(stats.keys())[:3]:
            stat = stats[key]
            print(f"\n{key}:")
            print(f"  Acceptance rate: {stat['acceptance_rate']:.2%}")
            print(f"  Both goals reached: {stat['both_goals_rate']:.2%}")
            print(f"  Avg goal distance: {stat['avg_goal_distance']:.3f}")
            print(f"  Avg balance: {stat['avg_balance']:.3f}")
            print(f"  Avg trade volume: {stat['avg_trade_volume']:.2f}")
            print(f"  Avg fairness: {stat['avg_fairness']:.3f}")

        # Save summary JSON
        print("\nSaving summary JSON...")
        self.save_summary_json(stats, output_dir)

        # Create consolidated heatmaps
        print("\nCreating consolidated heatmaps...")
        self.create_all_heatmaps(stats, output_dir)

        # Create language comparison graphs
        print("\nCreating language comparison graphs...")
        self.create_language_comparison_graphs(stats, output_dir)

        # Create cross-model graphs
        print("\nCreating cross-model graphs...")
        self.create_cross_model_graphs(stats, output_dir)

        # Create overall language comparison
        print("\nCreating overall language comparison...")
        self.create_overall_language_comparison(stats, output_dir)

        print("\n" + "=" * 80)
        print("Analysis complete!")
        print(f"Results saved to: {output_dir}")
        print("=" * 80)


def main():
    # Set the results directory
    results_dir = ".logs/final_final_trading"

    # Create analyzer and run
    analyzer = TradingGameAnalyzer(results_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
