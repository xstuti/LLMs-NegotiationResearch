#!/usr/bin/env python3
"""
Comprehensive analysis script for BuySell Negotiation Arena results.
Generates heatmaps, graphs, and statistical summaries.
"""

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 10)


class BuySellAnalyzer:
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.data = defaultdict(lambda: defaultdict(list))
        self.models = ["Claude-3.5-Haiku", "Claude-3-Haiku", "GPT-4o", "GPT-3.5"]
        self.languages = ["Hindi", "Gujarati", "Marwadi", "Punjabi"]
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
            player_outcome = summary.get("player_outcome", [0, 0])
            proposed_trade = summary.get("proposed_trade", {})

            # Extract trade price if accepted
            trade_price = None
            if final_response == "ACCEPT" and proposed_trade:
                # Navigate through the nested structure
                trade_value = proposed_trade.get("_value", {})
                blue_resource = trade_value.get("BLUE", {})
                blue_value = blue_resource.get("_value", {})
                trade_price = blue_value.get("ZUP", None)

            seller_adv = player_outcome[0] if len(player_outcome) > 0 else 0
            buyer_adv = player_outcome[1] if len(player_outcome) > 1 else 0

            return {
                "final_response": final_response,
                "seller_advantage": seller_adv,
                "buyer_advantage": buyer_adv,
                "trade_price": trade_price,
                "accepted": final_response == "ACCEPT",
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

            # Parse directory name: Model1_vs_Model2_Language_iter_N
            parts = subdir.name.split("_")
            if len(parts) < 5 or "vs" not in parts:
                continue

            try:
                vs_idx = parts.index("vs")
                iter_idx = parts.index("iter")

                # Extract model names and language
                model1_parts = parts[:vs_idx]
                model2_parts = parts[vs_idx + 1 : iter_idx - 1]
                language = parts[iter_idx - 1]
                iteration = int(parts[iter_idx + 1])

                model1 = "_".join(model1_parts)
                model2 = "_".join(model2_parts)

                # Normalize model names
                model1 = model1.replace("_", "-")
                model2 = model2.replace("_", "-")

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
                        self.data[key]["seller_advantages"].append(
                            game_data["seller_advantage"]
                        )
                        self.data[key]["buyer_advantages"].append(
                            game_data["buyer_advantage"]
                        )
                        self.data[key]["trade_prices"].append(game_data["trade_price"])
                        self.data[key]["accepted"].append(game_data["accepted"])
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
            seller_advs = [x for x in data["seller_advantages"] if x is not None]
            buyer_advs = [x for x in data["buyer_advantages"] if x is not None]
            trade_prices = [x for x in data["trade_prices"] if x is not None]
            accepted = data["accepted"]

            stats[key] = {
                "model1": data["model1"],
                "model2": data["model2"],
                "language": data["language"],
                "num_games": len(accepted),
                "acceptance_rate": sum(accepted) / len(accepted) if accepted else 0,
                # Seller statistics
                "seller_avg": np.mean(seller_advs) if seller_advs else 0,
                "seller_std": np.std(seller_advs) if seller_advs else 0,
                "seller_min": np.min(seller_advs) if seller_advs else 0,
                "seller_max": np.max(seller_advs) if seller_advs else 0,
                "seller_values": seller_advs,
                # Buyer statistics
                "buyer_avg": np.mean(buyer_advs) if buyer_advs else 0,
                "buyer_std": np.std(buyer_advs) if buyer_advs else 0,
                "buyer_min": np.min(buyer_advs) if buyer_advs else 0,
                "buyer_max": np.max(buyer_advs) if buyer_advs else 0,
                "buyer_values": buyer_advs,
                # Trade price statistics
                "price_avg": np.mean(trade_prices) if trade_prices else 0,
                "price_std": np.std(trade_prices) if trade_prices else 0,
                "price_min": np.min(trade_prices) if trade_prices else 0,
                "price_max": np.max(trade_prices) if trade_prices else 0,
                "price_values": trade_prices,
            }

        return stats

    def create_consolidated_heatmaps(
        self, language: str, stats: Dict, output_dir: Path
    ):
        """Create a single image with 4 heatmaps (2x2) for a language."""
        metrics = [
            ("seller_avg", "Average Seller Advantage"),
            ("buyer_avg", "Average Buyer Advantage"),
            ("acceptance_rate", "Acceptance Rate"),
            ("price_avg", "Average Trade Price (ZUP)"),
        ]

        fig, axes = plt.subplots(2, 2, figsize=(20, 18))
        fig.suptitle(
            f"{language} - Model Comparison Heatmaps",
            fontsize=18,
            fontweight="bold",
            y=0.995,
        )

        for idx, (metric, title) in enumerate(metrics):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]

            # Create matrix for the heatmap
            matrix = np.zeros((len(self.models), len(self.models)))

            for i, seller_model in enumerate(self.models):
                for j, buyer_model in enumerate(self.models):
                    key = f"{seller_model}_vs_{buyer_model}_{language}"
                    if key in stats:
                        matrix[i, j] = stats[key][metric]
                    else:
                        matrix[i, j] = np.nan

            # Create mask for NaN values
            mask = np.isnan(matrix)

            # Determine colormap based on metric
            if metric == "acceptance_rate":
                cmap = "YlGn"
                center = None
            elif "price" in metric:
                cmap = "YlOrRd"
                center = None
            else:
                cmap = "RdYlGn"
                center = 0

            sns.heatmap(
                matrix,
                annot=True,
                fmt=".2f",
                cmap=cmap,
                xticklabels=self.models,
                yticklabels=self.models,
                center=center,
                mask=mask,
                cbar_kws={"label": title},
                ax=ax,
                vmin=None if center is None else None,
                vmax=None,
            )

            ax.set_xlabel("Buyer Model (Model 2)", fontsize=11, fontweight="bold")
            ax.set_ylabel("Seller Model (Model 1)", fontsize=11, fontweight="bold")
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

    def create_cross_language_graphs(self, stats: Dict, output_dir: Path):
        """Create bar graphs combining values across languages for same model combo."""
        # Group by model combination
        model_combos = set()
        for key in stats.keys():
            parts = key.rsplit("_", 1)
            if len(parts) == 2:
                model_combos.add(parts[0])

        for combo in sorted(model_combos):
            # Collect data for this combo across all languages
            lang_seller_avgs = []
            lang_buyer_avgs = []
            lang_acceptance = []
            lang_price_avgs = []
            languages_present = []

            for lang in self.languages:
                key = f"{combo}_{lang}"
                if key in stats:
                    languages_present.append(lang)
                    lang_seller_avgs.append(stats[key]["seller_avg"])
                    lang_buyer_avgs.append(stats[key]["buyer_avg"])
                    lang_acceptance.append(stats[key]["acceptance_rate"])
                    lang_price_avgs.append(stats[key]["price_avg"])

            if not languages_present:
                continue

            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(
                f"{combo} - Cross-Language Comparison", fontsize=16, fontweight="bold"
            )

            x = np.arange(len(languages_present))
            width = 0.6

            # Seller advantage
            axes[0, 0].bar(x, lang_seller_avgs, width, color="coral", alpha=0.8)
            axes[0, 0].set_ylabel("Average Advantage", fontweight="bold")
            axes[0, 0].set_title("Seller Advantage by Language")
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(languages_present, rotation=45, ha="right")
            axes[0, 0].axhline(y=0, color="black", linestyle="--", linewidth=0.8)
            axes[0, 0].grid(axis="y", alpha=0.3)

            # Buyer advantage
            axes[0, 1].bar(x, lang_buyer_avgs, width, color="skyblue", alpha=0.8)
            axes[0, 1].set_ylabel("Average Advantage", fontweight="bold")
            axes[0, 1].set_title("Buyer Advantage by Language")
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(languages_present, rotation=45, ha="right")
            axes[0, 1].axhline(y=0, color="black", linestyle="--", linewidth=0.8)
            axes[0, 1].grid(axis="y", alpha=0.3)

            # Acceptance rate
            axes[1, 0].bar(x, lang_acceptance, width, color="lightgreen", alpha=0.8)
            axes[1, 0].set_ylabel("Acceptance Rate", fontweight="bold")
            axes[1, 0].set_title("Acceptance Rate by Language")
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(languages_present, rotation=45, ha="right")
            axes[1, 0].set_ylim([0, 1.1])
            axes[1, 0].grid(axis="y", alpha=0.3)

            # Average trade price
            axes[1, 1].bar(x, lang_price_avgs, width, color="plum", alpha=0.8)
            axes[1, 1].set_ylabel("Average Price (ZUP)", fontweight="bold")
            axes[1, 1].set_title("Average Trade Price by Language")
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(languages_present, rotation=45, ha="right")
            axes[1, 1].grid(axis="y", alpha=0.3)

            plt.tight_layout()
            safe_combo = combo.replace("/", "_")
            filename = f"cross_language_{safe_combo}.png"
            plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Created cross-language graph: {filename}")

    def create_cross_model_graphs(self, stats: Dict, output_dir: Path):
        """Create bar graphs combining values across models for same language."""
        for language in self.languages:
            # Collect all model combinations for this language
            combos = []
            seller_avgs = []
            buyer_avgs = []
            acceptance_rates = []
            price_avgs = []

            for key, stat in stats.items():
                if stat["language"] == language:
                    combo = f"{stat['model1']}_vs_{stat['model2']}"
                    combos.append(combo)
                    seller_avgs.append(stat["seller_avg"])
                    buyer_avgs.append(stat["buyer_avg"])
                    acceptance_rates.append(stat["acceptance_rate"])
                    price_avgs.append(stat["price_avg"])

            if not combos:
                continue

            # Sort by combo name for consistency
            sorted_data = sorted(
                zip(combos, seller_avgs, buyer_avgs, acceptance_rates, price_avgs)
            )
            combos, seller_avgs, buyer_avgs, acceptance_rates, price_avgs = zip(
                *sorted_data
            )

            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(20, 12))
            fig.suptitle(
                f"{language} - Cross-Model Comparison", fontsize=16, fontweight="bold"
            )

            x = np.arange(len(combos))
            width = 0.6

            # Seller advantage
            axes[0, 0].bar(x, seller_avgs, width, color="coral", alpha=0.8)
            axes[0, 0].set_ylabel("Average Advantage", fontweight="bold")
            axes[0, 0].set_title("Seller Advantage by Model Combination")
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(combos, rotation=90, ha="right", fontsize=8)
            axes[0, 0].axhline(y=0, color="black", linestyle="--", linewidth=0.8)
            axes[0, 0].grid(axis="y", alpha=0.3)

            # Buyer advantage
            axes[0, 1].bar(x, buyer_avgs, width, color="skyblue", alpha=0.8)
            axes[0, 1].set_ylabel("Average Advantage", fontweight="bold")
            axes[0, 1].set_title("Buyer Advantage by Model Combination")
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(combos, rotation=90, ha="right", fontsize=8)
            axes[0, 1].axhline(y=0, color="black", linestyle="--", linewidth=0.8)
            axes[0, 1].grid(axis="y", alpha=0.3)

            # Acceptance rate
            axes[1, 0].bar(x, acceptance_rates, width, color="lightgreen", alpha=0.8)
            axes[1, 0].set_ylabel("Acceptance Rate", fontweight="bold")
            axes[1, 0].set_title("Acceptance Rate by Model Combination")
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(combos, rotation=90, ha="right", fontsize=8)
            axes[1, 0].set_ylim([0, 1.1])
            axes[1, 0].grid(axis="y", alpha=0.3)

            # Average trade price
            axes[1, 1].bar(x, price_avgs, width, color="plum", alpha=0.8)
            axes[1, 1].set_ylabel("Average Price (ZUP)", fontweight="bold")
            axes[1, 1].set_title("Average Trade Price by Model Combination")
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(combos, rotation=90, ha="right", fontsize=8)
            axes[1, 1].grid(axis="y", alpha=0.3)

            plt.tight_layout()
            filename = f"cross_model_{language}.png"
            plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Created cross-model graph: {filename}")

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
                "seller_statistics": {
                    "average": float(stat["seller_avg"]),
                    "std_dev": float(stat["seller_std"]),
                    "min": float(stat["seller_min"]),
                    "max": float(stat["seller_max"]),
                    "all_values": [float(x) for x in stat["seller_values"]],
                },
                "buyer_statistics": {
                    "average": float(stat["buyer_avg"]),
                    "std_dev": float(stat["buyer_std"]),
                    "min": float(stat["buyer_min"]),
                    "max": float(stat["buyer_max"]),
                    "all_values": [float(x) for x in stat["buyer_values"]],
                },
                "trade_price_statistics": {
                    "average": float(stat["price_avg"]),
                    "std_dev": float(stat["price_std"]),
                    "min": float(stat["price_min"]) if stat["price_values"] else None,
                    "max": float(stat["price_max"]) if stat["price_values"] else None,
                    "all_values": [float(x) for x in stat["price_values"]],
                },
            }

        # Add overall statistics
        all_seller_advs = []
        all_buyer_advs = []
        all_prices = []
        all_acceptances = []

        for stat in stats.values():
            all_seller_advs.extend(stat["seller_values"])
            all_buyer_advs.extend(stat["buyer_values"])
            all_prices.extend(stat["price_values"])
            all_acceptances.append(stat["acceptance_rate"])

        summary["overall_statistics"] = {
            "total_games": len(self.all_results),
            "total_combinations": len(stats),
            "seller_advantage": {
                "mean": float(np.mean(all_seller_advs)) if all_seller_advs else 0,
                "std": float(np.std(all_seller_advs)) if all_seller_advs else 0,
            },
            "buyer_advantage": {
                "mean": float(np.mean(all_buyer_advs)) if all_buyer_advs else 0,
                "std": float(np.std(all_buyer_advs)) if all_buyer_advs else 0,
            },
            "trade_price": {
                "mean": float(np.mean(all_prices)) if all_prices else 0,
                "std": float(np.std(all_prices)) if all_prices else 0,
            },
            "average_acceptance_rate": float(np.mean(all_acceptances))
            if all_acceptances
            else 0,
        }

        output_file = output_dir / "summary.json"
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Saved summary to: {output_file}")

    def run_analysis(self):
        """Run complete analysis pipeline."""
        print("=" * 80)
        print("BuySell Negotiation Arena - Comprehensive Analysis")
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
            print(
                f"  Seller avg: {stat['seller_avg']:.2f} (std: {stat['seller_std']:.2f})"
            )
            print(
                f"  Buyer avg: {stat['buyer_avg']:.2f} (std: {stat['buyer_std']:.2f})"
            )
            print(f"  Acceptance rate: {stat['acceptance_rate']:.2%}")
            print(f"  Price avg: {stat['price_avg']:.2f}")

        # Save summary JSON
        print("\nSaving summary JSON...")
        self.save_summary_json(stats, output_dir)

        # Create consolidated heatmaps
        print("\nCreating consolidated heatmaps...")
        self.create_all_heatmaps(stats, output_dir)

        # Create cross-language graphs
        print("\nCreating cross-language graphs...")
        self.create_cross_language_graphs(stats, output_dir)

        # Create cross-model graphs
        print("\nCreating cross-model graphs...")
        self.create_cross_model_graphs(stats, output_dir)

        print("\n" + "=" * 80)
        print("Analysis complete!")
        print(f"Results saved to: {output_dir}")
        print("=" * 80)


def main():
    # Set the results directory
    results_dir = ".logs/buysell_final"

    # Create analyzer and run
    analyzer = BuySellAnalyzer(results_dir)
    analyzer.run_analysis()


if __name__ == "__main__":
    main()
