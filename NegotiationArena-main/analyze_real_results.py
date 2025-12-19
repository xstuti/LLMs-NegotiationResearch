#!/usr/bin/env python3
"""
Analyze Real Ultimatum Game Results

This script analyzes the actual game results from the log files,
extracting key metrics from game_state.json files.
"""

import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

# Add current directory to Python path for module imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


class UltimatumResultsAnalyzer:
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
        self.summary = {}

    def find_game_directories(self):
        """Find all game directories with the expected naming pattern"""
        game_dirs = []

        for item in self.results_dir.iterdir():
            if item.is_dir() and "_vs_" in item.name and "_iter_" in item.name:
                game_dirs.append(item)

        print(f"Found {len(game_dirs)} game directories")
        return sorted(game_dirs)

    def parse_directory_name(self, dir_name):
        """Parse directory name to extract model1, model2, behavior, and iteration

        Handles behavior names with underscores like "Marwadi_Forced"
        Format: Model1_vs_Model2_Behavior_iter_N
        """
        print(f"Parsing: {dir_name}")

        try:
            # Split by underscores
            parts = dir_name.split("_")
            print(f"  Parts: {parts}")

            # Find key indices
            vs_index = parts.index("vs")
            iter_index = parts.index("iter")

            # Model 1: everything before 'vs', rejoin with underscores
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

            # Iteration number
            iteration = int(parts[-1])

            print(f"  Parsed: {model1} vs {model2} | {behavior} | iter {iteration}")

            return model1, model2, behavior, iteration

        except Exception as e:
            print(f"  Error parsing {dir_name}: {e}")
            # Fallback parsing
            if "_vs_" in dir_name and "_iter_" in dir_name:
                # Remove iteration part
                base = dir_name.rsplit("_iter_", 1)[0]
                iteration = int(dir_name.rsplit("_iter_", 1)[1])

                # Split by _vs_
                parts = base.split("_vs_")
                if len(parts) == 2:
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
                                    "_".join(behavior_parts)
                                    if behavior_parts
                                    else "Unknown"
                                )
                                break

                    if model2 is None or behavior is None:
                        # Last resort: check if rest_str ends with a known behavior
                        for known_behavior in self.KNOWN_BEHAVIORS:
                            if rest_str.endswith(known_behavior):
                                idx = rest_str.rfind(known_behavior)
                                model2 = rest_str[:idx].rstrip("_")
                                behavior = known_behavior
                                break

                    if model2 is None or behavior is None:
                        model2 = rest_parts[0]
                        behavior = (
                            "_".join(rest_parts[1:])
                            if len(rest_parts) > 1
                            else "Unknown"
                        )

                    print(
                        f"  Fallback parsed: {model1} vs {model2} | {behavior} | iter {iteration}"
                    )
                    return model1, model2, behavior, iteration

            # If all fails, return defaults
            return "Unknown", "Unknown", "Unknown", 1

    def find_game_state_file(self, game_dir):
        """Find the game_state.json file in the game directory"""
        # Look for subdirectories (timestamp directories)
        for subdir in game_dir.iterdir():
            if subdir.is_dir():
                game_state_file = subdir / "game_state.json"
                if game_state_file.exists():
                    return game_state_file
        return None

    def extract_game_data(self, game_state_file, model1, model2, behavior, iteration):
        """Extract relevant data from a single game_state.json file"""
        try:
            with open(game_state_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Find the END iteration in game_state
            game_state = data.get("game_state", [])
            end_state = None

            for state in game_state:
                if state.get("current_iteration") == "END":
                    end_state = state
                    break

            if not end_state:
                print(f"Warning: No END state found in {game_state_file}")
                return None

            summary = end_state.get("summary", {})

            # Extract proposed trade (initial offer)
            proposed_trade = summary.get("proposed_trade", {})
            if proposed_trade.get("_type") == "trade":
                trade_value = proposed_trade.get("_value", {})
                red_offer = (
                    trade_value.get("RED", {}).get("_value", {}).get("Dollars", 0)
                )
                blue_offer = (
                    trade_value.get("BLUE", {}).get("_value", {}).get("Dollars", 0)
                )

                # In ultimatum game, the offer is what Player 1 (RED) gives to Player 2 (BLUE)
                initial_offer = red_offer  # This is what RED offers to give to BLUE
            else:
                initial_offer = 0

            # Extract final response
            final_response = summary.get("final_response", "UNKNOWN")

            # Extract final resources
            final_resources = summary.get("final_resources", [])
            player1_final = 0
            player2_final = 0

            if len(final_resources) >= 2:
                player1_final = final_resources[0].get("_value", {}).get("Dollars", 0)
                player2_final = final_resources[1].get("_value", {}).get("Dollars", 0)

            # Determine outcome
            if final_response == "REJECT":
                player1_final = 0
                player2_final = 0
                outcome = "REJECT"
            elif final_response == "ACCEPT":
                outcome = "ACCEPT"
            else:
                outcome = "UNKNOWN"

            game_data = {
                "model1": model1,
                "model2": model2,
                "behavior": behavior,
                "iteration": iteration,
                "initial_offer": initial_offer,
                "final_response": final_response,
                "outcome": outcome,
                "player1_final": player1_final,
                "player2_final": player2_final,
                "file_path": str(game_state_file),
            }

            return game_data

        except Exception as e:
            print(f"Error processing {game_state_file}: {str(e)}")
            return None

    def analyze_all_games(self):
        """Analyze all games and extract data"""
        game_dirs = self.find_game_directories()

        for game_dir in game_dirs:
            try:
                model1, model2, behavior, iteration = self.parse_directory_name(
                    game_dir.name
                )
                game_state_file = self.find_game_state_file(game_dir)

                if game_state_file:
                    game_data = self.extract_game_data(
                        game_state_file, model1, model2, behavior, iteration
                    )
                    if game_data:
                        self.raw_data.append(game_data)
                        print(f"✓ Processed {game_dir.name}")
                    else:
                        print(f"✗ Failed to extract data from {game_dir.name}")
                else:
                    print(f"✗ No game_state.json found in {game_dir.name}")

            except Exception as e:
                print(f"✗ Error processing {game_dir.name}: {str(e)}")

        print(f"\nTotal games processed: {len(self.raw_data)}")
        return self.raw_data

    def calculate_metrics(self):
        """Calculate summary metrics for each model combination and behavior"""
        # Group data by model combination and behavior
        groups = defaultdict(list)

        for game in self.raw_data:
            key = (game["model1"], game["model2"], game["behavior"])
            groups[key].append(game)

        summary = {}

        for (model1, model2, behavior), games in groups.items():
            combo_key = f"{model1}_vs_{model2}_{behavior}"

            # Filter valid games (exclude unknowns)
            valid_games = [g for g in games if g["outcome"] in ["ACCEPT", "REJECT"]]

            if not valid_games:
                print(f"Warning: No valid games found for {combo_key}")
                continue

            # Calculate metrics
            total_games = len(valid_games)
            accepts = [g for g in valid_games if g["outcome"] == "ACCEPT"]
            rejects = [g for g in valid_games if g["outcome"] == "REJECT"]

            # Win rate calculation (Player 1 wins if they get more than Player 2, excluding draws)
            player1_wins = 0
            player2_wins = 0
            draws = 0

            for game in valid_games:
                if game["player1_final"] > game["player2_final"]:
                    player1_wins += 1
                elif game["player2_final"] > game["player1_final"]:
                    player2_wins += 1
                else:
                    draws += 1

            # Win rate excluding draws
            non_draw_games = player1_wins + player2_wins
            win_rate = player1_wins / non_draw_games if non_draw_games > 0 else 0.0
            draw_rate = draws / total_games if total_games > 0 else 0.0

            # Payoff statistics
            player1_payoffs = [g["player1_final"] for g in valid_games]
            player2_payoffs = [g["player2_final"] for g in valid_games]
            initial_offers = [g["initial_offer"] for g in valid_games]

            # Summary statistics
            metrics = {
                "total_games": total_games,
                "accepts": len(accepts),
                "rejects": len(rejects),
                "acceptance_rate": len(accepts) / total_games
                if total_games > 0
                else 0.0,
                # Win rates
                "player1_wins": player1_wins,
                "player2_wins": player2_wins,
                "draws": draws,
                "win_rate_player1": win_rate,
                "win_rate_player2": 1 - win_rate if non_draw_games > 0 else 0.0,
                "draw_rate": draw_rate,
                # Payoffs
                "player1_payoff_avg": statistics.mean(player1_payoffs)
                if player1_payoffs
                else 0,
                "player1_payoff_std": statistics.stdev(player1_payoffs)
                if len(player1_payoffs) > 1
                else 0,
                "player1_payoffs": player1_payoffs,
                "player2_payoff_avg": statistics.mean(player2_payoffs)
                if player2_payoffs
                else 0,
                "player2_payoff_std": statistics.stdev(player2_payoffs)
                if len(player2_payoffs) > 1
                else 0,
                "player2_payoffs": player2_payoffs,
                # Initial offers
                "initial_offer_avg": statistics.mean(initial_offers)
                if initial_offers
                else 0,
                "initial_offer_std": statistics.stdev(initial_offers)
                if len(initial_offers) > 1
                else 0,
                "initial_offers": initial_offers,
                # Model info
                "model1": model1,
                "model2": model2,
                "behavior": behavior,
            }

            summary[combo_key] = metrics

            # Print summary for this combination
            print(f"\n{combo_key}:")
            print(
                f"  Games: {total_games} | Accepts: {len(accepts)} | Rejects: {len(rejects)}"
            )
            print(f"  Win rate (P1): {win_rate:.3f} | Draw rate: {draw_rate:.3f}")
            print(
                f"  Avg payoffs - P1: {metrics['player1_payoff_avg']:.1f}, P2: {metrics['player2_payoff_avg']:.1f}"
            )
            print(f"  Avg initial offer: {metrics['initial_offer_avg']:.1f}")

        self.summary = summary
        return summary

    def save_results(self):
        """Save results to JSON files"""
        # Save raw data
        raw_data_file = self.results_dir / "raw_game_data.json"
        with open(raw_data_file, "w", encoding="utf-8") as f:
            json.dump(self.raw_data, f, indent=2, ensure_ascii=False)

        # Save summary
        summary_file = self.results_dir / "summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2, ensure_ascii=False)

        # Create readable summary
        readable_file = self.results_dir / "readable_summary.txt"
        with open(readable_file, "w", encoding="utf-8") as f:
            f.write("ULTIMATUM GAME RESULTS SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            for combo_key, metrics in self.summary.items():
                f.write(f"{combo_key}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Games: {metrics['total_games']}\n")
                f.write(f"Acceptance Rate: {metrics['acceptance_rate']:.3f}\n")
                f.write(f"Player 1 Win Rate: {metrics['win_rate_player1']:.3f}\n")
                f.write(f"Player 2 Win Rate: {metrics['win_rate_player2']:.3f}\n")
                f.write(f"Draw Rate: {metrics['draw_rate']:.3f}\n")
                f.write(
                    f"Average Initial Offer: {metrics['initial_offer_avg']:.1f} ± {metrics['initial_offer_std']:.1f}\n"
                )
                f.write(
                    f"Player 1 Average Payoff: {metrics['player1_payoff_avg']:.1f} ± {metrics['player1_payoff_std']:.1f}\n"
                )
                f.write(
                    f"Player 2 Average Payoff: {metrics['player2_payoff_avg']:.1f} ± {metrics['player2_payoff_std']:.1f}\n"
                )
                f.write("\n")

        print(f"\nResults saved:")
        print(f"  Raw data: {raw_data_file}")
        print(f"  Summary: {summary_file}")
        print(f"  Readable: {readable_file}")

    def create_heatmap_data(self):
        """Create data for heatmap visualization"""
        # Get unique models and behaviors
        models = set()
        behaviors = set()

        for combo_key, metrics in self.summary.items():
            models.add(metrics["model1"])
            models.add(metrics["model2"])
            behaviors.add(metrics["behavior"])

        models = sorted(list(models))
        behaviors = sorted(list(behaviors))

        # Create matrices for each behavior
        heatmap_data = {}

        for behavior in behaviors:
            win_rates = {}
            payoffs = {}
            initial_offers = {}

            for model1 in models:
                win_rates[model1] = {}
                payoffs[model1] = {}
                initial_offers[model1] = {}

                for model2 in models:
                    combo_key = f"{model1}_vs_{model2}_{behavior}"

                    if combo_key in self.summary:
                        metrics = self.summary[combo_key]
                        win_rates[model1][model2] = metrics["win_rate_player1"]
                        payoffs[model1][model2] = metrics["player1_payoff_avg"]
                        initial_offers[model1][model2] = metrics["initial_offer_avg"]
                    else:
                        win_rates[model1][model2] = None
                        payoffs[model1][model2] = None
                        initial_offers[model1][model2] = None

            heatmap_data[behavior] = {
                "win_rates": win_rates,
                "payoffs": payoffs,
                "initial_offers": initial_offers,
                "models": models,
            }

        # Save heatmap data
        heatmap_file = self.results_dir / "heatmap_data.json"
        with open(heatmap_file, "w", encoding="utf-8") as f:
            json.dump(heatmap_data, f, indent=2)

        print(f"  Heatmap data: {heatmap_file}")

        return heatmap_data


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python analyze_real_results.py <results_directory>")
        print(
            "Example: python analyze_real_results.py .logs/ultimatum_social_behavior_20251130_180851"
        )
        return

    results_dir = sys.argv[1]

    if not os.path.exists(results_dir):
        print(f"Error: Results directory does not exist: {results_dir}")
        return

    print(f"Analyzing results in: {results_dir}")
    print("=" * 60)

    analyzer = UltimatumResultsAnalyzer(results_dir)

    # Analyze all games
    raw_data = analyzer.analyze_all_games()

    if not raw_data:
        print("No valid game data found!")
        return

    # Calculate metrics
    summary = analyzer.calculate_metrics()

    # Save results
    analyzer.save_results()

    # Create heatmap data
    analyzer.create_heatmap_data()

    print(f"\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"Processed {len(raw_data)} games")
    print(f"Generated {len(summary)} combination summaries")
    print(f"Check the generated files in: {results_dir}")


if __name__ == "__main__":
    main()
