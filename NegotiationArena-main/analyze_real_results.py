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


class TradingResultsAnalyzer:
    # Known behaviors to help with parsing
    KNOWN_BEHAVIORS = [
        "Hindi",
        "Gujarati",
        "Marwadi",
        "Punjabi",
        "Baseline",
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

    def find_game_directories(self):
        """Find all game directories with the expected naming pattern"""
        game_dirs = []

        for item in self.results_dir.iterdir():
            if not item.is_dir():
                continue
            parts = item.name.split("_")
            if len(parts) < 4:
                continue
            # last part should look like 'iter<num>' or 'iter_<num>'
            last = parts[-1]
            iter_num = None
            if last.startswith("iter"):
                suffix = last[len("iter") :]
                suffix = suffix.lstrip("_")
                if suffix.isdigit() and 1 <= int(suffix) <= 5:
                    iter_num = int(suffix)
            if iter_num is None:
                continue
            # basic shape: model1_model2_language_iter<num>
            game_dirs.append(item)

        print(f"Found {len(game_dirs)} game directories")
        return sorted(game_dirs)

    def parse_directory_name(self, dir_name):
        """Parse directory name to extract model1, model2, behavior (language), and iteration.

        Expected pattern (no 'vs' token): model1_model2_language_iter<number>
        Example: GPT-4o_GPT-3.5_Punjabi_iter4
        """
        print(f"Parsing: {dir_name}")

        try:
            parts = dir_name.split("_")
            if len(parts) < 4:
                raise ValueError("Not enough segments for expected pattern")

            # Last part = iteration number, prefixed by 'iter'
            last = parts[-1]
            if not last.startswith("iter"):
                raise ValueError("Last segment must start with 'iter'")
            iter_suffix = last[len("iter") :].lstrip("_")
            iteration = int(iter_suffix)

            # Second to last = behavior/language
            behavior_candidate = parts[-2]
            behavior = behavior_candidate
            for known_behavior in self.KNOWN_BEHAVIORS:
                if behavior_candidate == known_behavior:
                    behavior = known_behavior
                    break

            # Remaining parts before behavior = model1 + model2
            model_parts = parts[:-2]  # everything except behavior and iteration
            if len(model_parts) < 2:
                raise ValueError("Need at least two segments for model1 and model2")

            # Try to match model1 using known models (greedy on prefix)
            def normalize(name):
                return name.replace("-", "_")

            best_idx = 1  # default split after first segment
            best_match_len = 0
            for i in range(1, len(model_parts)):
                candidate = "_".join(model_parts[:i])
                for known_model in self.KNOWN_MODELS:
                    if normalize(candidate) == normalize(known_model):
                        if i > best_match_len:
                            best_match_len = i
                            best_idx = i

            model1 = "_".join(model_parts[:best_idx])
            model2 = "_".join(model_parts[best_idx:])

            # Normalize to known canonical names if possible
            for known_model in self.KNOWN_MODELS:
                if normalize(model1) == normalize(known_model):
                    model1 = known_model
                if normalize(model2) == normalize(known_model):
                    model2 = known_model

            print(f"  Parsed: {model1} vs {model2} | {behavior} | iter {iteration}")
            return model1, model2, behavior, iteration

        except Exception as e:
            print(f"  Error parsing {dir_name}: {e}")
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
        """Extract relevant data from a single game_state.json file.

        This is tailored for the TRADING game structure, not the ultimatum game.
        We assume:
          - The final trade is stored in summary['proposed_trade'] as a trade over
            generic resources (e.g. X, Y) for both players RED and BLUE.
          - The final resources for each player are stored in summary['final_resources'],
            each being a resource dict over the same resource types.
          - Negotiation rounds correspond to numeric 'current_iteration' entries
            between 'START' and 'END' in the top-level 'game_state' list.
        """
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

            # Basic summary section
            summary = end_state.get("summary", {})

            # -------------------------------
            # Negotiation rounds
            # -------------------------------
            negotiation_rounds = 0
            try:
                numeric_iterations = []
                for state in game_state:
                    it = state.get("current_iteration")
                    # Some logs may store iterations as int, others as str
                    if isinstance(it, int):
                        numeric_iterations.append(it)
                    else:
                        try:
                            numeric_iterations.append(int(it))
                        except (TypeError, ValueError):
                            continue
                negotiation_rounds = max(numeric_iterations) if numeric_iterations else 0
            except Exception:
                negotiation_rounds = 0

            # -------------------------------
            # Extract proposed trade (final agreed trade)
            # -------------------------------
            trade_volume = 0
            proposed_trade = summary.get("proposed_trade")
            if isinstance(proposed_trade, dict) and proposed_trade.get("_type") == "trade":
                trade_value = proposed_trade.get("_value", {})

                def sum_resource(res_obj):
                    """Sum all resource units in a resource object of shape:
                    {'_type': 'resource', '_value': {'X': int, 'Y': int, ...}}
                    """
                    if not isinstance(res_obj, dict):
                        return 0
                    value = res_obj.get("_value", {})
                    if not isinstance(value, dict):
                        return 0
                    return sum(v for v in value.values() if isinstance(v, (int, float)))

                red_res = trade_value.get("RED")
                blue_res = trade_value.get("BLUE")

                red_gives = sum_resource(red_res)
                blue_gives = sum_resource(blue_res)

                # Total trade volume is the sum of all units moved between players
                trade_volume = red_gives + blue_gives

            # -------------------------------
            # Extract final response
            # -------------------------------
            final_response = summary.get("final_response", "UNKNOWN")

            # -------------------------------
            # Extract final resources and compute payoffs
            # -------------------------------
            final_resources = summary.get("final_resources", [])

            def sum_final_resource(res_obj):
                """Sum all resource units in a final resource object."""
                if not isinstance(res_obj, dict):
                    return 0
                value = res_obj.get("_value", {})
                if not isinstance(value, dict):
                    return 0
                return sum(v for v in value.values() if isinstance(v, (int, float)))

            player1_final_resources = 0
            player2_final_resources = 0

            if len(final_resources) >= 2:
                player1_final_resources = sum_final_resource(final_resources[0])
                player2_final_resources = sum_final_resource(final_resources[1])

            # Determine outcome; on REJECT, both players keep (or revert to) 0 payoff in our metric
            if final_response == "REJECT":
                player1_final_resources = 0
                player2_final_resources = 0
                outcome = "REJECT"
            elif final_response == "ACCEPT":
                outcome = "ACCEPT"
            else:
                outcome = "UNKNOWN"

            # Determine winner based on total resources (ignoring ties)
            if player1_final_resources > player2_final_resources:
                winner = "PLAYER1"
            elif player2_final_resources > player1_final_resources:
                winner = "PLAYER2"
            else:
                winner = "TIE"

            game_data = {
                "model1": model1,
                "model2": model2,
                "behavior": behavior,
                "iteration": iteration,
                # Trading-game-specific metrics
                "negotiation_rounds": negotiation_rounds,
                "trade_volume": trade_volume,
                # Outcome & payoffs
                "final_response": final_response,
                "outcome": outcome,
                "winner": winner,
                "player1_final_resources": player1_final_resources,
                "player2_final_resources": player2_final_resources,
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

            # Win rate calculation (Player 1 wins if they get more total resources than Player 2)
            player1_wins = 0
            player2_wins = 0
            draws = 0

            for game in valid_games:
                p1 = game.get("player1_final_resources", 0)
                p2 = game.get("player2_final_resources", 0)
                if p1 > p2:
                    player1_wins += 1
                elif p2 > p1:
                    player2_wins += 1
                else:
                    draws += 1

            # Win rate excluding draws
            non_draw_games = player1_wins + player2_wins
            win_rate = player1_wins / non_draw_games if non_draw_games > 0 else 0.0
            draw_rate = draws / total_games if total_games > 0 else 0.0

            # Payoff statistics (total resources for each player)
            player1_payoffs = [g.get("player1_final_resources", 0) for g in valid_games]
            player2_payoffs = [g.get("player2_final_resources", 0) for g in valid_games]

            # Trade metrics
            trade_volumes = [g.get("trade_volume", 0) for g in valid_games]
            negotiation_rounds = [g.get("negotiation_rounds", 0) for g in valid_games]

            # Summary statistics
            metrics = {
                "total_games": total_games,
                "accepts": len(accepts),
                "rejects": len(rejects),
                "acceptance_rate": len(accepts) / total_games if total_games > 0 else 0.0,
                # Win rates
                "player1_wins": player1_wins,
                "player2_wins": player2_wins,
                "draws": draws,
                "win_rate_player1": win_rate,
                "win_rate_player2": 1 - win_rate if non_draw_games > 0 else 0.0,
                "draw_rate": draw_rate,
                # Payoffs (average total resources after trade)
                "player1_payoff_avg": statistics.mean(player1_payoffs)
                if player1_payoffs
                else 0.0,
                "player1_payoff_std": statistics.stdev(player1_payoffs)
                if len(player1_payoffs) > 1
                else 0.0,
                "player1_payoffs": player1_payoffs,
                "player2_payoff_avg": statistics.mean(player2_payoffs)
                if player2_payoffs
                else 0.0,
                "player2_payoff_std": statistics.stdev(player2_payoffs)
                if len(player2_payoffs) > 1
                else 0.0,
                "player2_payoffs": player2_payoffs,
                # Trade metrics
                "avg_trade_volume": statistics.mean(trade_volumes)
                if trade_volumes
                else 0.0,
                "trade_volume_std": statistics.stdev(trade_volumes)
                if len(trade_volumes) > 1
                else 0.0,
                "trade_volumes": trade_volumes,
                "avg_negotiation_rounds": statistics.mean(negotiation_rounds)
                if negotiation_rounds
                else 0.0,
                "negotiation_rounds_std": statistics.stdev(negotiation_rounds)
                if len(negotiation_rounds) > 1
                else 0.0,
                "negotiation_rounds": negotiation_rounds,
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
                f"  Avg payoffs (total resources) - P1: {metrics['player1_payoff_avg']:.1f}, P2: {metrics['player2_payoff_avg']:.1f}"
            )
            print(
                f"  Avg trade volume: {metrics['avg_trade_volume']:.1f} | "
                f"Avg negotiation rounds: {metrics['avg_negotiation_rounds']:.1f}"
            )

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

            total_games = len(games)
            accepts = [g for g in games if g.get("outcome") == "ACCEPT"]
            rejects = [g for g in games if g.get("outcome") == "REJECT"]

            # Wins by player1 vs player2 (ties excluded from denominator)
            player1_wins = 0
            player2_wins = 0
            draws = 0
            for g in games:
                p1 = g.get("player1_final_resources", 0)
                p2 = g.get("player2_final_resources", 0)
                if p1 > p2:
                    player1_wins += 1
                elif p2 > p1:
                    player2_wins += 1
                else:
                    draws += 1

            non_draw_games = player1_wins + player2_wins
            win_rate_player1 = (
                player1_wins / non_draw_games if non_draw_games > 0 else 0.0
            )
            draw_rate = draws / total_games if total_games > 0 else 0.0

            trade_volumes = [g.get("trade_volume", 0) for g in games]
            negotiation_rounds = [g.get("negotiation_rounds", 0) for g in games]
            player1_payoffs = [g.get("player1_final_resources", 0) for g in games]
            player2_payoffs = [g.get("player2_final_resources", 0) for g in games]

            behavior_metrics = {
                "behavior": behavior,
                "total_games": total_games,
                "accepts": len(accepts),
                "rejects": len(rejects),
                "acceptance_rate": len(accepts) / total_games if total_games > 0 else 0.0,
                "player1_wins": player1_wins,
                "player2_wins": player2_wins,
                "draws": draws,
                "win_rate_player1": win_rate_player1,
                "win_rate_player2": 1 - win_rate_player1 if non_draw_games > 0 else 0.0,
                "draw_rate": draw_rate,
                "avg_trade_volume": statistics.mean(trade_volumes)
                if trade_volumes
                else 0.0,
                "trade_volume_std": statistics.stdev(trade_volumes)
                if len(trade_volumes) > 1
                else 0.0,
                "avg_negotiation_rounds": statistics.mean(negotiation_rounds)
                if negotiation_rounds
                else 0.0,
                "negotiation_rounds_std": statistics.stdev(negotiation_rounds)
                if len(negotiation_rounds) > 1
                else 0.0,
                "player1_payoff_avg": statistics.mean(player1_payoffs)
                if player1_payoffs
                else 0.0,
                "player1_payoff_std": statistics.stdev(player1_payoffs)
                if len(player1_payoffs) > 1
                else 0.0,
                "player2_payoff_avg": statistics.mean(player2_payoffs)
                if player2_payoffs
                else 0.0,
                "player2_payoff_std": statistics.stdev(player2_payoffs)
                if len(player2_payoffs) > 1
                else 0.0,
            }

            behavior_summary[behavior] = behavior_metrics

        self.behavior_summary = behavior_summary
        return behavior_summary

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

        # Save behavior-level summary (averaged across model combinations for each behavior)
        behavior_summary_file = self.results_dir / "behavior_summary.json"
        with open(behavior_summary_file, "w", encoding="utf-8") as f:
            json.dump(self.behavior_summary, f, indent=2, ensure_ascii=False)

        # Also save behavior-level summary as CSV (for easy table generation)
        behavior_csv_file = self.results_dir / "behavior_summary.csv"
        with open(behavior_csv_file, "w", encoding="utf-8") as f:
            headers = [
                "behavior",
                "total_games",
                "accepts",
                "rejects",
                "acceptance_rate",
                "win_rate_player1",
                "win_rate_player2",
                "draw_rate",
                "avg_trade_volume",
                "trade_volume_std",
                "avg_negotiation_rounds",
                "negotiation_rounds_std",
                "player1_payoff_avg",
                "player1_payoff_std",
                "player2_payoff_avg",
                "player2_payoff_std",
            ]
            f.write(",".join(headers) + "\n")
            for behavior, metrics in self.behavior_summary.items():
                row = [
                    behavior,
                    str(metrics["total_games"]),
                    str(metrics["accepts"]),
                    str(metrics["rejects"]),
                    f"{metrics['acceptance_rate']:.4f}",
                    f"{metrics['win_rate_player1']:.4f}",
                    f"{metrics['win_rate_player2']:.4f}",
                    f"{metrics['draw_rate']:.4f}",
                    f"{metrics['avg_trade_volume']:.4f}",
                    f"{metrics['trade_volume_std']:.4f}",
                    f"{metrics['avg_negotiation_rounds']:.4f}",
                    f"{metrics['negotiation_rounds_std']:.4f}",
                    f"{metrics['player1_payoff_avg']:.4f}",
                    f"{metrics['player1_payoff_std']:.4f}",
                    f"{metrics['player2_payoff_avg']:.4f}",
                    f"{metrics['player2_payoff_std']:.4f}",
                ]
                f.write(",".join(row) + "\n")

        # Create readable summary
        readable_file = self.results_dir / "readable_summary.txt"
        with open(readable_file, "w", encoding="utf-8") as f:
            f.write("TRADING GAME RESULTS SUMMARY\n")
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
                    f"Average Trade Volume: {metrics['avg_trade_volume']:.1f} ± {metrics['trade_volume_std']:.1f}\n"
                )
                f.write(
                    f"Average Negotiation Rounds: {metrics['avg_negotiation_rounds']:.1f} ± {metrics['negotiation_rounds_std']:.1f}\n"
                )
                f.write(
                    f"Player 1 Average Payoff (total resources): {metrics['player1_payoff_avg']:.1f} ± {metrics['player1_payoff_std']:.1f}\n"
                )
                f.write(
                    f"Player 2 Average Payoff (total resources): {metrics['player2_payoff_avg']:.1f} ± {metrics['player2_payoff_std']:.1f}\n"
                )
                f.write("\n")

            # Behavior-level aggregated metrics (averaged across all model combinations)
            f.write("\nBEHAVIOR-LEVEL AVERAGES\n")
            f.write("=" * 50 + "\n")
            for behavior, metrics in self.behavior_summary.items():
                f.write(f"{behavior}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Games: {metrics['total_games']}\n")
                f.write(f"Acceptance Rate: {metrics['acceptance_rate']:.3f}\n")
                f.write(f"Player 1 Win Rate: {metrics['win_rate_player1']:.3f}\n")
                f.write(f"Player 2 Win Rate: {metrics['win_rate_player2']:.3f}\n")
                f.write(f"Draw Rate: {metrics['draw_rate']:.3f}\n")
                f.write(
                    f"Average Trade Volume: {metrics['avg_trade_volume']:.1f} ± {metrics['trade_volume_std']:.1f}\n"
                )
                f.write(
                    f"Average Negotiation Rounds: {metrics['avg_negotiation_rounds']:.1f} ± {metrics['negotiation_rounds_std']:.1f}\n"
                )
                f.write(
                    f"Player 1 Average Payoff (total resources): {metrics['player1_payoff_avg']:.1f} ± {metrics['player1_payoff_std']:.1f}\n"
                )
                f.write(
                    f"Player 2 Average Payoff (total resources): {metrics['player2_payoff_avg']:.1f} ± {metrics['player2_payoff_std']:.1f}\n"
                )
                f.write("\n")

        print(f"\nResults saved:")
        print(f"  Raw data: {raw_data_file}")
        print(f"  Summary: {summary_file}")
        print(f"  Readable: {readable_file}")
        print(f"  Behavior summary (JSON): {behavior_summary_file}")
        print(f"  Behavior summary (CSV): {behavior_csv_file}")

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
            payoff_p1 = {}
            payoff_p2 = {}

            for model1 in models:
                win_rates[model1] = {}
                payoff_p1[model1] = {}
                payoff_p2[model1] = {}

                for model2 in models:
                    combo_key = f"{model1}_vs_{model2}_{behavior}"

                    if combo_key in self.summary:
                        metrics = self.summary[combo_key]
                        win_rates[model1][model2] = metrics["win_rate_player1"]
                        # Payoff heatmaps: average total resources after trade
                        payoff_p1[model1][model2] = metrics["player1_payoff_avg"]
                        payoff_p2[model1][model2] = metrics["player2_payoff_avg"]
                    else:
                        win_rates[model1][model2] = None
                        payoff_p1[model1][model2] = None
                        payoff_p2[model1][model2] = None

            heatmap_data[behavior] = {
                "win_rates": win_rates,
                "payoff_player1": payoff_p1,
                "payoff_player2": payoff_p2,
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

    analyzer = TradingResultsAnalyzer(results_dir)

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