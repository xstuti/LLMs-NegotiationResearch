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
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import levene, mannwhitneyu
from itertools import combinations
import statsmodels.api as sm
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import sys

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

            # Extract final resources - sum ALL resource types
            final_resources = summary.get("final_resources", [])
            player1_final = 0
            player2_final = 0

            if len(final_resources) >= 2:
                # Sum all resources for player 1
                player1_resources = final_resources[0].get("_value", {})
                player1_final = sum(player1_resources.values())

                # Sum all resources for player 2
                player2_resources = final_resources[1].get("_value", {})
                player2_final = sum(player2_resources.values())

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
    
    def welch_anova(self, groups_dict):
        """Perform Welch's ANOVA (robust to unequal variances)."""
        data = []
        for group, values in groups_dict.items():
            for v in values:
                if v is not None and not np.isnan(v):
                    data.append({"group": group, "value": v})
        
        if len(data) == 0:
            return {"F": np.nan, "p_value": np.nan, "df_num": np.nan, "df_den": np.nan}
        
        df = pd.DataFrame(data)
        
        try:
            result = anova_oneway(df["value"], df["group"], use_var="unequal")
            
            if hasattr(result, 'statistic'):
                f_stat = result.statistic
                p_val = result.pvalue
            else:
                f_stat = result[0]
                p_val = result[1]
            
            k = len(groups_dict)
            df_num = k - 1
            
            groups_list = list(groups_dict.values())
            ns = [len(g) for g in groups_list]
            vars = [np.var(g, ddof=1) if len(g) > 1 else 0 for g in groups_list]
            
            numerator = sum([(1 - n_i/sum(ns)) * var_i for n_i, var_i in zip(ns, vars)])**2
            denominator = sum([((1 - n_i/sum(ns))**2 * var_i**2) / (n_i - 1) for n_i, var_i in zip(ns, vars)])
            
            if denominator > 0:
                df_den = numerator / denominator
            else:
                df_den = sum(ns) - k
            
            return {
                "F": float(f_stat),
                "p_value": float(p_val),
                "df_num": float(df_num),
                "df_den": float(df_den)
            }
        except Exception as e:
            print(f"Error in Welch ANOVA: {e}")
            return {"F": np.nan, "p_value": np.nan, "df_num": np.nan, "df_den": np.nan}


    def welch_ttest(self, group1, group2):
        """Perform Welch's t-test."""
        group1 = np.array([x for x in group1 if x is not None and not np.isnan(x)])
        group2 = np.array([x for x in group2 if x is not None and not np.isnan(x)])
        
        if len(group1) < 2 or len(group2) < 2:
            return {"t": np.nan, "p_value": np.nan, "df": np.nan}
        
        t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
        
        n1, n2 = len(group1), len(group2)
        v1, v2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        if v1 == 0 and v2 == 0:
            df = n1 + n2 - 2
        else:
            df = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))
        
        return {"t": t_stat, "p_value": p_val, "df": df}


    def hedges_g(self, group1, group2):
        """Calculate Hedges' g effect size."""
        group1 = np.array([x for x in group1 if x is not None and not np.isnan(x)])
        group2 = np.array([x for x in group2 if x is not None and not np.isnan(x)])
        
        n1, n2 = len(group1), len(group2)
        
        if n1 < 2 or n2 < 2:
            return np.nan
        
        pooled_std = np.sqrt(((n1-1)*np.var(group1, ddof=1) + (n2-1)*np.var(group2, ddof=1)) / (n1+n2-2))
        
        if pooled_std == 0:
            return 0.0
        
        d = (np.mean(group1) - np.mean(group2)) / pooled_std
        correction = 1 - (3 / (4*(n1+n2-2) - 1))
        
        return d * correction


    def omega_squared(self, groups):
        """Calculate omega-squared effect size."""
        all_vals = np.concatenate(groups)
        grand_mean = np.mean(all_vals)
        n_total = len(all_vals)
        k = len(groups)
        
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_within = sum(np.sum((g - np.mean(g))**2) for g in groups)
        
        ms_between = ss_between / (k - 1)
        ms_within = ss_within / (n_total - k)
        
        omega2 = (ss_between - (k-1)*ms_within) / (ss_between + ss_within + ms_within)
        
        return max(0, omega2)


    def calculate_behavior_summary(self):
        """Calculate aggregate metrics by behavior across all model combinations."""
        behavior_groups = defaultdict(list)
        
        for game in self.raw_data:
            behavior_groups[game["behavior"]].append(game)
        
        behavior_summary = {}
        
        for behavior, games in behavior_groups.items():
            valid_games = [g for g in games if g["outcome"] in ["ACCEPT", "REJECT"]]
            
            if not valid_games:
                continue
            
            # Binary metrics
            accepts = [1 if g["outcome"] == "ACCEPT" else 0 for g in valid_games]
            
            # Continuous metrics
            initial_offers = [g["initial_offer"] for g in valid_games]
            player1_payoffs = [g["player1_final"] for g in valid_games]
            player2_payoffs = [g["player2_final"] for g in valid_games]
            
            # Win rates
            p1_wins = sum(1 for g in valid_games if g["player1_final"] > g["player2_final"])
            p2_wins = sum(1 for g in valid_games if g["player2_final"] > g["player1_final"])
            draws = sum(1 for g in valid_games if g["player1_final"] == g["player2_final"])
            
            non_draws = p1_wins + p2_wins
            win_rate_p1 = p1_wins / non_draws if non_draws > 0 else 0.0
            
            def mean_std(lst):
                m = statistics.mean(lst) if lst else 0.0
                s = statistics.stdev(lst) if len(lst) > 1 else 0.0
                return m, s
            
            acc_m, acc_s = mean_std(accepts)
            offer_m, offer_s = mean_std(initial_offers)
            p1_m, p1_s = mean_std(player1_payoffs)
            p2_m, p2_s = mean_std(player2_payoffs)
            
            behavior_summary[behavior] = {
                "behavior": behavior,
                "total_games": len(valid_games),
                "acceptance_rate_mean": acc_m,
                "acceptance_rate_std": acc_s,
                "initial_offer_mean": offer_m,
                "initial_offer_std": offer_s,
                "player1_payoff_mean": p1_m,
                "player1_payoff_std": p1_s,
                "player2_payoff_mean": p2_m,
                "player2_payoff_std": p2_s,
                "player1_wins": p1_wins,
                "player2_wins": p2_wins,
                "draws": draws,
                "win_rate_player1": win_rate_p1,
            }
        
        return behavior_summary


    def run_comprehensive_statistical_analysis(self):
        """
        Comprehensive statistical analysis for Ultimatum Game.
        
        Key aspects:
        1. Initial Offer - continuous, reflects proposer strategy
        2. Acceptance Rate - binary, reflects responder threshold
        3. Player Payoffs - continuous, reflects final outcomes
        4. Win Rate - derived from payoffs, strategic success
        """
        out_dir = Path(self.results_dir) / "stats"
        out_dir.mkdir(exist_ok=True)
        
        log_file = out_dir / "statistical_analysis_log.txt"
        original_stdout = sys.stdout
        
        with open(log_file, 'w', encoding='utf-8') as log_f:
            sys.stdout = log_f
            
            df = pd.DataFrame(self.raw_data)
            df = df[df["outcome"].isin(["ACCEPT", "REJECT"])]
            
            print("="*80)
            print("COMPREHENSIVE STATISTICAL ANALYSIS - ULTIMATUM GAME")
            print("="*80)
            print(f"\nTotal valid games: {len(df)}")
            print(f"Accepted: {len(df[df['outcome'] == 'ACCEPT'])}")
            print(f"Rejected: {len(df[df['outcome'] == 'REJECT'])}")
            
            # Calculate behavior-level summaries
            behavior_summary = self.calculate_behavior_summary()
            
            results = []
            
            # ===================================================================
            # CONTINUOUS METRICS - PARAMETRIC TESTS
            # ===================================================================
            
            continuous_metrics = {
                "initial_offer": "Initial Offer (Proposer Strategy)",
                "player1_final": "Player 1 Payoff (Proposer Outcome)",
                "player2_final": "Player 2 Payoff (Responder Outcome)",
            }
            
            for metric, label in continuous_metrics.items():
                print(f"\n{'='*70}")
                print(f"METRIC: {label}")
                print(f"{'='*70}")
                
                # Gather data by behavior
                groups_dict = {}
                behaviors = []
                
                for b in sorted(df["behavior"].unique()):
                    vals = df.loc[
                        (df["behavior"] == b) & df[metric].notna(), 
                        metric
                    ].values
                    
                    if len(vals) >= 5:
                        groups_dict[b] = vals
                        behaviors.append(b)
                        print(f"  {b}: n={len(vals)}, mean={np.mean(vals):.2f}, "
                            f"std={np.std(vals, ddof=1):.2f}")
                
                if len(groups_dict) < 2:
                    print(f"  Skipping {label} - insufficient groups")
                    continue
                
                groups = list(groups_dict.values())
                
                # 1. Variance homogeneity test
                lev_stat, lev_p = levene(*groups)
                print(f"\n  Levene's Test: W={lev_stat:.4f}, p={lev_p:.4f}")
                
                if lev_p < 0.05:
                    print("  → Variances UNEQUAL (p < 0.05) - Welch's tests appropriate")
                else:
                    print("  → Variances equal (p >= 0.05)")
                
                # 2. Welch's ANOVA
                welch_result = self.welch_anova(groups_dict)
                print(f"\n  Welch's ANOVA:")
                print(f"    F({welch_result['df_num']:.2f}, {welch_result['df_den']:.2f}) = {welch_result['F']:.4f}")
                print(f"    p-value = {welch_result['p_value']:.4e}")
                
                omega2 = self.omega_squared(groups)
                print(f"    Omega-squared = {omega2:.4f}")
                
                # Interpret effect size
                if omega2 < 0.01:
                    effect_interp = "negligible"
                elif omega2 < 0.06:
                    effect_interp = "small"
                elif omega2 < 0.14:
                    effect_interp = "medium"
                else:
                    effect_interp = "large"
                
                print(f"    Effect size interpretation: {effect_interp}")
                
                results.append({
                    "metric": label,
                    "test": "Welch_ANOVA",
                    "F": welch_result['F'],
                    "df_num": welch_result['df_num'],
                    "df_den": welch_result['df_den'],
                    "p_value": welch_result['p_value'],
                    "omega_squared": omega2,
                    "effect_interpretation": effect_interp,
                    "levene_W": lev_stat,
                    "levene_p": lev_p,
                    "significant": welch_result['p_value'] < 0.05
                })
                
                # 3. Pairwise comparisons (if overall significant)
                if welch_result['p_value'] < 0.05:
                    print(f"\n  Pairwise Comparisons (Welch's t-tests):")
                    
                    pairwise_results = []
                    
                    for b1, b2 in combinations(behaviors, 2):
                        g1, g2 = groups_dict[b1], groups_dict[b2]
                        
                        ttest_result = self.welch_ttest(g1, g2)
                        hedges = self.hedges_g(g1, g2)
                        
                        pairwise_results.append({
                            "pair": f"{b1} vs {b2}",
                            "p_value": ttest_result['p_value']
                        })
                        
                        results.append({
                            "metric": label,
                            "test": "Welch_t_test",
                            "comparison": f"{b1} vs {b2}",
                            "t": ttest_result['t'],
                            "df": ttest_result['df'],
                            "p_value": ttest_result['p_value'],
                            "hedges_g": hedges,
                            "mean_diff": np.mean(g1) - np.mean(g2)
                        })
                    
                    # Bonferroni correction
                    p_values = [r['p_value'] for r in pairwise_results]
                    _, p_corrected, _, _ = multipletests(p_values, method='bonferroni')
                    
                    print(f"\n    {'Comparison':<35} {'Mean Diff':<12} {'Hedges g':<10} {'p':<10} {'p_corr':<10} {'Sig'}")
                    print(f"    {'-'*90}")
                    
                    idx = 0
                    for b1, b2 in combinations(behaviors, 2):
                        g1, g2 = groups_dict[b1], groups_dict[b2]
                        mean_diff = np.mean(g1) - np.mean(g2)
                        hedges = self.hedges_g(g1, g2)
                        
                        sig = "***" if p_corrected[idx] < 0.001 else "**" if p_corrected[idx] < 0.01 else "*" if p_corrected[idx] < 0.05 else "ns"
                        
                        print(f"    {f'{b1} vs {b2}':<35} {mean_diff:>11.2f} {hedges:>9.3f} {p_values[idx]:>9.4f} {p_corrected[idx]:>9.4f} {sig:>3}")
                        idx += 1
                else:
                    print("\n  → No significant overall effect, skipping pairwise comparisons")
            
            # ===================================================================
            # BINARY METRIC: ACCEPTANCE RATE
            # ===================================================================
            
            print(f"\n{'='*70}")
            print("BINARY METRIC: Acceptance Rate")
            print(f"{'='*70}")
            
            behaviors = sorted(df["behavior"].unique())
            contingency = []
            
            for b in behaviors:
                sub = df[df["behavior"] == b]
                accepted = int((sub["outcome"] == "ACCEPT").sum())
                rejected = int((sub["outcome"] == "REJECT").sum())
                total = accepted + rejected
                rate = accepted / total if total > 0 else 0
                
                contingency.append([accepted, rejected])
                print(f"  {b}: {accepted}/{total} accepted ({rate*100:.1f}%)")
            
            contingency = np.array(contingency)
            
            # Check for degenerate cases
            degenerate = any(row[0] == 0 or row[1] == 0 for row in contingency)
            
            if degenerate:
                print("\n  WARNING: At least one behavior has perfect acceptance/rejection")
                print("  Chi-square test may be unreliable")
                
                results.append({
                    "metric": "Acceptance Rate",
                    "test": "Chi_square",
                    "note": "Degenerate case - perfect separation",
                    "chi2": np.nan,
                    "p_value": np.nan
                })
            else:
                chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
                print(f"\n  Chi-square test: chi-squared({dof}) = {chi2:.4f}, p = {p_chi:.4f}")
                
                # Cramer's V effect size for chi-square
                n = contingency.sum()
                cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))
                print(f"  Cramer's V = {cramers_v:.4f}")
                
                if cramers_v < 0.1:
                    effect_interp = "negligible"
                elif cramers_v < 0.3:
                    effect_interp = "small"
                elif cramers_v < 0.5:
                    effect_interp = "medium"
                else:
                    effect_interp = "large"
                
                print(f"  Effect size interpretation: {effect_interp}")
                
                results.append({
                    "metric": "Acceptance Rate",
                    "test": "Chi_square",
                    "chi2": chi2,
                    "df": dof,
                    "p_value": p_chi,
                    "cramers_v": cramers_v,
                    "effect_interpretation": effect_interp,
                    "significant": p_chi < 0.05
                })
                
                # Pairwise proportion tests
                if p_chi < 0.05:
                    print("\n  Pairwise Proportion Tests:")
                    
                    from statsmodels.stats.proportion import proportions_ztest
                    pairwise_p = []
                    
                    for i, b1 in enumerate(behaviors):
                        for j, b2 in enumerate(behaviors):
                            if i >= j:
                                continue
                            
                            count = np.array([contingency[i, 0], contingency[j, 0]])
                            nobs = np.array([contingency[i].sum(), contingency[j].sum()])
                            
                            z_stat, p_val = proportions_ztest(count, nobs)
                            pairwise_p.append(p_val)
                            
                            results.append({
                                "metric": "Acceptance Rate",
                                "test": "Proportion_test",
                                "comparison": f"{b1} vs {b2}",
                                "z": z_stat,
                                "p_value": p_val
                            })
                    
                    # Bonferroni correction
                    _, p_corrected, _, _ = multipletests(pairwise_p, method='bonferroni')
                    
                    print(f"\n    {'Comparison':<35} {'Rate Diff':<12} {'p':<10} {'p_corr':<10} {'Sig'}")
                    print(f"    {'-'*80}")
                    
                    idx = 0
                    for i, b1 in enumerate(behaviors):
                        for j, b2 in enumerate(behaviors):
                            if i >= j:
                                continue
                            
                            rate1 = contingency[i, 0] / contingency[i].sum()
                            rate2 = contingency[j, 0] / contingency[j].sum()
                            diff = rate1 - rate2
                            
                            sig = "***" if p_corrected[idx] < 0.001 else "**" if p_corrected[idx] < 0.01 else "*" if p_corrected[idx] < 0.05 else "ns"
                            
                            print(f"    {f'{b1} vs {b2}':<35} {diff:>11.3f} {pairwise_p[idx]:>9.4f} {p_corrected[idx]:>9.4f} {sig:>3}")
                            idx += 1
            
            # ===================================================================
            # SAVE RESULTS
            # ===================================================================
            
            print(f"\n{'='*80}")
            print("SAVING RESULTS")
            print(f"{'='*80}")
            
            results_df = pd.DataFrame(results)
            
            csv_file = out_dir / "statistical_tests_ultimatum.csv"
            results_df.to_csv(csv_file, index=False)
            print(f"  Saved: {csv_file}")
            
            json_file = out_dir / "statistical_tests_ultimatum.json"
            results_df.to_json(json_file, orient='records', indent=2)
            print(f"  Saved: {json_file}")
            
            # Save behavior summary
            behavior_df = pd.DataFrame(behavior_summary.values())
            behavior_csv = out_dir / "behavior_summary_ultimatum.csv"
            behavior_df.to_csv(behavior_csv, index=False)
            print(f"  Saved: {behavior_csv}")
            
            # Create interpretation guide
            interp_file = out_dir / "interpretation_guide.txt"
            with open(interp_file, 'w', encoding='utf-8') as inf:
                inf.write("="*80 + "\n")
                inf.write("STATISTICAL RESULTS INTERPRETATION GUIDE - ULTIMATUM GAME\n")
                inf.write("="*80 + "\n\n")
                
                inf.write("HOW TO INTERPRET THESE RESULTS FOR YOUR PAPER\n")
                inf.write("-"*80 + "\n\n")
                
                # Overall tests
                inf.write("1. OVERALL LANGUAGE EFFECTS\n\n")
                overall_tests = results_df[results_df['test'].isin(['Welch_ANOVA', 'Chi_square'])]
                
                for _, row in overall_tests.iterrows():
                    inf.write(f"{row['metric']}:\n")
                    
                    if row['test'] == 'Welch_ANOVA':
                        inf.write(f"  Statistical Test: Welch's ANOVA\n")
                        inf.write(f"  F({row['df_num']:.1f}, {row['df_den']:.1f}) = {row['F']:.3f}, p = {row['p_value']:.4f}\n")
                        inf.write(f"  Effect Size: omega-squared = {row['omega_squared']:.4f} ({row['effect_interpretation']})\n")
                        
                        if row['significant']:
                            inf.write(f"  INTERPRETATION: Languages DIFFER significantly on {row['metric']}\n")
                            inf.write(f"  → See pairwise comparisons below for which languages differ\n")
                        else:
                            inf.write(f"  INTERPRETATION: Languages DO NOT differ significantly\n")
                            inf.write(f"  → No reliable evidence that language affects {row['metric']}\n")
                    
                    elif row['test'] == 'Chi_square':
                        if pd.notna(row.get('chi2')):
                            inf.write(f"  Statistical Test: Chi-square test\n")
                            inf.write(f"  chi-squared({row['df']:.0f}) = {row['chi2']:.3f}, p = {row['p_value']:.4f}\n")
                            inf.write(f"  Effect Size: Cramer's V = {row.get('cramers_v', 'NA'):.4f} ({row.get('effect_interpretation', 'NA')})\n")
                            
                            if row['significant']:
                                inf.write(f"  INTERPRETATION: Acceptance rates DIFFER across languages\n")
                            else:
                                inf.write(f"  INTERPRETATION: Acceptance rates DO NOT differ\n")
                                inf.write(f"  → Languages produce similar acceptance levels\n")
                        else:
                            inf.write(f"  WARNING: Degenerate case (perfect separation)\n")
                            inf.write(f"  → At least one language has 100% or 0% acceptance\n")
                    
                    inf.write("\n")
                
                # Significant pairwise comparisons
                inf.write("\n2. SIGNIFICANT PAIRWISE DIFFERENCES (after Bonferroni correction)\n\n")
                pairwise = results_df[results_df['test'].isin(['Welch_t_test', 'Proportion_test'])]
                
                for metric in pairwise['metric'].unique():
                    metric_tests = pairwise[pairwise['metric'] == metric]
                    
                    # Estimate Bonferroni threshold
                    n_comparisons = len(metric_tests)
                    bonf_threshold = 0.05 / n_comparisons if n_comparisons > 0 else 0.05
                    
                    sig_tests = metric_tests[metric_tests['p_value'] < bonf_threshold]
                    
                    if len(sig_tests) > 0:
                        inf.write(f"{metric}:\n")
                        for _, row in sig_tests.iterrows():
                            inf.write(f"  {row['comparison']}:\n")
                            
                            if row['test'] == 'Welch_t_test':
                                inf.write(f"    Mean difference = {row.get('mean_diff', 'NA'):.2f}\n")
                                inf.write(f"    Hedges' g = {row.get('hedges_g', 'NA'):.3f}\n")
                                inf.write(f"    p = {row['p_value']:.4f}\n")
                                
                                g = row.get('hedges_g', 0)
                                if abs(g) < 0.2:
                                    effect = "negligible"
                                elif abs(g) < 0.5:
                                    effect = "small"
                                elif abs(g) < 0.8:
                                    effect = "medium"
                                else:
                                    effect = "large"
                                
                                inf.write(f"    Effect size: {effect}\n")
                                
                            elif row['test'] == 'Proportion_test':
                                inf.write(f"    z = {row.get('z', 'NA'):.3f}\n")
                                inf.write(f"    p = {row['p_value']:.4f}\n")
                            
                            inf.write("\n")
                    else:
                        inf.write(f"{metric}: No significant pairwise differences\n\n")
                
                # How to report in paper
                inf.write("\n3. HOW TO REPORT IN YOUR PAPER\n\n")
                inf.write("Example for Initial Offer:\n")
                inf.write('  "Initial offers differed significantly across languages,\n')
                inf.write('   Welch\'s F(4, 450.2) = 5.23, p < .001, omega-squared = .042.\n')
                inf.write('   Pairwise comparisons with Bonferroni correction revealed\n')
                inf.write('   that Hindi yielded significantly lower offers (M = 15.5)\n')
                inf.write('   compared to English (M = 21.7, p_corrected = .003, g = -0.28)."\n\n')
                
                inf.write("Example for Acceptance Rate:\n")
                inf.write('  "Acceptance rates differed significantly across languages,\n')
                inf.write('   chi-squared(4) = 12.34, p = .015, Cramer\'s V = .15.\n')
                inf.write('   Hindi showed lower acceptance (87.6%) compared to\n')
                inf.write('   English (98.2%, p_corrected = .008)."\n\n')
                
                inf.write("Example for Non-Significant Result:\n")
                inf.write('  "Player 2 payoffs did not differ significantly across languages,\n')
                inf.write('   Welch\'s F(4, 445.3) = 1.23, p = .30, suggesting that responder\n')
                inf.write('   outcomes remain stable regardless of linguistic framing."\n')
            
            print(f"  Saved: {interp_file}")
            
            print(f"\n{'='*80}")
            print("STATISTICAL ANALYSIS COMPLETE")
            print(f"{'='*80}\n")
        
        sys.stdout = original_stdout
        
        print(f"\nStatistical analysis complete. Results saved to: {out_dir}")
        print(f"  - Full log: {log_file}")
        print(f"  - CSV results: {csv_file}")
        print(f"  - JSON results: {json_file}")
        print(f"  - Behavior summary: {behavior_csv}")
        print(f"  - Interpretation guide: {interp_file}")
        
        return results_df

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
