#!/usr/bin/env python3
"""
Analyze Real Ultimatum Game Results

This script analyzes the actual game results from the log files,
extracting key metrics from game_state.json files.
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
from scipy.stats import mannwhitneyu, kruskal
from statsmodels.stats.proportion import proportions_ztest
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
                if suffix.isdigit() and 1 <= int(suffix) <= 20:
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

            # Core structures
            game_state = data.get("game_state", [])
            players = data.get("players", [])

            # Locate START and END states (if present)
            start_state = None
            end_state = None
            for state in game_state:
                ci = state.get("current_iteration")
                if ci == "START":
                    start_state = state
                if ci == "END":
                    end_state = state
                    break

            # Determine whether any player actually ACCEPTed by scanning
            # 1) state-level public info dicts (player1_response/player2_response)
            # 2) top-level state entries that include player_public_info_dict
            # 3) each player's `conversation` contents for <player answer> tags
            accepted = False
            final_accepting_state = None

            # (A) scan game_state entries
            for state in game_state:
                # check explicit player response objects
                for player_key in ["player1_response", "player2_response", "player_public_info_dict"]:
                    resp = state.get(player_key, {})
                    if isinstance(resp, dict):
                        pub_info = resp.get("player_public_info_dict", resp) if player_key != "player_public_info_dict" else resp
                        if isinstance(pub_info, dict) and pub_info.get("player answer", "").upper() == "ACCEPT":
                            accepted = True
                            final_accepting_state = state
                            break
                if accepted:
                    break

                # also check the human-readable public answer string for accept/proposal/reject tags
                pstr = state.get("player_public_answer_string")
                if isinstance(pstr, str) and pstr:
                    up = pstr.upper()
                    if "<ACCEPT" in up or "<ACCEPT>" in up:
                        accepted = True
                        final_accepting_state = state
                        break

            # (B) scan players' conversation text if still not found
            if not accepted:
                import re

                # match either <player answer>ACCEPT</player answer> or tags like <ACCEPT>, <PROPOSAL>, <REJECT>
                tag_player_ans_re = re.compile(r"<\s*player\s+answer\s*>\s*([A-Za-z]+)\s*<", re.IGNORECASE)
                tag_simple_re = re.compile(r"<\s*(ACCEPT|REJECT|PROPOSAL)\b", re.IGNORECASE)
                for p in players:
                    conv = p.get("conversation", [])
                    for msg in conv:
                        content = msg.get("content") if isinstance(msg, dict) else (msg if isinstance(msg, str) else "")
                        if not content:
                            continue
                        # check full player-answer tag first
                        m = tag_player_ans_re.search(content)
                        if m:
                            ans = m.group(1).upper()
                            if ans == "ACCEPT":
                                accepted = True
                                break
                        # check for simple tags like <ACCEPT>
                        m2 = tag_simple_re.search(content)
                        if m2:
                            ans2 = m2.group(1).upper()
                            if ans2 == "ACCEPT":
                                accepted = True
                                break
                    if accepted:
                        break

            # Find the END iteration in game_state
            if not end_state:
                # Check if the game ended successfully without END state
                if game_state:
                    last_state = game_state[-1]
                    p1_resp = last_state.get("player1_response", {})
                    p2_resp = last_state.get("player2_response", {})
                    if (p1_resp.get("tag") == "ACCEPT" or 
                        p2_resp.get("tag") == "ACCEPT"):
                        # Use the last state as end_state
                        end_state = last_state
                        print(f"Using last state as end state for {game_state_file}")
                    else:
                        print(f"Warning: No END state and last state not accepted in {game_state_file}")
                        return None
                else:
                    print(f"Warning: No END state found in {game_state_file}")
                    return None

            # Basic summary section
            summary = end_state.get("summary", {})

            # If no summary, find the last proposed trade
            proposed_trade = summary.get("proposed_trade")
            if not proposed_trade:
                # Find the last non-NONE newly proposed trade
                for state in reversed(game_state):
                    for player_key in ["player1_response", "player2_response"]:
                        resp = state.get(player_key, {})
                        pub_info = resp.get("player_public_info_dict", {})
                        trade = pub_info.get("newly proposed trade")
                        if trade and trade != "NONE" and isinstance(trade, dict):
                            proposed_trade = trade
                            break
                    if proposed_trade:
                        break

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
            if not final_response or final_response == "UNKNOWN":
                if accepted:
                    final_response = "ACCEPT"
                else:
                    final_response = "REJECT"  # or UNKNOWN, but assume REJECT if not accepted

            # -------------------------------
            # Extract final resources and compute payoffs
            # -------------------------------
            final_resources = summary.get("final_resources", [])

            # If no final_resources and we have proposed_trade, calculate from initial
            if not final_resources and proposed_trade:
                initial_resources = data.get("player_initial_resources", [])
                if len(initial_resources) >= 2:
                    initial_red = initial_resources[0].get("_value", {})
                    initial_blue = initial_resources[1].get("_value", {})
                    trade_value = proposed_trade.get("_value", {})
                    red_trade = trade_value.get("RED", {}).get("_value", {})
                    blue_trade = trade_value.get("BLUE", {}).get("_value", {})
                    final_red = {k: initial_red.get(k, 0) - red_trade.get(k, 0) + blue_trade.get(k, 0) for k in set(initial_red) | set(red_trade) | set(blue_trade)}
                    final_blue = {k: initial_blue.get(k, 0) - blue_trade.get(k, 0) + red_trade.get(k, 0) for k in set(initial_blue) | set(red_trade) | set(blue_trade)}
                    final_resources = [
                        {"_type": "resource", "_value": final_red},
                        {"_type": "resource", "_value": final_blue}
                    ]

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

            # Accepted games for metrics that should only include successful trades
            accepted_games = [g for g in valid_games if g.get("outcome") == "ACCEPT"]

            # Calculate requested metrics only
            total_games = len(valid_games)
            accepts = [1 if g["outcome"] == "ACCEPT" else 0 for g in valid_games]

            # Trade volume (only from accepted trades)
            trade_volumes = [g.get("trade_volume", 0) for g in accepted_games]

            # Negotiation rounds (keep for all valid games)
            negotiation_rounds = [g.get("negotiation_rounds", 0) for g in valid_games]

            # Payoffs (only from accepted trades)
            player1_payoffs = [g.get("player1_final_resources", 0) for g in accepted_games]
            player2_payoffs = [g.get("player2_final_resources", 0) for g in accepted_games]

            # Win counts ignoring ties, only for accepted trades
            accepted_games = [g for g in valid_games if g.get("outcome") == "ACCEPT"]
            p1_wins = 0
            p2_wins = 0
            for g in accepted_games:
                p1 = g.get("player1_final_resources", 0)
                p2 = g.get("player2_final_resources", 0)
                if p1 > p2:
                    p1_wins += 1
                elif p2 > p1:
                    p2_wins += 1

            non_draws = p1_wins + p2_wins
            win_rate_p1 = p1_wins / non_draws if non_draws > 0 else 0.0

            # Statistics (means and stds)
            def mean_std(lst):
                m = statistics.mean(lst) if lst else 0.0
                s = statistics.stdev(lst) if len(lst) > 1 else 0.0
                return m, s

            trade_mean, trade_std = mean_std(trade_volumes)
            accepts_mean, accepts_std = mean_std(accepts)
            p1_mean, p1_std = mean_std(player1_payoffs)
            p2_mean, p2_std = mean_std(player2_payoffs)

            metrics = {
                "total_games": total_games,
                "acceptance_rate_mean": accepts_mean,
                "acceptance_rate_std": accepts_std,
                "avg_trade_volume_mean": trade_mean,
                "avg_trade_volume_std": trade_std,
                "avg_negotiation_rounds": statistics.mean(negotiation_rounds) if negotiation_rounds else 0.0,
                "negotiation_rounds_std": statistics.stdev(negotiation_rounds) if len(negotiation_rounds) > 1 else 0.0,
                "player1_payoff_mean": p1_mean,
                "player1_payoff_std": p1_std,
                "player2_payoff_mean": p2_mean,
                "player2_payoff_std": p2_std,
                "player1_wins": p1_wins,
                "player2_wins": p2_wins,
                "non_draws": non_draws,
                "win_rate_player1": win_rate_p1,
                "model1": model1,
                "model2": model2,
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
            print(f"  Win rate (P1, excluding ties): {win_rate_p1:.3f} | Draw rate: {draw_rate:.3f}")
            print(f"  Avg payoffs (total resources) - P1: {metrics['player1_payoff_mean']:.1f}, P2: {metrics['player2_payoff_mean']:.1f}")
            print(
                f"  Avg trade volume: {metrics['avg_trade_volume_mean']:.1f} ± {metrics['avg_trade_volume_std']:.1f} | "
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

            # compute per-game binary accepts and numeric metrics
            accepts = [1 if g.get("outcome") == "ACCEPT" else 0 for g in games]
            trade_volumes = [g.get("trade_volume", 0) for g in games]
            negotiation_rounds = [g.get("negotiation_rounds", 0) for g in games]
            player1_payoffs = [g.get("player1_final_resources", 0) for g in games]
            player2_payoffs = [g.get("player2_final_resources", 0) for g in games]

            # wins ignoring ties
            p1_wins = sum(1 for g in games if g.get("player1_final_resources", 0) > g.get("player2_final_resources", 0))
            p2_wins = sum(1 for g in games if g.get("player2_final_resources", 0) > g.get("player1_final_resources", 0))
            non_draws = p1_wins + p2_wins
            win_rate_p1 = p1_wins / non_draws if non_draws > 0 else 0.0

            def mean_std(lst):
                m = statistics.mean(lst) if lst else 0.0
                s = statistics.stdev(lst) if len(lst) > 1 else 0.0
                return m, s

            acc_m, acc_s = mean_std(accepts)
            tv_m, tv_s = mean_std(trade_volumes)
            p1_m, p1_s = mean_std(player1_payoffs)
            p2_m, p2_s = mean_std(player2_payoffs)

            behavior_metrics = {
                "behavior": behavior,
                "total_games": len(games),
                "acceptance_rate_mean": acc_m,
                "acceptance_rate_std": acc_s,
                "avg_trade_volume_mean": tv_m,
                "avg_trade_volume_std": tv_s,
                "avg_negotiation_rounds": statistics.mean(negotiation_rounds) if negotiation_rounds else 0.0,
                "negotiation_rounds_std": statistics.stdev(negotiation_rounds) if len(negotiation_rounds) > 1 else 0.0,
                "player1_payoff_mean": p1_m,
                "player1_payoff_std": p1_s,
                "player2_payoff_mean": p2_m,
                "player2_payoff_std": p2_s,
                "player1_wins": p1_wins,
                "player2_wins": p2_wins,
                "non_draws": non_draws,
                "win_rate_player1": win_rate_p1,
            }

            behavior_summary[behavior] = behavior_metrics

        self.behavior_summary = behavior_summary
        return behavior_summary

    def create_bar_plots_and_tables(self):
        """Create bar plots (mean ± std) for acceptance rate, trade volume, and payoffs.
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
        tv_means = [self.behavior_summary[b]["avg_trade_volume_mean"] for b in behaviors]
        tv_stds = [self.behavior_summary[b]["avg_trade_volume_std"] for b in behaviors]
        p1_means = [self.behavior_summary[b]["player1_payoff_mean"] for b in behaviors]
        p1_stds = [self.behavior_summary[b]["player1_payoff_std"] for b in behaviors]
        p2_means = [self.behavior_summary[b]["player2_payoff_mean"] for b in behaviors]
        p2_stds = [self.behavior_summary[b]["player2_payoff_std"] for b in behaviors]
        win_means = [self.behavior_summary[b]["win_rate_player1"] for b in behaviors]
        # std for win rate across games: approximate by computing per-behavior per-game binary win for P1
        win_stds = []
        for b in behaviors:
            games = [g for g in self.raw_data if g["behavior"] == b]
            wins = []
            for g in games:
                p1 = g.get("player1_final_resources", 0)
                p2 = g.get("player2_final_resources", 0)
                if p1 > p2:
                    wins.append(1)
                elif p2 > p1:
                    wins.append(0)
            win_stds.append(statistics.stdev(wins) if len(wins) > 1 else 0.0)

        # Styling similar to provided figure
        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        })

        colors = ["#4C4CB8", "#2E8BC0", "#18A162", "#6AD34F", "#D5E86B", "#B7E06C"]
        # pad colors to behaviors
        bar_colors = [colors[i % len(colors)] for i in range(len(behaviors))]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Acceptance rate
        ax = axes[0, 0]
        x = np.arange(len(behaviors))
        # no error bars on visualizations; stds remain in CSV/tables
        ax.bar(x, acc_means, color=bar_colors, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(behaviors, rotation=30, ha='right')
        ax.set_ylim(0, 1.05)
        ax.set_title('Average Acceptance Rate by Behavior')
        for i, v in enumerate(acc_means):
            ax.text(i, v + 0.02, f"{v*100:.2f}%", ha='center', fontsize=9)

        # Trade volume
        ax = axes[0, 1]
        ax.bar(x, tv_means, color=bar_colors, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(behaviors, rotation=30, ha='right')
        ax.set_title('Average Trade Volume by Behavior')
        for i, v in enumerate(tv_means):
            ax.text(i, v + (max(tv_means) * 0.02 if tv_means else 0.1), f"{v:.1f}", ha='center', fontsize=9)

        # Payoffs (grouped bars)
        ax = axes[1, 0]
        width = 0.35
        ax.bar(x - width/2, p1_means, width, label='Player 1', color='#2E86AB', edgecolor='black')
        ax.bar(x + width/2, p2_means, width, label='Player 2', color='#A23E48', edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(behaviors, rotation=30, ha='right')
        ax.set_title('Average Payoffs by Behavior')
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        for i in range(len(behaviors)):
            ax.text(i - width/2, p1_means[i] + (max(p1_means + p2_means) * 0.02 if p1_means or p2_means else 0.1), f"{p1_means[i]:.1f}", ha='center', fontsize=9)
            ax.text(i + width/2, p2_means[i] + (max(p1_means + p2_means) * 0.02 if p1_means or p2_means else 0.1), f"{p2_means[i]:.1f}", ha='center', fontsize=9)

        # Win rate (player1)
        ax = axes[1, 1]
        ax.bar(x, win_means, color=bar_colors, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(behaviors, rotation=30, ha='right')
        ax.set_ylim(0, 1.05)
        ax.set_title('Average Win Rate (Player 1) by Behavior')
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
                "avg_trade_volume_mean",
                "avg_trade_volume_std",
                "player1_payoff_mean",
                "player1_payoff_std",
                "player2_payoff_mean",
                "player2_payoff_std",
                "win_rate_player1",
            ]
            f.write(",".join(headers) + "\n"
            )
            for b in behaviors:
                m = self.behavior_summary[b]
                row = [
                    b,
                    str(m.get("total_games", 0)),
                    f"{m.get('acceptance_rate_mean', 0):.4f}",
                    f"{m.get('acceptance_rate_std', 0):.4f}",
                    f"{m.get('avg_trade_volume_mean', 0):.4f}",
                    f"{m.get('avg_trade_volume_std', 0):.4f}",
                    f"{m.get('player1_payoff_mean', 0):.4f}",
                    f"{m.get('player1_payoff_std', 0):.4f}",
                    f"{m.get('player2_payoff_mean', 0):.4f}",
                    f"{m.get('player2_payoff_std', 0):.4f}",
                    f"{m.get('win_rate_player1', 0):.4f}",
                ]
                f.write(",".join(row) + "\n")

        print(f"Saved behavior bar plot: {fig_file}")
        print(f"Saved behavior CSV table: {csv_file}")


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
                "acceptance_rate_mean",
                "acceptance_rate_std",
                "avg_trade_volume_mean",
                "avg_trade_volume_std",
                "avg_negotiation_rounds",
                "negotiation_rounds_std",
                "player1_payoff_mean",
                "player1_payoff_std",
                "player2_payoff_mean",
                "player2_payoff_std",
                "win_rate_player1",
            ]
            f.write(",".join(headers) + "\n")
            for behavior, metrics in self.behavior_summary.items():
                row = [
                    behavior,
                    str(metrics.get("total_games", 0)),
                    f"{metrics.get('acceptance_rate_mean', 0):.4f}",
                    f"{metrics.get('acceptance_rate_std', 0):.4f}",
                    f"{metrics.get('avg_trade_volume_mean', 0):.4f}",
                    f"{metrics.get('avg_trade_volume_std', 0):.4f}",
                    f"{metrics.get('avg_negotiation_rounds', 0):.4f}",
                    f"{metrics.get('negotiation_rounds_std', 0):.4f}",
                    f"{metrics.get('player1_payoff_mean', 0):.4f}",
                    f"{metrics.get('player1_payoff_std', 0):.4f}",
                    f"{metrics.get('player2_payoff_mean', 0):.4f}",
                    f"{metrics.get('player2_payoff_std', 0):.4f}",
                    f"{metrics.get('win_rate_player1', 0):.4f}",
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
                f.write(f"Acceptance Rate (mean): {metrics.get('acceptance_rate_mean', 0):.3f} ± {metrics.get('acceptance_rate_std', 0):.3f}\n")
                f.write(f"Player 1 Win Rate (excluding ties): {metrics.get('win_rate_player1', 0):.3f}\n")
                f.write(f"Average Trade Volume: {metrics.get('avg_trade_volume_mean', 0):.1f} ± {metrics.get('avg_trade_volume_std', 0):.1f}\n")
                f.write(f"Average Negotiation Rounds: {metrics.get('avg_negotiation_rounds', 0):.1f} ± {metrics.get('negotiation_rounds_std', 0):.1f}\n")
                f.write(f"Player 1 Average Payoff: {metrics.get('player1_payoff_mean', 0):.1f} ± {metrics.get('player1_payoff_std', 0):.1f}\n")
                f.write(f"Player 2 Average Payoff: {metrics.get('player2_payoff_mean', 0):.1f} ± {metrics.get('player2_payoff_std', 0):.1f}\n")
                f.write("\n")

            # Behavior-level aggregated metrics (averaged across all model combinations)
            f.write("\nBEHAVIOR-LEVEL AVERAGES\n")
            f.write("=" * 50 + "\n")
            for behavior, metrics in self.behavior_summary.items():
                f.write(f"{behavior}\n")
                f.write("-" * 40 + "\n")
                f.write(f"Total Games: {metrics.get('total_games', 0)}\n")
                f.write(f"Acceptance Rate (mean): {metrics.get('acceptance_rate_mean', 0):.3f} ± {metrics.get('acceptance_rate_std', 0):.3f}\n")
                f.write(f"Player 1 Win Rate (excluding ties): {metrics.get('win_rate_player1', 0):.3f}\n")
                f.write(f"Average Trade Volume: {metrics.get('avg_trade_volume_mean', 0):.1f} ± {metrics.get('avg_trade_volume_std', 0):.1f}\n")
                f.write(f"Average Negotiation Rounds: {metrics.get('avg_negotiation_rounds', 0):.1f} ± {metrics.get('negotiation_rounds_std', 0):.1f}\n")
                f.write(f"Player 1 Average Payoff: {metrics.get('player1_payoff_mean', 0):.1f} ± {metrics.get('player1_payoff_std', 0):.1f}\n")
                f.write(f"Player 2 Average Payoff: {metrics.get('player2_payoff_mean', 0):.1f} ± {metrics.get('player2_payoff_std', 0):.1f}\n")
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
                        # win rate per combo already excludes ties in calculate_metrics
                        win_rates[model1][model2] = metrics.get("win_rate_player1", None)
                        # Payoff heatmaps: average total resources after trade (mean)
                        payoff_p1[model1][model2] = metrics.get("player1_payoff_mean", None)
                        payoff_p2[model1][model2] = metrics.get("player2_payoff_mean", None)
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
        heatmap_file = self.results_dir / "all_heatmaps_trading_data.json"
        with open(heatmap_file, "w", encoding="utf-8") as f:
            json.dump(heatmap_data, f, indent=2)

        # Render all heatmaps in one big image
        plots_dir = Path(self.results_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        behaviors = list(heatmap_data.keys())
        num_behaviors = len(behaviors)
        num_types = 2  # payoff_player1, payoff_player2

        fig, axes = plt.subplots(nrows=num_behaviors, ncols=num_types, figsize=(12, 4 * num_behaviors))

        type_configs = [
            ("payoff_player1", "Average Payoff (Player 1)", lambda v: f"{v:.1f}", 'Blues', None, None),
            ("payoff_player2", "Average Payoff (Player 2)", lambda v: f"{v:.1f}", 'Blues', None, None),
        ]

        for b_idx, behavior in enumerate(behaviors):
            data = heatmap_data[behavior]
            models = data["models"]
            m = len(models)

            for t_idx, (heatmap_type, title_suffix, value_format, cmap_name, vmin, vmax) in enumerate(type_configs):
                ax = axes[b_idx, t_idx]

                matrix = np.full((m, m), np.nan, dtype=float)
                for i, a in enumerate(models):
                    for j, b in enumerate(models):
                        val = data[heatmap_type].get(a, {}).get(b, None)
                        if val is None:
                            matrix[i, j] = np.nan
                        else:
                            matrix[i, j] = float(val)

                masked_matrix = np.ma.masked_invalid(matrix)
                cmap = plt.cm.get_cmap(cmap_name)
                im = ax.imshow(masked_matrix, vmin=vmin, vmax=vmax, cmap=cmap)
                ax.set_xticks(range(m))
                ax.set_yticks(range(m))
                ax.set_xticklabels(models, rotation=45, ha='right')
                ax.set_yticklabels(models)

                # Set titles only for the top row and left column
                if b_idx == 0:
                    ax.set_title(title_suffix, fontsize=14, pad=6)
                    ax.set_ylabel(f'{behavior}\nModel 1', fontsize=12, labelpad=6)

                if t_idx == 0:
                    ax.set_ylabel(f'{behavior}\nModel 1', fontsize=12)

                # Add colorbar
                cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
                cbar.ax.set_ylabel('Average Payoff', rotation=270, labelpad=15)

                # Annotate heatmap cells
                for i in range(m):
                    for j in range(m):
                        val = matrix[i, j]
                        if i == j:
                            txt = "N/A"
                            txt_color = 'gray'
                            fontsize = 8
                        elif np.ma.is_masked(masked_matrix[i, j]):
                            txt = "-"
                            txt_color = 'gray'
                            fontsize = 8
                        else:
                            txt = value_format(val)
                            # Use white text for high values, black for low
                            if not np.isnan(matrix.max()) and val > matrix.max() * 0.7:
                                txt_color = 'white'
                            else:
                                txt_color = 'black'
                            fontsize = 10
                        ax.text(j, i, txt, ha='center', va='center', color=txt_color, fontsize=fontsize, fontweight='bold')

        plt.tight_layout(pad=0.8, w_pad=0.4, h_pad=0.6)

        all_heatmaps_file = plots_dir / 'all_heatmaps.png'
        fig.savefig(all_heatmaps_file, dpi=150)
        plt.close(fig)

        print(f"  Heatmap data JSON: {heatmap_file}")
        print(f"  All heatmaps image: {all_heatmaps_file}")

        return heatmap_data
    

    def welch_anova(self, groups_dict):
        """
        Perform Welch's ANOVA on grouped data (robust to unequal variances).
        
        Parameters:
            groups_dict (dict): {group_name: list_of_values}
        
        Returns:
            dict with F-statistic, p-value, df_num, df_den
        """
        data = []
        for group, values in groups_dict.items():
            for v in values:
                if v is not None and not np.isnan(v):
                    data.append({"group": group, "value": v})
        
        if len(data) == 0:
            return {"F": np.nan, "p_value": np.nan, "df_num": np.nan, "df_den": np.nan}
        
        df = pd.DataFrame(data)
        
        try:
            result = anova_oneway(
                df["value"],
                df["group"],
                use_var="unequal"  # Welch's ANOVA
            )
            
            # Handle tuple return format
            if hasattr(result, 'statistic'):
                f_stat = result.statistic
                p_val = result.pvalue
            else:
                f_stat = result[0]
                p_val = result[1]
            
            # Calculate degrees of freedom manually
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
        """
        Perform Welch's t-test (does not assume equal variances).
        """
        group1 = np.array([x for x in group1 if x is not None and not np.isnan(x)])
        group2 = np.array([x for x in group2 if x is not None and not np.isnan(x)])
        
        if len(group1) < 2 or len(group2) < 2:
            return {"t": np.nan, "p_value": np.nan, "df": np.nan}
        
        t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
        
        # Welch-Satterthwaite degrees of freedom
        n1, n2 = len(group1), len(group2)
        v1, v2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        if v1 == 0 and v2 == 0:
            df = n1 + n2 - 2
        else:
            df = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))
        
        return {"t": t_stat, "p_value": p_val, "df": df}


    def mann_whitney_test(self, group1, group2):
        """
        Perform Mann-Whitney U test (non-parametric alternative to t-test).
        Useful when distributions are highly skewed.
        """
        group1 = np.array([x for x in group1 if x is not None and not np.isnan(x)])
        group2 = np.array([x for x in group2 if x is not None and not np.isnan(x)])
        
        if len(group1) < 3 or len(group2) < 3:
            return {"U": np.nan, "p_value": np.nan, "effect_size": np.nan}
        
        try:
            u_stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
            
            # Calculate rank-biserial correlation (effect size)
            n1, n2 = len(group1), len(group2)
            r = 1 - (2*u_stat) / (n1 * n2)
            
            return {"U": u_stat, "p_value": p_val, "effect_size": r}
        except Exception as e:
            print(f"Error in Mann-Whitney: {e}")
            return {"U": np.nan, "p_value": np.nan, "effect_size": np.nan}


    def hedges_g(self, group1, group2):
        """
        Calculate Hedges' g effect size (corrected for small samples).
        """
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
        """
        Calculate omega-squared effect size (less biased than eta-squared).
        """
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


    def kruskal_wallis_test(self, groups_dict):
        """
        Perform Kruskal-Wallis H-test (non-parametric alternative to ANOVA).
        Useful when distributions are heavily skewed or have outliers.
        """
        groups = [np.array([x for x in v if x is not None and not np.isnan(x)]) 
                for v in groups_dict.values()]
        
        # Filter out empty groups
        groups = [g for g in groups if len(g) >= 3]
        
        if len(groups) < 2:
            return {"H": np.nan, "p_value": np.nan, "df": np.nan}
        
        try:
            h_stat, p_val = stats.kruskal(*groups)
            df = len(groups) - 1
            
            return {"H": h_stat, "p_value": p_val, "df": df}
        except Exception as e:
            print(f"Error in Kruskal-Wallis: {e}")
            return {"H": np.nan, "p_value": np.nan, "df": np.nan}


    def run_comprehensive_statistical_analysis(self):
        """
        Perform comprehensive behavior-level statistical analysis for Trading Game.
        
        Key differences from Buy-Sell:
        1. Trade Volume is continuous but often zero-inflated (rejected trades)
        2. Payoffs may be highly skewed (outliers from extreme trades)
        3. Need both parametric (Welch) and non-parametric (Mann-Whitney) tests
        """
        out_dir = Path(self.results_dir) / "stats"
        out_dir.mkdir(exist_ok=True)
        
        # Redirect all print output to a log file
        log_file = out_dir / "statistical_analysis_log.txt"
        original_stdout = sys.stdout
        
        with open(log_file, 'w', encoding='utf-8') as log_f:
            sys.stdout = log_f
            
            df = pd.DataFrame(self.raw_data)
            # Only analyze accepted trades for most metrics
            df_accepted = df[df["outcome"] == "ACCEPT"]
            
            print("="*80)
            print("COMPREHENSIVE STATISTICAL ANALYSIS - TRADING/RESOURCE EXCHANGE GAME")
            print("="*80)
            print(f"\nTotal games: {len(df)}")
            print(f"Accepted trades: {len(df_accepted)}")
            print(f"Rejected trades: {len(df[df['outcome'] == 'REJECT'])}")
            
            results = []
            
            # ===================================================================
            # CONTINUOUS METRICS - PARAMETRIC TESTS (WELCH'S ANOVA)
            # ===================================================================
            
            continuous_metrics = {
                "trade_volume": ("Trade Volume", df_accepted),
                "player1_final_resources": ("Player 1 Payoff", df_accepted),
                "player2_final_resources": ("Player 2 Payoff", df_accepted),
                "negotiation_rounds": ("Negotiation Rounds", df),  # All games
            }
            
            for metric, (label, data_subset) in continuous_metrics.items():
                print(f"\n{'='*70}")
                print(f"METRIC: {label}")
                print(f"{'='*70}")
                
                # Gather data by behavior
                groups_dict = {}
                behaviors = []
                
                for b in sorted(data_subset["behavior"].unique()):
                    vals = data_subset.loc[
                        (data_subset["behavior"] == b) & data_subset[metric].notna(), 
                        metric
                    ].values
                    
                    if len(vals) >= 5:  # Minimum threshold
                        groups_dict[b] = vals
                        behaviors.append(b)
                        print(f"  {b}: n={len(vals)}, mean={np.mean(vals):.2f}, "
                            f"std={np.std(vals, ddof=1):.2f}, "
                            f"median={np.median(vals):.2f}")
                
                if len(groups_dict) < 2:
                    print(f"  Skipping {label} - insufficient groups")
                    continue
                
                groups = list(groups_dict.values())
                
                # 1. Test for homogeneity of variance
                lev_stat, lev_p = levene(*groups)
                print(f"\n  Levene's Test: W={lev_stat:.4f}, p={lev_p:.4f}")
                
                if lev_p < 0.05:
                    print("  → Variances are UNEQUAL (p < 0.05)")
                else:
                    print("  → Variances are equal (p ≥ 0.05)")
                
                # 2. Check for normality and skewness
                skewness_values = {b: stats.skew(groups_dict[b]) for b in behaviors}
                print(f"\n  Skewness by behavior:")
                for b in behaviors:
                    print(f"    {b}: {skewness_values[b]:.3f}")
                
                max_skew = max(abs(s) for s in skewness_values.values())
                heavily_skewed = max_skew > 1.0
                
                if heavily_skewed:
                    print(f"  → HEAVILY SKEWED (max |skew| = {max_skew:.3f})")
                    print(f"  → Will use non-parametric tests in addition to parametric")
                
                # 3a. PARAMETRIC: Welch's ANOVA
                welch_result = self.welch_anova(groups_dict)
                print(f"\n  Welch's ANOVA (Parametric):")
                print(f"    F({welch_result['df_num']:.2f}, {welch_result['df_den']:.2f}) = {welch_result['F']:.4f}")
                print(f"    p-value = {welch_result['p_value']:.4e}")
                
                omega2 = self.omega_squared(groups)
                print(f"    Omega² = {omega2:.4f}")
                
                results.append({
                    "metric": label,
                    "test": "Welch_ANOVA",
                    "F": welch_result['F'],
                    "df_num": welch_result['df_num'],
                    "df_den": welch_result['df_den'],
                    "p_value": welch_result['p_value'],
                    "omega_squared": omega2,
                    "levene_W": lev_stat,
                    "levene_p": lev_p,
                    "max_skewness": max_skew,
                    "significant": welch_result['p_value'] < 0.05
                })
                
                # 3b. NON-PARAMETRIC: Kruskal-Wallis (if skewed)
                if heavily_skewed:
                    kw_result = self.kruskal_wallis_test(groups_dict)
                    print(f"\n  Kruskal-Wallis H-test (Non-parametric):")
                    print(f"    H({kw_result['df']:.0f}) = {kw_result['H']:.4f}")
                    print(f"    p-value = {kw_result['p_value']:.4e}")
                    
                    results.append({
                        "metric": label,
                        "test": "Kruskal_Wallis",
                        "H": kw_result['H'],
                        "df": kw_result['df'],
                        "p_value": kw_result['p_value'],
                        "significant": kw_result['p_value'] < 0.05
                    })
                
                # 4. PAIRWISE COMPARISONS (if overall effect significant)
                if welch_result['p_value'] < 0.05 or (heavily_skewed and kw_result['p_value'] < 0.05):
                    print(f"\n  Pairwise Comparisons:")
                    
                    pairwise_results = []
                    
                    for b1, b2 in combinations(behaviors, 2):
                        g1, g2 = groups_dict[b1], groups_dict[b2]
                        
                        # Parametric: Welch's t-test
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
                        
                        # Non-parametric: Mann-Whitney (if skewed)
                        if heavily_skewed:
                            mw_result = self.mann_whitney_test(g1, g2)
                            results.append({
                                "metric": label,
                                "test": "Mann_Whitney_U",
                                "comparison": f"{b1} vs {b2}",
                                "U": mw_result['U'],
                                "p_value": mw_result['p_value'],
                                "rank_biserial": mw_result['effect_size']
                            })
                    
                    # Bonferroni correction
                    p_values = [r['p_value'] for r in pairwise_results]
                    _, p_corrected, _, _ = multipletests(p_values, method='bonferroni')
                    
                    print(f"\n    {'Comparison':<30} {'Mean Diff':<12} {'Hedges g':<10} {'p':<10} {'p_corr':<10} {'Sig'}")
                    print(f"    {'-'*85}")
                    
                    idx = 0
                    for b1, b2 in combinations(behaviors, 2):
                        g1, g2 = groups_dict[b1], groups_dict[b2]
                        mean_diff = np.mean(g1) - np.mean(g2)
                        hedges = self.hedges_g(g1, g2)
                        
                        sig = "***" if p_corrected[idx] < 0.001 else "**" if p_corrected[idx] < 0.01 else "*" if p_corrected[idx] < 0.05 else "ns"
                        
                        print(f"    {f'{b1} vs {b2}':<30} {mean_diff:>11.2f} {hedges:>9.3f} {p_values[idx]:>9.4f} {p_corrected[idx]:>9.4f} {sig:>3}")
                        idx += 1
            
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
            
            # Check for degenerate cases (all accept or all reject)
            degenerate = any(row[0] == 0 or row[1] == 0 for row in contingency)
            
            if degenerate:
                print("\n  ⚠ WARNING: At least one behavior has perfect acceptance/rejection")
                print("  Chi-square test may be unreliable or undefined")
                
                results.append({
                    "metric": "Acceptance Rate",
                    "test": "Chi_square",
                    "note": "Degenerate case - perfect separation in at least one group",
                    "chi2": np.nan,
                    "p_value": np.nan
                })
            else:
                # Chi-square test
                chi2, p_chi, dof, expected = stats.chi2_contingency(contingency)
                print(f"\n  Chi-square test: χ²({dof}) = {chi2:.4f}, p = {p_chi:.4f}")
                
                results.append({
                    "metric": "Acceptance Rate",
                    "test": "Chi_square",
                    "chi2": chi2,
                    "df": dof,
                    "p_value": p_chi,
                    "significant": p_chi < 0.05
                })
                
                # Pairwise proportion tests (if significant)
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
                    
                    print(f"\n    {'Comparison':<30} {'Rate Diff':<12} {'p':<10} {'p_corr':<10} {'Sig'}")
                    print(f"    {'-'*75}")
                    
                    idx = 0
                    for i, b1 in enumerate(behaviors):
                        for j, b2 in enumerate(behaviors):
                            if i >= j:
                                continue
                            
                            rate1 = contingency[i, 0] / contingency[i].sum()
                            rate2 = contingency[j, 0] / contingency[j].sum()
                            diff = rate1 - rate2
                            
                            sig = "***" if p_corrected[idx] < 0.001 else "**" if p_corrected[idx] < 0.01 else "*" if p_corrected[idx] < 0.05 else "ns"
                            
                            print(f"    {f'{b1} vs {b2}':<30} {diff:>11.3f} {pairwise_p[idx]:>9.4f} {p_corrected[idx]:>9.4f} {sig:>3}")
                            idx += 1
            
            # ===================================================================
            # SAVE RESULTS
            # ===================================================================
            
            print(f"\n{'='*80}")
            print("SAVING RESULTS")
            print(f"{'='*80}")
            
            results_df = pd.DataFrame(results)
            
            csv_file = out_dir / "statistical_tests_trading.csv"
            results_df.to_csv(csv_file, index=False)
            print(f"  Saved: {csv_file}")
            
            json_file = out_dir / "statistical_tests_trading.json"
            results_df.to_json(json_file, orient='records', indent=2)
            print(f"  Saved: {json_file}")
            
            # Create summary table
            summary_file = out_dir / "statistical_summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as sf:
                sf.write("="*80 + "\n")
                sf.write("STATISTICAL ANALYSIS SUMMARY - TRADING GAME\n")
                sf.write("="*80 + "\n\n")
                
                # Overall tests
                sf.write("OVERALL TESTS (Welch's ANOVA / Kruskal-Wallis)\n")
                sf.write("-"*80 + "\n")
                overall_tests = results_df[results_df['test'].isin(['Welch_ANOVA', 'Kruskal_Wallis', 'Chi_square'])]
                for _, row in overall_tests.iterrows():
                    sf.write(f"\n{row['metric']} ({row['test']}):\n")
                    if row['test'] == 'Welch_ANOVA':
                        sf.write(f"  F({row['df_num']:.2f}, {row['df_den']:.2f}) = {row['F']:.4f}, p = {row['p_value']:.4e}\n")
                        sf.write(f"  Effect size (omega-squared) = {row['omega_squared']:.4f}\n")
                    elif row['test'] == 'Kruskal_Wallis':
                        sf.write(f"  H({row['df']:.0f}) = {row['H']:.4f}, p = {row['p_value']:.4e}\n")
                    elif row['test'] == 'Chi_square':
                        sf.write(f"  chi-squared({row.get('df', 'NA')}) = {row.get('chi2', 'NA')}, p = {row.get('p_value', 'NA')}\n")
                    sf.write(f"  Significant: {'YES' if row.get('significant', False) else 'NO'}\n")
                
                # Significant pairwise comparisons
                sf.write("\n\nSIGNIFICANT PAIRWISE COMPARISONS (p < 0.05 after correction)\n")
                sf.write("-"*80 + "\n")
                pairwise = results_df[results_df['test'].isin(['Welch_t_test', 'Mann_Whitney_U', 'Proportion_test'])]
                
                # You'll need to recalculate corrected p-values here or store them
                # For now, we'll just show raw significant ones
                for metric in pairwise['metric'].unique():
                    metric_tests = pairwise[pairwise['metric'] == metric]
                    sig_tests = metric_tests[metric_tests['p_value'] < 0.05/len(metric_tests)]  # Rough Bonferroni
                    
                    if len(sig_tests) > 0:
                        sf.write(f"\n{metric}:\n")
                        for _, row in sig_tests.iterrows():
                            sf.write(f"  {row['comparison']}: ")
                            if row['test'] == 'Welch_t_test':
                                sf.write(f"Hedges' g = {row.get('hedges_g', 'NA'):.3f}, p = {row['p_value']:.4f}\n")
                            elif row['test'] == 'Mann_Whitney_U':
                                sf.write(f"r = {row.get('rank_biserial', 'NA'):.3f}, p = {row['p_value']:.4f}\n")
            
            print(f"  Saved: {summary_file}")
            
            print(f"\n{'='*80}")
            print("STATISTICAL ANALYSIS COMPLETE")
            print(f"{'='*80}\n")
        
        # Restore stdout
        sys.stdout = original_stdout
        
        print(f"\nStatistical analysis complete. Results saved to: {out_dir}")
        print(f"  - Full log: {log_file}")
        print(f"  - CSV results: {out_dir / 'statistical_tests_trading.csv'}")
        print(f"  - JSON results: {out_dir / 'statistical_tests_trading.json'}")
        print(f"  - Summary: {out_dir / 'statistical_summary.txt'}")
        
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

    analyzer = TradingResultsAnalyzer(results_dir)

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

    # Save results (JSON/CSV/readable)
    analyzer.save_results()

    # Create heatmap data (and images) for win rates
    analyzer.create_heatmap_data()

    print(f"\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"Processed {len(raw_data)} games")
    print(f"Generated {len(summary)} combination summaries")
    print(f"Check the generated files in: {results_dir}")

    # Run comprehensive statistical analysis
    print("\n" + "="*60)
    print("Running Statistical Analysis...")
    print("="*60)
    analyzer.run_comprehensive_statistical_analysis()


if __name__ == "__main__":
    main()