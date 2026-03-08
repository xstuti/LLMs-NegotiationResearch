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
from matplotlib.patches import Rectangle
from scipy.stats import f_oneway, levene, chi2_contingency, mannwhitneyu
from itertools import combinations
import pandas as pd
from scipy.stats import chi2_contingency
import statsmodels.api as sm
from statsmodels.stats.oneway import anova_oneway
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import levene
from itertools import combinations
import statsmodels.api as sm
from statsmodels.stats.oneway import anova_oneway
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

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

    def find_game_directories(self):
        """Find all game directories with the expected naming pattern recursively"""
        game_dirs = []

        for item in self.results_dir.rglob("*"):
            if not item.is_dir():
                continue
            parts = item.name.split("_")
            if len(parts) < 4:
                continue
            # last part should look like 'iter<num>' or 'iter_<num>' or 'iter' '<num>'
            last = parts[-1]
            iter_num = None
            if last.startswith("iter"):
                suffix = last[len("iter") :].lstrip("_")
                if suffix.isdigit() and 1 <= int(suffix) <= 30:
                    iter_num = int(suffix)
            elif len(parts) >= 5 and parts[-2] == "iter" and parts[-1].isdigit() and 1 <= int(parts[-1]) <= 30:
                iter_num = int(parts[-1])
            if iter_num is None:
                continue
            # basic shape: model1_model2_language_iter<num>
            game_dirs.append(item)

        print(f"Found {len(game_dirs)} game directories")
        return sorted(game_dirs)

    def parse_directory_name(self, dir_name):
        """Parse directory name to extract seller_model, buyer_model, behavior (language), and iteration.

        Expected pattern: seller_model_buyer_model_behavior_iter<number>
        Example: GPT-3.5_GPT-4o_English_iter_7 or Claude-3.5-Haiku_Claude-3.5-Haiku_English_iter_1
        """
        print(f"Parsing: {dir_name}")

        try:
            parts = dir_name.split("_")
            if len(parts) < 4:
                raise ValueError("Not enough segments for expected pattern")

            # Determine iteration and adjust parts
            iteration = None
            behavior_idx = -2
            if parts[-1].startswith("iter"):
                iter_suffix = parts[-1][len("iter") :].lstrip("_")
                iteration = int(iter_suffix)
            elif len(parts) >= 5 and parts[-2] == "iter" and parts[-1].isdigit():
                iteration = int(parts[-1])
                behavior_idx = -3
            else:
                raise ValueError("Invalid iteration format")

            # Behavior/language
            behavior_candidate = parts[behavior_idx]
            behavior = behavior_candidate
            for known_behavior in self.KNOWN_BEHAVIORS:
                if behavior_candidate == known_behavior:
                    behavior = known_behavior
                    break

            # Remaining parts before behavior = seller_model + buyer_model
            model_parts = parts[:behavior_idx]  # everything except behavior and iteration parts
            if len(model_parts) < 2:
                raise ValueError("Need at least two segments for seller_model and buyer_model")

            # Try to match seller_model using known models (greedy on prefix)
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

            seller_model = "_".join(model_parts[:best_idx])
            buyer_model = "_".join(model_parts[best_idx:])

            # Normalize to known canonical names if possible
            for known_model in self.KNOWN_MODELS:
                if normalize(seller_model) == normalize(known_model):
                    seller_model = known_model
                if normalize(buyer_model) == normalize(known_model):
                    buyer_model = known_model

            print(f"  Parsed: {seller_model} vs {buyer_model} | {behavior} | iter {iteration}")
            return seller_model, buyer_model, behavior, iteration

        except Exception as e:
            print(f"  Error parsing {dir_name}: {e}")
            return "Unknown", "Unknown", "Unknown", 1

    def find_game_state_file(self, game_dir):
        """Find the game_state.json file in the game directory"""
        # Look for subdirectories (random directories)
        for subdir in game_dir.iterdir():
            if subdir.is_dir():
                game_state_file = subdir / "game_state.json"
                if game_state_file.exists():
                    return game_state_file
        return None

    def extract_game_data(self, game_state_file, seller_model, buyer_model, behavior, iteration):
        """Extract relevant data from a single game_state.json file for buy-sell game."""
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

            # Detect whether any interaction (proposal/accept/reject) occurred at all.
            # We'll scan both state-level public info and player conversations for player answers.
            any_interaction = False
            interaction_reason = None

            # Check game_state entries for public info with player answer
            for state in game_state:
                pub = state.get("player_public_info_dict") or {}
                if isinstance(pub, dict):
                    pa = pub.get("player answer") or pub.get("player_answer")
                    if isinstance(pa, str) and pa.strip():
                        any_interaction = True
                        interaction_reason = f"Found player_public_info_dict with answer={pa}"
                        break
                # some logs embed a combined player_public_answer_string
                pstr = state.get("player_public_answer_string")
                if isinstance(pstr, str) and ("<PROPOSAL>" in pstr.upper() or "<ACCEPT>" in pstr.upper() or "<REJECT>" in pstr.upper()):
                    any_interaction = True
                    interaction_reason = "Found player_public_answer_string containing tags"
                    break

            # If still none, scan players' conversation for any <player answer> tags (PROPOSAL/ACCEPT/REJECT)
            if not any_interaction:
                import re as _re
                tag_any_re = _re.compile(r"<\s*player\s+answer\s*>\s*([A-Za-z]+)\s*<", _re.IGNORECASE)
                for p in players:
                    conv = p.get("conversation", [])
                    for msg in conv:
                        content = msg.get("content") if isinstance(msg, dict) else (msg if isinstance(msg, str) else "")
                        if not content:
                            continue
                        m = tag_any_re.search(content)
                        if m:
                            any_interaction = True
                            interaction_reason = f"Found conversation tag with answer={m.group(1)}"
                            break
                    if any_interaction:
                        break

            no_interaction = not any_interaction
            no_interaction_reason = None if any_interaction else "no proposals or player answers found"

            # Determine valuations (seller/buyer) from multiple possible places
            seller_valuation = 40
            buyer_valuation = 60

            # 1) try settings.player_valuation in start_state
            if start_state:
                settings = start_state.get("settings", {})
                player_valuation = settings.get("player_valuation")
                if isinstance(player_valuation, list) and len(player_valuation) >= 2:
                    if isinstance(player_valuation[0], (int, float)):
                        seller_valuation = player_valuation[0]
                    if isinstance(player_valuation[1], (int, float)):
                        buyer_valuation = player_valuation[1]

            # 2) fallback: try player_goals in end_state or start_state
            if (seller_valuation, buyer_valuation) == (40, 60):
                goals_source = end_state or start_state
                if goals_source:
                    goals = goals_source.get("player_goals", [])
                    if isinstance(goals, list) and len(goals) >= 2:
                        try:
                            seller_goal = goals[0].get("_value", {}).get("_value", {}).get("_value", {})
                            if isinstance(seller_goal, dict) and "X" in seller_goal and isinstance(seller_goal["X"], (int, float)):
                                seller_valuation = seller_goal["X"]
                        except Exception:
                            pass
                        try:
                            buyer_goal = goals[1].get("_value", {}).get("_value", {}).get("_value", {})
                            if isinstance(buyer_goal, dict) and "X" in buyer_goal and isinstance(buyer_goal["X"], (int, float)):
                                buyer_valuation = buyer_goal["X"]
                        except Exception:
                            pass

            # Basic summary section: prefer end_state.summary, then top-level summary
            summary = {}
            if end_state and isinstance(end_state, dict):
                summary = end_state.get("summary", {}) or {}
            summary = summary or data.get("summary", {}) or {}

            # Extract proposed_trade robustly: summary.proposed_trade may be dict or a string
            proposed_trade = summary.get("proposed_trade")

            # If no structured proposed_trade, try scanning states/player public info for last proposed trade string
            if not proposed_trade:
                # look for recently proposed trade in reversed game_state
                for state in reversed(game_state):
                    # try known response keys
                    for player_key in ["player1_response", "player2_response", "player_public_info_dict"]:
                        resp = state.get(player_key, {})
                        pub_info = resp.get("player_public_info_dict", resp) if isinstance(resp, dict) else {}
                        trade = pub_info.get("newly proposed trade") if isinstance(pub_info, dict) else None
                        if trade and trade != "NONE":
                            proposed_trade = trade
                            break
                    if proposed_trade:
                        break

            # Helper: attempt to extract buyer ZUP price from different formats
            def extract_zup_price(obj):
                """Return int price if found, otherwise None."""
                import re

                # If structured dict following _type/_value pattern
                if isinstance(obj, dict):
                    # typical structure: {"_type":"trade","_value":{"RED":{...},"BLUE":{"_value":{"ZUP":90}}}}
                    try:
                        tv = obj.get("_value", {})
                        blue = tv.get("BLUE", {})
                        if isinstance(blue, dict):
                            bval = blue.get("_value", {})
                            if isinstance(bval, dict) and "ZUP" in bval and isinstance(bval["ZUP"], (int, float)):
                                return int(bval["ZUP"])
                    except Exception:
                        pass
                    # legacy keys
                    try:
                        # sometimes keys are uppercase names
                        blue = obj.get("BLUE") or obj.get("blue")
                        if isinstance(blue, dict):
                            bval = blue.get("_value", blue.get("value", {}))
                            if isinstance(bval, dict) and "ZUP" in bval and isinstance(bval["ZUP"], (int, float)):
                                return int(bval["ZUP"])
                    except Exception:
                        pass

                # If it's a string, use regex
                if isinstance(obj, str):
                    m = re.search(r"ZUP\s*[:]*\s*(\d{1,6})", obj, re.IGNORECASE)
                    if m:
                        try:
                            return int(m.group(1))
                        except Exception:
                            return None

                return None

            # Negotiation rounds: count numeric iterations
            negotiation_rounds = 0
            try:
                numeric_iterations = []
                for state in game_state:
                    it = state.get("current_iteration")
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

            # Determine trade_price via structured proposed_trade or extracted string
            trade_price = None
            if proposed_trade is not None:
                trade_price = extract_zup_price(proposed_trade)

            # If still None, try from summary final proposed trade structures
            if trade_price is None and isinstance(summary, dict):
                trade_price = extract_zup_price(summary.get("proposed_trade"))

            # Final response
            final_response = summary.get("final_response") if isinstance(summary, dict) else None
            if not final_response:
                if accepted:
                    final_response = "ACCEPT"
                else:
                    final_response = "UNKNOWN"

            # Trade occurred if accepted or final_response == ACCEPT
            trade_occurred = (str(final_response).upper() == "ACCEPT") or accepted

            # Advantages only for accepted trades
            seller_advantage = None
            buyer_advantage = None
            if trade_occurred and trade_price is not None and isinstance(seller_valuation, (int, float)) and isinstance(buyer_valuation, (int, float)):
                seller_advantage = trade_price - seller_valuation
                buyer_advantage = buyer_valuation - trade_price

            game_data = {
                "seller_model": seller_model,
                "buyer_model": buyer_model,
                "behavior": behavior,
                "iteration": iteration,
                # game_completed: True if there is an END summary or an actual accepted response
                "game_completed": (end_state is not None or accepted) and not no_interaction,
                "trade_occurred": trade_occurred,
                "negotiation_rounds": negotiation_rounds,
                "seller_advantage": seller_advantage,
                "buyer_advantage": buyer_advantage,
                "trade_price": trade_price,
                "seller_valuation": seller_valuation,
                "buyer_valuation": buyer_valuation,
                "final_response": final_response,
                "no_interaction": no_interaction,
                "no_interaction_reason": no_interaction_reason,
                "file_path": str(game_state_file),
            }

            return game_data

        except Exception as e:
            print(f"Error processing {game_state_file}: {str(e)}")
            return None

    def analyze_all_games(self):
        """Analyze all games by parsing game_state.json files"""
        game_dirs = self.find_game_directories()

        for game_dir in game_dirs:
            try:
                seller_model, buyer_model, behavior, iteration = self.parse_directory_name(game_dir.name)
                game_state_file = self.find_game_state_file(game_dir)

                if game_state_file:
                    game_data = self.extract_game_data(game_state_file, seller_model, buyer_model, behavior, iteration)
                    if game_data:
                        self.raw_data.append(game_data)
                else:
                    print(f"Warning: No game_state.json found in {game_dir}")

            except Exception as e:
                print(f"Error processing {game_dir}: {str(e)}")

        print(f"\nTotal games processed: {len(self.raw_data)}")
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

            # Filter valid games (exclude errors) - games that completed
            valid_games = [g for g in games if g.get("game_completed", False)]

            if not valid_games:
                print(f"Warning: No valid games found for {combo_key}")
                continue

            # Accepted games for metrics that should only include successful trades
            accepted_games = [g for g in valid_games if g.get("trade_occurred", False)]

            def has_valid_advantages(g):
                return (
                    isinstance(g.get("seller_advantage"), (int, float)) and
                    isinstance(g.get("buyer_advantage"), (int, float))
                )

            accepted_games_with_adv = [g for g in accepted_games if has_valid_advantages(g)]


            # Calculate requested metrics only
            total_games = len(valid_games)
            accepts = [1 if g.get("trade_occurred", False) else 0 for g in valid_games]

            # Negotiation rounds (keep for all valid games)
            negotiation_rounds = [g.get("negotiation_rounds", 0) for g in valid_games]

            # Seller and buyer advantages (only from accepted trades)
            seller_advantages = [g.get("seller_advantage", 0) for g in accepted_games_with_adv]
            buyer_advantages = [g.get("buyer_advantage", 0) for g in accepted_games_with_adv]

            # Win counts for seller (player 1), only for accepted trades
            p1_wins = 0
            p2_wins = 0
            for g in accepted_games_with_adv:
                seller_adv = g["seller_advantage"]
                buyer_adv = g["buyer_advantage"]
                if seller_adv > buyer_adv:
                    p1_wins += 1
                elif buyer_adv > seller_adv:
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
            valid_adv_games = [
                g for g in accepted_games
                if isinstance(g.get("seller_advantage"), (int, float))
                and isinstance(g.get("buyer_advantage"), (int, float))
            ]

            seller_advantages = [g["seller_advantage"] for g in valid_adv_games]
            buyer_advantages  = [g["buyer_advantage"] for g in valid_adv_games]

            # Wins for seller
            valid_adv_games = [
                g for g in accepted_games
                if isinstance(g.get("seller_advantage"), (int, float))
                and isinstance(g.get("buyer_advantage"), (int, float))
            ]

            p1_wins = sum(1 for g in valid_adv_games if g["seller_advantage"] > g["buyer_advantage"])
            p2_wins = sum(1 for g in valid_adv_games if g["buyer_advantage"] > g["seller_advantage"])

            non_draws = p1_wins + p2_wins
            win_rate_p1 = p1_wins / non_draws if non_draws > 0 else 0.0
            draws = len(valid_adv_games) - non_draws
            draw_rate = draws / len(valid_adv_games) if valid_adv_games else 0.0


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
                "draw_rate": draw_rate,
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
                "draw_rate",
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
                    f"{metrics['draw_rate']:.4f}",
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
                "draw_rate",
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
                    f"{metrics['draw_rate']:.4f}",
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

        # Render combined heatmap image for all behaviors
        plots_dir = Path(self.results_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        behaviors_list = list(heatmap_data.keys())
        n_behaviors = len(behaviors_list)
        if n_behaviors == 0:
            print("No heatmap data to plot")
            return heatmap_data

        # Get models from first behavior (assuming same for all)
        models = heatmap_data[behaviors_list[0]]["models"]
        m = len(models)

        # Combined heatmap for all behaviors: n_behaviors rows, 2 columns (seller, buyer)
        fig, axes = plt.subplots(n_behaviors, 2, figsize=(12, 4 * n_behaviors))

        for row_idx, behavior in enumerate(behaviors_list):
            data = heatmap_data[behavior]

            for col_idx, (heatmap_type, title_suffix, value_format, cmap_name) in enumerate([
                ("seller_advantages", f"Seller Advantage — {behavior}", lambda v: f"{v:.1f}", 'Blues'),
                ("buyer_advantages", f"Buyer Advantage — {behavior}", lambda v: f"{v:.1f}", 'Oranges'),
            ]):
                ax = axes[row_idx, col_idx]
                matrix = np.full((m, m), np.nan, dtype=float)
                for i, a in enumerate(models):
                    for j, b in enumerate(models):
                        val = data[heatmap_type].get(a, {}).get(b, None)
                        if val is None:
                            matrix[i, j] = np.nan
                        else:
                            matrix[i, j] = float(val)

                # Create masked array for missing data
                masked = np.ma.masked_invalid(matrix)

                # Create custom colormap
                cmap = plt.cm.get_cmap(cmap_name)

                # Get valid range for normalization
                valid_vals = matrix[~np.isnan(matrix)]
                if len(valid_vals) > 0:
                    vmin, vmax = valid_vals.min(), valid_vals.max()
                else:
                    vmin, vmax = 0, 10

                im = ax.imshow(masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")

                # Add text annotations
                for i in range(m):
                    for j in range(m):
                        val = matrix[i, j]
                        if i == j:  # diagonal
                            # Gray out diagonal
                            rect = Rectangle(
                                (j - 0.5, i - 0.5),
                                1,
                                1,
                                facecolor="lightgray",
                                alpha=0.3,
                                edgecolor="gray",
                            )
                            ax.add_patch(rect)
                            ax.text(
                                j,
                                i,
                                "N/A",
                                ha="center",
                                va="center",
                                color="gray",
                                fontweight="bold",
                                fontsize=10,
                            )
                        elif np.isnan(val):
                            ax.text(
                                j,
                                i,
                                "-",
                                ha="center",
                                va="center",
                                color="gray",
                                fontweight="bold",
                                fontsize=10,
                            )
                        else:
                            # Normalize for text color
                            norm_value = (val - vmin) / (vmax - vmin) if vmax > vmin else 0
                            text_color = "white" if norm_value > 0.6 else "black"
                            ax.text(
                                j,
                                i,
                                value_format(val),
                                ha="center",
                                va="center",
                                color=text_color,
                                fontweight="bold",
                                fontsize=11,
                            )

                ax.set_title(title_suffix, fontweight="bold", pad=15)
                ax.set_xlabel("Player 2", fontweight="bold")
                ax.set_ylabel("Player 1", fontweight="bold")
                ax.set_xticks(range(m))
                ax.set_yticks(range(m))
                ax.set_xticklabels(models, rotation=45, ha="right")
                ax.set_yticklabels(models)

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
                cbar.ax.set_ylabel('Average Advantage', rotation=270, labelpad=15)

        heatmap_file_img = plots_dir / 'all_heatmaps_buysell.png'
        fig.tight_layout()
        fig.savefig(heatmap_file_img, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        print(f"  Heatmap data JSON: {heatmap_file}")
        print(f"  Combined heatmap image saved: {plots_dir / 'all_heatmaps_buysell.png'}")

        return heatmap_data

    #!/usr/bin/env python3
    """
    Enhanced statistical analysis methods for buy-sell game with Welch's tests
    Add these methods to your BuySellResultsAnalyzer class
    """

    #!/usr/bin/env python3
    """
    Enhanced statistical analysis methods for buy-sell game with Welch's tests
    Add these methods to your BuySellResultsAnalyzer class
    """
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
                use_var="unequal"  # This makes it Welch's ANOVA
            )
            
            # Handle different return formats from statsmodels
            # result could be a tuple or an object with attributes
            if hasattr(result, 'statistic'):
                f_stat = result.statistic
                p_val = result.pvalue
            else:
                # It's a tuple: (statistic, pvalue)
                f_stat = result[0]
                p_val = result[1]
            
            # Calculate degrees of freedom manually for Welch's ANOVA
            k = len(groups_dict)  # number of groups
            df_num = k - 1
            
            # Welch-Satterthwaite approximation for denominator df
            groups_list = list(groups_dict.values())
            ns = [len(g) for g in groups_list]
            vars = [np.var(g, ddof=1) if len(g) > 1 else 0 for g in groups_list]
            
            # Calculate denominator df using Welch-Satterthwaite formula
            numerator = sum([(1 - n_i/sum(ns)) * var_i for n_i, var_i in zip(ns, vars)])**2
            denominator = sum([((1 - n_i/sum(ns))**2 * var_i**2) / (n_i - 1) for n_i, var_i in zip(ns, vars)])
            
            if denominator > 0:
                df_den = numerator / denominator
            else:
                df_den = sum(ns) - k  # fallback to regular ANOVA df
            
            return {
                "F": float(f_stat),
                "p_value": float(p_val),
                "df_num": float(df_num),
                "df_den": float(df_den)
            }
        except Exception as e:
            print(f"Error in Welch ANOVA: {e}")
            import traceback
            traceback.print_exc()
            return {"F": np.nan, "p_value": np.nan, "df_num": np.nan, "df_den": np.nan}


    def welch_ttest(self, group1, group2):
        """
        Perform Welch's t-test (does not assume equal variances).
        
        Parameters:
            group1, group2: arrays of values
        
        Returns:
            dict with t-statistic, p-value, degrees of freedom
        """
        group1 = np.array([x for x in group1 if x is not None and not np.isnan(x)])
        group2 = np.array([x for x in group2 if x is not None and not np.isnan(x)])
        
        if len(group1) < 2 or len(group2) < 2:
            return {"t": np.nan, "p_value": np.nan, "df": np.nan}
        
        t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)
        
        # Calculate Welch-Satterthwaite degrees of freedom
        n1, n2 = len(group1), len(group2)
        v1, v2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        if v1 == 0 and v2 == 0:
            df = n1 + n2 - 2
        else:
            df = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))
        
        return {"t": t_stat, "p_value": p_val, "df": df}


    def hedges_g(self, group1, group2):
        """
        Calculate Hedges' g (corrected for small sample bias, better than Cohen's d).
        
        Parameters:
            group1, group2: arrays of values
        
        Returns:
            float: Hedges' g effect size
        """
        group1 = np.array([x for x in group1 if x is not None and not np.isnan(x)])
        group2 = np.array([x for x in group2 if x is not None and not np.isnan(x)])
        
        n1, n2 = len(group1), len(group2)
        
        if n1 < 2 or n2 < 2:
            return np.nan
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1-1)*np.var(group1, ddof=1) + (n2-1)*np.var(group2, ddof=1)) / (n1+n2-2))
        
        if pooled_std == 0:
            return 0.0
        
        # Cohen's d
        d = (np.mean(group1) - np.mean(group2)) / pooled_std
        
        # Correction factor for Hedges' g
        correction = 1 - (3 / (4*(n1+n2-2) - 1))
        
        return d * correction


    def omega_squared(self, groups):
        """
        Calculate omega-squared effect size (less biased than eta-squared).
        
        Parameters:
            groups: list of arrays
        
        Returns:
            float: omega-squared value
        """
        all_vals = np.concatenate(groups)
        grand_mean = np.mean(all_vals)
        n_total = len(all_vals)
        k = len(groups)
        
        # Between-group sum of squares
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        
        # Within-group sum of squares
        ss_within = sum(np.sum((g - np.mean(g))**2) for g in groups)
        
        # Mean squares
        ms_between = ss_between / (k - 1)
        ms_within = ss_within / (n_total - k)
        
        # Omega-squared
        omega2 = (ss_between - (k-1)*ms_within) / (ss_between + ss_within + ms_within)
        
        return max(0, omega2)  # Omega-squared should be non-negative


    def mann_whitney_test(self, group1, group2):
        """
        Perform Mann-Whitney U test (non-parametric alternative to t-test).
        Does not assume normality, suitable for skewed or small samples.
        """
        group1 = np.array([x for x in group1 if x is not None and not np.isnan(x)])
        group2 = np.array([x for x in group2 if x is not None and not np.isnan(x)])

        if len(group1) < 3 or len(group2) < 3:
            return {"U": np.nan, "p_value": np.nan}

        try:
            u_stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
            return {"U": u_stat, "p_value": p_val}
        except Exception as e:
            print(f"Error in Mann-Whitney: {e}")
            return {"U": np.nan, "p_value": np.nan}


    def kruskal_wallis_test(self, groups_dict):
        """
        Perform Kruskal-Wallis H-test (non-parametric alternative to one-way ANOVA).
        Tests whether any language group differs significantly from the others.
        """
        groups = [np.array([x for x in v if x is not None and not np.isnan(x)])
                  for v in groups_dict.values()]
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
        Perform statistical analysis across language behaviors for the Buy-Sell Game.

        Tests used:
          - Kruskal-Wallis H-test: overall significance across all languages (per metric)
          - Mann-Whitney U test: pairwise significance between language pairs
          - Benjamini-Hochberg FDR correction: applied to pairwise p-values
          - Chi-square test: for the binary acceptance rate metric
          - Proportion z-test: pairwise acceptance rate comparisons (BH corrected)
        """
        out_dir = Path(self.results_dir) / "stats"
        out_dir.mkdir(exist_ok=True)

        log_file = out_dir / "statistical_analysis_log.txt"
        original_stdout = sys.stdout

        with open(log_file, 'w', encoding='utf-8') as log_f:
            sys.stdout = log_f

            df = pd.DataFrame(self.raw_data)
            df = df[df["game_completed"] == True]
            df_accepted = df[df["trade_occurred"] == True]

            print("=" * 80)
            print("STATISTICAL ANALYSIS - BUY-SELL GAME (LANGUAGE COMPARISON)")
            print("=" * 80)
            print(f"\nTotal completed games: {len(df)}")
            print(f"Accepted trades: {len(df_accepted)}")
            print(f"Rejected trades: {len(df[df['trade_occurred'] == False])}")

            results = []

            # ===================================================================
            # CONTINUOUS METRICS
            # For each metric:
            #   1. Kruskal-Wallis: is there any significant difference across languages?
            #   2. Mann-Whitney U (pairwise, always): which language pairs differ?
            #   3. Benjamini-Hochberg FDR correction on pairwise p-values
            # ===================================================================

            continuous_metrics = {
                "negotiation_rounds": ("Negotiation Rounds", df),
                "seller_advantage":   ("Seller Advantage",   df_accepted),
                "buyer_advantage":    ("Buyer Advantage",    df_accepted),
                "trade_price":        ("Trade Price",        df_accepted),
            }

            for metric, (label, data_subset) in continuous_metrics.items():
                print(f"\n{'='*70}")
                print(f"METRIC: {label}")
                print(f"{'='*70}")

                groups_dict = {}
                behaviors = []

                for b in sorted(data_subset["behavior"].unique()):
                    vals = data_subset.loc[
                        data_subset["behavior"] == b, metric
                    ].dropna().values

                    if len(vals) >= 3:
                        groups_dict[b] = vals
                        behaviors.append(b)
                        print(f"  {b}: n={len(vals)}, mean={np.mean(vals):.2f}, "
                              f"median={np.median(vals):.2f}, std={np.std(vals, ddof=1):.2f}")

                if len(groups_dict) < 2:
                    print(f"  Skipping {label} - insufficient groups")
                    continue

                # --- Kruskal-Wallis (overall test) ---
                kw_result = self.kruskal_wallis_test(groups_dict)
                sig_overall = "YES" if kw_result['p_value'] < 0.05 else "NO"
                print(f"\n  Kruskal-Wallis H-test (overall):")
                print(f"    H({kw_result['df']:.0f}) = {kw_result['H']:.4f}, p = {kw_result['p_value']:.4e}  [Significant: {sig_overall}]")

                results.append({
                    "metric": label,
                    "test": "Kruskal_Wallis",
                    "comparison": "overall",
                    "H": kw_result['H'],
                    "df": kw_result['df'],
                    "p_value": kw_result['p_value'],
                    "p_corrected": kw_result['p_value'],
                    "significant": kw_result['p_value'] < 0.05
                })

                # --- Mann-Whitney U (pairwise, always run) ---
                pairs = list(combinations(behaviors, 2))
                raw_p = []
                pair_stats = []

                for b1, b2 in pairs:
                    g1, g2 = groups_dict[b1], groups_dict[b2]
                    mw = self.mann_whitney_test(g1, g2)
                    raw_p.append(mw['p_value'])
                    pair_stats.append({
                        "b1": b1, "b2": b2,
                        "U": mw['U'],
                        "p_value": mw['p_value'],
                        "mean_diff": np.mean(g1) - np.mean(g2)
                    })

                # Benjamini-Hochberg FDR correction
                valid_mask = [not np.isnan(p) for p in raw_p]
                p_corrected = np.full(len(raw_p), np.nan)
                if any(valid_mask):
                    valid_p = [p for p, v in zip(raw_p, valid_mask) if v]
                    _, corr, _, _ = multipletests(valid_p, method='fdr_bh')
                    vi = 0
                    for i, v in enumerate(valid_mask):
                        if v:
                            p_corrected[i] = corr[vi]
                            vi += 1

                print(f"\n  Pairwise Mann-Whitney U tests (Benjamini-Hochberg FDR corrected):")
                print(f"    {'Comparison':<30} {'Mean Diff':>10} {'U':>10} {'p':>10} {'p_corr':>10} {'Sig'}")
                print(f"    {'-'*78}")

                for i, ps in enumerate(pair_stats):
                    pc = p_corrected[i]
                    sig = "***" if pc < 0.001 else "**" if pc < 0.01 else "*" if pc < 0.05 else "ns"
                    label_str = f"{ps['b1']} vs {ps['b2']}"
                    print(f"    {label_str:<30} {ps['mean_diff']:>10.2f} "
                          f"{ps['U']:>10.1f} {ps['p_value']:>10.4f} {pc:>10.4f} {sig:>3}")

                    results.append({
                        "metric": label,
                        "test": "Mann_Whitney_U",
                        "comparison": f"{ps['b1']} vs {ps['b2']}",
                        "U": ps['U'],
                        "mean_diff": ps['mean_diff'],
                        "p_value": ps['p_value'],
                        "p_corrected": pc,
                        "significant": pc < 0.05
                    })

            # ===================================================================
            # BINARY METRIC: ACCEPTANCE RATE
            # Chi-square across all languages + pairwise proportion z-tests (BH corrected)
            # ===================================================================

            print(f"\n{'='*70}")
            print("BINARY METRIC: Acceptance Rate")
            print(f"{'='*70}")

            behaviors = sorted(df["behavior"].unique())
            contingency = []

            for b in behaviors:
                sub = df[df["behavior"] == b]
                accepted = int((sub["trade_occurred"] == True).sum())
                rejected = int((sub["trade_occurred"] == False).sum())
                total = accepted + rejected
                rate = accepted / total if total > 0 else 0
                contingency.append([accepted, rejected])
                print(f"  {b}: {accepted}/{total} accepted ({rate*100:.1f}%)")

            contingency = np.array(contingency)
            degenerate = any(row[0] == 0 or row[1] == 0 for row in contingency)

            if degenerate:
                print("\n  WARNING: At least one language has perfect acceptance or rejection.")
                print("  Chi-square test is unreliable — skipping.")
                results.append({
                    "metric": "Acceptance Rate",
                    "test": "Chi_square",
                    "comparison": "overall",
                    "note": "Degenerate case",
                    "p_value": np.nan,
                    "p_corrected": np.nan,
                    "significant": False
                })
            else:
                # Chi-square overall
                chi2_stat, p_chi, dof, _ = stats.chi2_contingency(contingency)
                sig_chi = "YES" if p_chi < 0.05 else "NO"
                print(f"\n  Chi-square test (overall):")
                print(f"    chi2({dof}) = {chi2_stat:.4f}, p = {p_chi:.4e}  [Significant: {sig_chi}]")

                results.append({
                    "metric": "Acceptance Rate",
                    "test": "Chi_square",
                    "comparison": "overall",
                    "chi2": chi2_stat,
                    "df": dof,
                    "p_value": p_chi,
                    "p_corrected": p_chi,
                    "significant": p_chi < 0.05
                })

                # Pairwise proportion z-tests (always run, BH corrected)
                pairs = [(i, j) for i in range(len(behaviors)) for j in range(len(behaviors)) if i < j]
                raw_p = []
                pair_stats = []

                for i, j in pairs:
                    count = np.array([contingency[i, 0], contingency[j, 0]])
                    nobs = np.array([contingency[i].sum(), contingency[j].sum()])
                    z_stat, p_val = proportions_ztest(count, nobs)
                    rate1 = contingency[i, 0] / contingency[i].sum()
                    rate2 = contingency[j, 0] / contingency[j].sum()
                    raw_p.append(p_val)
                    pair_stats.append({
                        "b1": behaviors[i], "b2": behaviors[j],
                        "z": z_stat, "p_value": p_val,
                        "rate_diff": rate1 - rate2
                    })

                _, p_corrected, _, _ = multipletests(raw_p, method='fdr_bh')

                print(f"\n  Pairwise proportion z-tests (Benjamini-Hochberg FDR corrected):")
                print(f"    {'Comparison':<30} {'Rate Diff':>10} {'z':>8} {'p':>10} {'p_corr':>10} {'Sig'}")
                print(f"    {'-'*75}")

                for i, ps in enumerate(pair_stats):
                    pc = p_corrected[i]
                    sig = "***" if pc < 0.001 else "**" if pc < 0.01 else "*" if pc < 0.05 else "ns"
                    label_str = f"{ps['b1']} vs {ps['b2']}"
                    print(f"    {label_str:<30} {ps['rate_diff']:>10.3f} "
                          f"{ps['z']:>8.3f} {ps['p_value']:>10.4f} {pc:>10.4f} {sig:>3}")

                    results.append({
                        "metric": "Acceptance Rate",
                        "test": "Proportion_z_test",
                        "comparison": f"{ps['b1']} vs {ps['b2']}",
                        "z": ps['z'],
                        "rate_diff": ps['rate_diff'],
                        "p_value": ps['p_value'],
                        "p_corrected": pc,
                        "significant": pc < 0.05
                    })

            # ===================================================================
            # SAVE RESULTS
            # ===================================================================

            print(f"\n{'='*80}")
            print("SAVING RESULTS")
            print(f"{'='*80}")

            results_df = pd.DataFrame(results)

            csv_file = out_dir / "statistical_tests_buysell.csv"
            results_df.to_csv(csv_file, index=False)
            print(f"  Saved: {csv_file}")

            json_file = out_dir / "statistical_tests_buysell.json"
            results_df.to_json(json_file, orient='records', indent=2)
            print(f"  Saved: {json_file}")

            # Human-readable summary
            summary_file = out_dir / "statistical_summary.txt"
            with open(summary_file, 'w', encoding='utf-8') as sf:
                sf.write("=" * 80 + "\n")
                sf.write("STATISTICAL ANALYSIS SUMMARY - BUY-SELL GAME\n")
                sf.write("=" * 80 + "\n\n")
                sf.write("Tests used:\n")
                sf.write("  - Kruskal-Wallis H-test: overall difference across all languages\n")
                sf.write("  - Mann-Whitney U test: pairwise language comparisons\n")
                sf.write("  - Chi-square test: overall acceptance rate difference\n")
                sf.write("  - Proportion z-test: pairwise acceptance rate comparisons\n")
                sf.write("  - Benjamini-Hochberg FDR correction applied to all pairwise p-values\n")
                sf.write("  Significance: * p<0.05  ** p<0.01  *** p<0.001  ns = not significant\n\n")

                sf.write("OVERALL TESTS\n")
                sf.write("-" * 80 + "\n")
                overall = results_df[results_df['comparison'] == 'overall']
                for _, row in overall.iterrows():
                    sf.write(f"\n{row['metric']} ({row['test']}):\n")
                    if row['test'] == 'Kruskal_Wallis':
                        sf.write(f"  H({row['df']:.0f}) = {row['H']:.4f}, p = {row['p_value']:.4e}\n")
                    elif row['test'] == 'Chi_square':
                        chi2_val = row.get('chi2', 'NA')
                        chi2_str = f"{chi2_val:.4f}" if isinstance(chi2_val, float) and not np.isnan(chi2_val) else "NA"
                        sf.write(f"  chi2({row.get('df', 'NA')}) = {chi2_str}, p = {row['p_value']:.4e}\n")
                    sf.write(f"  Significant: {'YES' if row.get('significant', False) else 'NO'}\n")

                sf.write("\n\nPAIRWISE COMPARISONS (all pairs, Benjamini-Hochberg FDR corrected)\n")
                sf.write("-" * 80 + "\n")
                pairwise = results_df[results_df['comparison'] != 'overall']
                for metric in pairwise['metric'].unique():
                    sf.write(f"\n{metric}:\n")
                    sf.write(f"  {'Comparison':<30} {'p':>10} {'p_corr':>10} {'Sig'}\n")
                    sf.write(f"  {'-'*58}\n")
                    for _, row in pairwise[pairwise['metric'] == metric].iterrows():
                        pc = row.get('p_corrected', np.nan)
                        sig = "***" if pc < 0.001 else "**" if pc < 0.01 else "*" if pc < 0.05 else "ns"
                        sf.write(f"  {row['comparison']:<30} {row['p_value']:>10.4f} {pc:>10.4f} {sig:>3}\n")

            print(f"  Saved: {summary_file}")

            print(f"\n{'='*80}")
            print("STATISTICAL ANALYSIS COMPLETE")
            print(f"{'='*80}\n")

        # Restore stdout
        sys.stdout = original_stdout

        print(f"\nStatistical analysis complete. Results saved to: {out_dir}")
        print(f"  - Full log: {log_file}")
        print(f"  - CSV results: {out_dir / 'statistical_tests_buysell.csv'}")
        print(f"  - JSON results: {out_dir / 'statistical_tests_buysell.json'}")
        print(f"  - Summary: {out_dir / 'statistical_summary.txt'}")

        return results_df

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

    analyzer.run_comprehensive_statistical_analysis()

    print(f"\n" + "=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print(f"Processed {len(raw_data)} games")
    print(f"Generated {len(summary)} combination summaries")
    print(f"Check the generated files in: {results_dir}")


if __name__ == "__main__":
    main()