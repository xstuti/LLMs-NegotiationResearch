#!/usr/bin/env python3
"""
Analyze Buy-Sell Game Results

This script analyzes the buy-sell game results from the log files,
extracting key metrics from game_state.json files.
"""

import json
import os
import re
import statistics

import matplotlib

matplotlib.use("Agg")
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest

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


class BuySellResultsAnalyzer:
    KNOWN_BEHAVIORS = ["Hindi", "Gujarati", "Punjabi", "English"]
    KNOWN_MODELS = [
        "GPT-4o",
        "Claude-3-Haiku",
        "Claude-3.5-Haiku",
        "Llama-3.3-70B-Instruct",
    ]

    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.raw_data = []
        self.summary = {}
        self.behavior_summary = {}

    def find_game_directories(self):
        """Find all game directories. Flat layout: results_dir/model1_model2_language_iterN/"""
        game_dirs = []
        for item in self.results_dir.iterdir():
            if not item.is_dir():
                continue
            parts = item.name.split("_")
            if len(parts) < 4:
                continue
            last = parts[-1]
            if not last.startswith("iter"):
                continue
            suffix = last[len("iter") :]
            if not suffix.isdigit() or not (1 <= int(suffix) <= 30):
                continue
            game_dirs.append(item)
        print(f"Found {len(game_dirs)} game directories")
        return sorted(game_dirs)

    def parse_directory_name(self, dir_name):
        """Parse seller_model, buyer_model, behavior, iteration from dir name.
        Pattern: seller_buyer_language_iterN  e.g. GPT-4o_GPT-3.5_Punjabi_iter4
        """
        print(f"Parsing: {dir_name}")
        try:
            parts = dir_name.split("_")
            if len(parts) < 4:
                raise ValueError("Not enough segments")
            last = parts[-1]
            if not last.startswith("iter"):
                raise ValueError("Last segment must start with 'iter'")
            iteration = int(last[len("iter") :])
            behavior_candidate = parts[-2]
            behavior = behavior_candidate
            for kb in self.KNOWN_BEHAVIORS:
                if behavior_candidate == kb:
                    behavior = kb
                    break
            model_parts = parts[:-2]
            if len(model_parts) < 2:
                raise ValueError("Need at least two model segments")

            def normalize(n):
                return n.replace("-", "_")

            best_idx, best_len = 1, 0
            for i in range(1, len(model_parts)):
                candidate = "_".join(model_parts[:i])
                for km in self.KNOWN_MODELS:
                    if normalize(candidate) == normalize(km) and i > best_len:
                        best_len, best_idx = i, i

            seller_model = "_".join(model_parts[:best_idx])
            buyer_model = "_".join(model_parts[best_idx:])
            for km in self.KNOWN_MODELS:
                if normalize(seller_model) == normalize(km):
                    seller_model = km
                if normalize(buyer_model) == normalize(km):
                    buyer_model = km

            print(
                f"  Parsed: {seller_model} vs {buyer_model} | {behavior} | iter {iteration}"
            )
            return seller_model, buyer_model, behavior, iteration
        except Exception as e:
            print(f"  Error parsing {dir_name}: {e}")
            return "Unknown", "Unknown", "Unknown", 1

    def find_game_state_file(self, game_dir):
        """Find game_state.json inside the single random subdirectory."""
        for subdir in game_dir.iterdir():
            if subdir.is_dir():
                f = subdir / "game_state.json"
                if f.exists():
                    return f
        return None

    def extract_game_data(
        self, game_state_file, seller_model, buyer_model, behavior, iteration
    ):
        """Extract buy-sell metrics from game_state.json.

        Uses the same END-state tolerance as the trading file:
        if no END state, checks whether last state has an ACCEPT tag
        before giving up.
        """
        try:
            with open(game_state_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            game_state = data.get("game_state", [])
            players = data.get("players", [])

            # --- Locate START / END states ---
            start_state = end_state = None
            for state in game_state:
                ci = state.get("current_iteration")
                if ci == "START":
                    start_state = state
                if ci == "END":
                    end_state = state
                    break

            # --- Detect ACCEPT (state-level public info) ---
            accepted = False
            for state in game_state:
                for pk in [
                    "player1_response",
                    "player2_response",
                    "player_public_info_dict",
                ]:
                    resp = state.get(pk, {})
                    if isinstance(resp, dict):
                        pub = (
                            resp.get("player_public_info_dict", resp)
                            if pk != "player_public_info_dict"
                            else resp
                        )
                        if (
                            isinstance(pub, dict)
                            and pub.get("player answer", "").upper() == "ACCEPT"
                        ):
                            accepted = True
                            break
                if accepted:
                    break
                pstr = state.get("player_public_answer_string")
                if isinstance(pstr, str) and ("<ACCEPT" in pstr.upper()):
                    accepted = True
                    break

            # --- Detect ACCEPT (conversation text) ---
            if not accepted:
                pa_re = re.compile(
                    r"<\s*player\s+answer\s*>\s*([A-Za-z]+)\s*<", re.IGNORECASE
                )
                si_re = re.compile(r"<\s*(ACCEPT|REJECT|PROPOSAL)\b", re.IGNORECASE)
                for p in players:
                    for msg in p.get("conversation", []):
                        content = (
                            msg.get("content")
                            if isinstance(msg, dict)
                            else (msg if isinstance(msg, str) else "")
                        )
                        if not content:
                            continue
                        m = pa_re.search(content)
                        if m and m.group(1).upper() == "ACCEPT":
                            accepted = True
                            break
                        m2 = si_re.search(content)
                        if m2 and m2.group(1).upper() == "ACCEPT":
                            accepted = True
                            break
                    if accepted:
                        break

            # --- Handle missing END state (same logic as trading file) ---
            if not end_state:
                if game_state:
                    last = game_state[-1]
                    p1r = last.get("player1_response", {})
                    p2r = last.get("player2_response", {})
                    if p1r.get("tag") == "ACCEPT" or p2r.get("tag") == "ACCEPT":
                        end_state = last
                        print(f"Using last state as end state for {game_state_file}")
                    else:
                        print(
                            f"Warning: No END state and last state not accepted in {game_state_file}"
                        )
                        return None
                else:
                    print(f"Warning: No game_state entries in {game_state_file}")
                    return None

            # --- Summary from end_state ---
            summary = end_state.get("summary") or data.get("summary") or {}

            # --- Negotiation rounds ---
            negotiation_rounds = 0
            try:
                iters = []
                for state in game_state:
                    it = state.get("current_iteration")
                    if isinstance(it, int):
                        iters.append(it)
                    else:
                        try:
                            iters.append(int(it))
                        except:
                            pass
                negotiation_rounds = max(iters) if iters else 0
            except Exception:
                negotiation_rounds = 0

            # --- Valuations ---
            seller_valuation, buyer_valuation = 40, 60
            if start_state:
                pv = start_state.get("settings", {}).get("player_valuation")
                if isinstance(pv, list) and len(pv) >= 2:
                    if isinstance(pv[0], (int, float)):
                        seller_valuation = pv[0]
                    if isinstance(pv[1], (int, float)):
                        buyer_valuation = pv[1]
            if (seller_valuation, buyer_valuation) == (40, 60):
                gs = end_state or start_state
                if gs:
                    goals = gs.get("player_goals", [])
                    for idx, default in [(0, None), (1, None)]:
                        try:
                            g = (
                                goals[idx]
                                .get("_value", {})
                                .get("_value", {})
                                .get("_value", {})
                            )
                            if (
                                isinstance(g, dict)
                                and "X" in g
                                and isinstance(g["X"], (int, float))
                            ):
                                if idx == 0:
                                    seller_valuation = g["X"]
                                else:
                                    buyer_valuation = g["X"]
                        except Exception:
                            pass

            # --- Extract proposed trade ---
            proposed_trade = summary.get("proposed_trade")
            if not proposed_trade:
                for state in reversed(game_state):
                    for pk in [
                        "player1_response",
                        "player2_response",
                        "player_public_info_dict",
                    ]:
                        resp = state.get(pk, {})
                        pub = (
                            resp.get("player_public_info_dict", resp)
                            if isinstance(resp, dict)
                            else {}
                        )
                        t = (
                            pub.get("newly proposed trade")
                            if isinstance(pub, dict)
                            else None
                        )
                        if t and t != "NONE":
                            proposed_trade = t
                            break
                    if proposed_trade:
                        break

            def extract_zup(obj):
                if isinstance(obj, dict):
                    try:
                        blue = obj.get("_value", {}).get("BLUE", {})
                        bv = blue.get("_value", {})
                        if (
                            isinstance(bv, dict)
                            and "ZUP" in bv
                            and isinstance(bv["ZUP"], (int, float))
                        ):
                            return int(bv["ZUP"])
                    except Exception:
                        pass
                    try:
                        blue = obj.get("BLUE") or obj.get("blue")
                        if isinstance(blue, dict):
                            bv = blue.get("_value", blue.get("value", {}))
                            if (
                                isinstance(bv, dict)
                                and "ZUP" in bv
                                and isinstance(bv["ZUP"], (int, float))
                            ):
                                return int(bv["ZUP"])
                    except Exception:
                        pass
                if isinstance(obj, str):
                    m = re.search(r"ZUP\s*[:]*\s*(\d{1,6})", obj, re.IGNORECASE)
                    if m:
                        try:
                            return int(m.group(1))
                        except:
                            pass
                return None

            trade_price = (
                extract_zup(proposed_trade) if proposed_trade is not None else None
            )
            if trade_price is None:
                trade_price = extract_zup(summary.get("proposed_trade"))

            # --- Outcome ---
            final_response = summary.get("final_response") or (
                "ACCEPT" if accepted else "REJECT"
            )
            trade_occurred = (str(final_response).upper() == "ACCEPT") or accepted
            outcome = "ACCEPT" if trade_occurred else "REJECT"

            # --- Advantages ---
            seller_advantage = buyer_advantage = None
            if trade_occurred and trade_price is not None:
                seller_advantage = trade_price - seller_valuation
                buyer_advantage = buyer_valuation - trade_price

            return {
                "seller_model": seller_model,
                "buyer_model": buyer_model,
                "model1": seller_model,
                "model2": buyer_model,  # aliases for stats methods
                "behavior": behavior,
                "iteration": iteration,
                "trade_occurred": trade_occurred,
                "trade_price": trade_price,
                "seller_valuation": seller_valuation,
                "buyer_valuation": buyer_valuation,
                "seller_advantage": seller_advantage,
                "buyer_advantage": buyer_advantage,
                "negotiation_rounds": negotiation_rounds,
                "final_response": final_response,
                "outcome": outcome,
                "file_path": str(game_state_file),
            }

        except Exception as e:
            print(f"Error processing {game_state_file}: {e}")
            return None

    def analyze_all_games(self):
        game_dirs = self.find_game_directories()
        for game_dir in game_dirs:
            try:
                seller, buyer, behavior, iteration = self.parse_directory_name(
                    game_dir.name
                )
                gsf = self.find_game_state_file(game_dir)
                if gsf:
                    gd = self.extract_game_data(gsf, seller, buyer, behavior, iteration)
                    if gd:
                        self.raw_data.append(gd)
                        print(f"+ Processed {game_dir.name}")
                    else:
                        print(f"- Failed to extract data from {game_dir.name}")
                else:
                    print(f"- No game_state.json found in {game_dir.name}")
            except Exception as e:
                print(f"- Error processing {game_dir.name}: {e}")
        print(f"\nTotal games processed: {len(self.raw_data)}")
        return self.raw_data

    def calculate_metrics(self):
        groups = defaultdict(list)
        for game in self.raw_data:
            groups[
                (game["seller_model"], game["buyer_model"], game["behavior"])
            ].append(game)

        summary = {}
        for (seller_model, buyer_model, behavior), games in groups.items():
            combo_key = f"{seller_model}_vs_{buyer_model}_{behavior}"
            valid = [g for g in games if g["outcome"] in ["ACCEPT", "REJECT"]]
            if not valid:
                print(f"Warning: No valid games for {combo_key}")
                continue

            accepted = [g for g in valid if g["outcome"] == "ACCEPT"]
            total = len(valid)
            accepts = [1 if g["outcome"] == "ACCEPT" else 0 for g in valid]
            rounds = [g.get("negotiation_rounds", 0) for g in valid]

            def has_adv(g):
                return isinstance(
                    g.get("seller_advantage"), (int, float)
                ) and isinstance(g.get("buyer_advantage"), (int, float))

            adv_games = [g for g in accepted if has_adv(g)]
            s_advs = [g["seller_advantage"] for g in adv_games]
            b_advs = [g["buyer_advantage"] for g in adv_games]

            p1w = sum(
                1 for g in adv_games if g["seller_advantage"] > g["buyer_advantage"]
            )
            p2w = sum(
                1 for g in adv_games if g["buyer_advantage"] > g["seller_advantage"]
            )
            nd = p1w + p2w
            wr = p1w / nd if nd > 0 else 0.0

            def ms(lst):
                m = statistics.mean(lst) if lst else 0.0
                s = statistics.stdev(lst) if len(lst) > 1 else 0.0
                return m, s

            am, as_ = ms(accepts)
            sm, ss = ms(s_advs)
            bm, bs = ms(b_advs)

            metrics = {
                "total_games": total,
                "acceptance_rate_mean": am,
                "acceptance_rate_std": as_,
                "avg_negotiation_rounds": statistics.mean(rounds) if rounds else 0.0,
                "negotiation_rounds_std": statistics.stdev(rounds)
                if len(rounds) > 1
                else 0.0,
                "seller_advantage_mean": sm,
                "seller_advantage_std": ss,
                "buyer_advantage_mean": bm,
                "buyer_advantage_std": bs,
                "seller_wins": p1w,
                "buyer_wins": p2w,
                "non_draws": nd,
                "win_rate_seller": wr,
                "seller_model": seller_model,
                "buyer_model": buyer_model,
                "model1": seller_model,
                "model2": buyer_model,
                "behavior": behavior,
            }
            summary[combo_key] = metrics

            na, nr = sum(accepts), total - sum(accepts)
            dr = (total - nd) / total if total else 0.0
            print(f"\n{combo_key}:")
            print(f"  Games: {total} | Accepts: {na} | Rejects: {nr}")
            print(f"  Win rate (Seller excl. ties): {wr:.3f} | Draw rate: {dr:.3f}")
            print(
                f"  Avg advantages - Seller: {sm:.1f}+/-{ss:.1f}, Buyer: {bm:.1f}+/-{bs:.1f}"
            )
            print(f"  Avg negotiation rounds: {metrics['avg_negotiation_rounds']:.1f}")

        self.summary = summary
        return summary

    def calculate_behavior_summary(self):
        behavior_groups = defaultdict(list)
        for game in self.raw_data:
            behavior_groups[game["behavior"]].append(game)

        behavior_summary = {}
        for behavior, games in behavior_groups.items():
            if not games:
                continue
            valid = [g for g in games if g["outcome"] in ["ACCEPT", "REJECT"]]
            accepted = [g for g in valid if g["outcome"] == "ACCEPT"]
            accepts = [1 if g["outcome"] == "ACCEPT" else 0 for g in valid]
            rounds = [g.get("negotiation_rounds", 0) for g in valid]

            def has_adv(g):
                return isinstance(
                    g.get("seller_advantage"), (int, float)
                ) and isinstance(g.get("buyer_advantage"), (int, float))

            adv = [g for g in accepted if has_adv(g)]
            s_advs = [g["seller_advantage"] for g in adv]
            b_advs = [g["buyer_advantage"] for g in adv]

            p1w = sum(1 for g in adv if g["seller_advantage"] > g["buyer_advantage"])
            p2w = sum(1 for g in adv if g["buyer_advantage"] > g["seller_advantage"])
            nd = p1w + p2w
            wr = p1w / nd if nd > 0 else 0.0
            dr = (len(adv) - nd) / len(adv) if adv else 0.0

            def ms(lst):
                m = statistics.mean(lst) if lst else 0.0
                s = statistics.stdev(lst) if len(lst) > 1 else 0.0
                return m, s

            am, as_ = ms(accepts)
            sm, ss = ms(s_advs)
            bm, bs = ms(b_advs)

            behavior_summary[behavior] = {
                "behavior": behavior,
                "total_games": len(valid),
                "acceptance_rate_mean": am,
                "acceptance_rate_std": as_,
                "avg_negotiation_rounds": statistics.mean(rounds) if rounds else 0.0,
                "negotiation_rounds_std": statistics.stdev(rounds)
                if len(rounds) > 1
                else 0.0,
                "seller_advantage_mean": sm,
                "seller_advantage_std": ss,
                "buyer_advantage_mean": bm,
                "buyer_advantage_std": bs,
                "seller_wins": p1w,
                "buyer_wins": p2w,
                "non_draws": nd,
                "win_rate_seller": wr,
                "draw_rate": dr,
            }

        self.behavior_summary = behavior_summary
        return behavior_summary

    def create_bar_plots_and_tables(self):
        out_dir = Path(self.results_dir) / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        behaviors = sorted(self.behavior_summary.keys())
        if not behaviors:
            print("No behavior summary to plot")
            return

        acc_m = [self.behavior_summary[b]["acceptance_rate_mean"] for b in behaviors]
        s_m = [self.behavior_summary[b]["seller_advantage_mean"] for b in behaviors]
        b_m = [self.behavior_summary[b]["buyer_advantage_mean"] for b in behaviors]
        win_m = [self.behavior_summary[b]["win_rate_seller"] for b in behaviors]

        colors = ["#4C4CB8", "#2E8BC0", "#18A162", "#6AD34F", "#D5E86B", "#B7E06C"]
        bc = [colors[i % len(colors)] for i in range(len(behaviors))]
        x = np.arange(len(behaviors))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        def bar_plot(ax, vals, title, pct=False, ylabel=None):
            ax.bar(x, vals, color=bc, edgecolor="black")
            ax.set_xticks(x)
            ax.set_xticklabels(behaviors, rotation=30, ha="right")
            ax.set_title(title)
            if ylabel:
                ax.set_ylabel(ylabel)
            if pct:
                ax.set_ylim(0, 1.05)
            off = (max(abs(v) for v in vals) * 0.03) if vals else 0.1
            for i, v in enumerate(vals):
                lbl = f"{v * 100:.2f}%" if pct else f"{v:.2f}"
                ax.text(i, v + off, lbl, ha="center", fontsize=9)

        bar_plot(axes[0, 0], acc_m, "Acceptance Rate by Behavior", pct=True)
        bar_plot(axes[0, 1], s_m, "Seller Advantage by Behavior", ylabel="ZUP surplus")
        bar_plot(axes[1, 0], b_m, "Buyer Advantage by Behavior", ylabel="ZUP surplus")
        bar_plot(axes[1, 1], win_m, "Seller Win Rate by Behavior", pct=True)

        plt.tight_layout()
        ff = out_dir / "behavior_bar_summary.png"
        fig.savefig(ff, dpi=150)
        plt.close(fig)

        csv_file = Path(self.results_dir) / "behavior_metrics_summary.csv"
        hdrs = [
            "behavior",
            "total_games",
            "acceptance_rate_mean",
            "acceptance_rate_std",
            "seller_advantage_mean",
            "seller_advantage_std",
            "buyer_advantage_mean",
            "buyer_advantage_std",
            "win_rate_seller",
            "avg_negotiation_rounds",
            "negotiation_rounds_std",
        ]
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write(",".join(hdrs) + "\n")
            for b in behaviors:
                m = self.behavior_summary[b]
                row = [
                    b,
                    str(m["total_games"]),
                    f"{m['acceptance_rate_mean']:.4f}",
                    f"{m['acceptance_rate_std']:.4f}",
                    f"{m['seller_advantage_mean']:.4f}",
                    f"{m['seller_advantage_std']:.4f}",
                    f"{m['buyer_advantage_mean']:.4f}",
                    f"{m['buyer_advantage_std']:.4f}",
                    f"{m['win_rate_seller']:.4f}",
                    f"{m['avg_negotiation_rounds']:.4f}",
                    f"{m['negotiation_rounds_std']:.4f}",
                ]
                f.write(",".join(row) + "\n")

        print(f"Saved bar plot: {ff}")
        print(f"Saved CSV: {csv_file}")

    def create_model_summary_csv(self):
        csv_file = self.results_dir / "data_summary_model.csv"
        hdrs = [
            "modelA",
            "modelB",
            "lang",
            "buyer_adv",
            "seller_adv",
            "nego_rounds",
            "acceptance_rate",
        ]
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write(",".join(hdrs) + "\n")
            for ck, m in self.summary.items():
                row = [
                    str(m.get("model1", "")),
                    str(m.get("model2", "")),
                    str(m.get("behavior", "")),
                    f"{m.get('buyer_advantage_mean', 0):.4f}",
                    f"{m.get('seller_advantage_mean', 0):.4f}",
                    f"{m.get('avg_negotiation_rounds', 0):.4f}",
                    f"{m.get('acceptance_rate_mean', 0):.4f}",
                ]
                f.write(",".join(row) + "\n")

    def save_results(self):
        self.create_model_summary_csv()

        def jdump(obj, path):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False)

        jdump(self.raw_data, self.results_dir / "raw_game_data.json")
        jdump(self.summary, self.results_dir / "summary.json")
        jdump(self.behavior_summary, self.results_dir / "behavior_summary.json")

        csv_file = self.results_dir / "behavior_summary.csv"
        hdrs = [
            "behavior",
            "total_games",
            "acceptance_rate_mean",
            "acceptance_rate_std",
            "seller_advantage_mean",
            "seller_advantage_std",
            "buyer_advantage_mean",
            "buyer_advantage_std",
            "win_rate_seller",
            "avg_negotiation_rounds",
            "negotiation_rounds_std",
        ]
        with open(csv_file, "w", encoding="utf-8") as f:
            f.write(",".join(hdrs) + "\n")
            for behavior, m in self.behavior_summary.items():
                row = [
                    behavior,
                    str(m.get("total_games", 0)),
                    f"{m.get('acceptance_rate_mean', 0):.4f}",
                    f"{m.get('acceptance_rate_std', 0):.4f}",
                    f"{m.get('seller_advantage_mean', 0):.4f}",
                    f"{m.get('seller_advantage_std', 0):.4f}",
                    f"{m.get('buyer_advantage_mean', 0):.4f}",
                    f"{m.get('buyer_advantage_std', 0):.4f}",
                    f"{m.get('win_rate_seller', 0):.4f}",
                    f"{m.get('avg_negotiation_rounds', 0):.4f}",
                    f"{m.get('negotiation_rounds_std', 0):.4f}",
                ]
                f.write(",".join(row) + "\n")

        readable = self.results_dir / "readable_summary.txt"
        with open(readable, "w", encoding="utf-8") as f:
            f.write("BUY-SELL GAME RESULTS SUMMARY\n" + "=" * 50 + "\n\n")
            for ck, m in self.summary.items():
                f.write(f"{ck}\n" + "-" * 40 + "\n")
                f.write(f"Total Games: {m['total_games']}\n")
                f.write(
                    f"Acceptance Rate: {m.get('acceptance_rate_mean', 0):.3f} +/- {m.get('acceptance_rate_std', 0):.3f}\n"
                )
                f.write(
                    f"Seller Win Rate (excl. ties): {m.get('win_rate_seller', 0):.3f}\n"
                )
                f.write(
                    f"Seller Advantage: {m.get('seller_advantage_mean', 0):.2f} +/- {m.get('seller_advantage_std', 0):.2f}\n"
                )
                f.write(
                    f"Buyer Advantage: {m.get('buyer_advantage_mean', 0):.2f} +/- {m.get('buyer_advantage_std', 0):.2f}\n"
                )
                f.write(
                    f"Avg Negotiation Rounds: {m.get('avg_negotiation_rounds', 0):.1f} +/- {m.get('negotiation_rounds_std', 0):.1f}\n\n"
                )
            f.write("\nBEHAVIOR-LEVEL AVERAGES\n" + "=" * 50 + "\n")
            for b, m in self.behavior_summary.items():
                f.write(f"{b}\n" + "-" * 40 + "\n")
                f.write(f"Total Games: {m.get('total_games', 0)}\n")
                f.write(
                    f"Acceptance Rate: {m.get('acceptance_rate_mean', 0):.3f} +/- {m.get('acceptance_rate_std', 0):.3f}\n"
                )
                f.write(
                    f"Seller Win Rate (excl. ties): {m.get('win_rate_seller', 0):.3f}\n"
                )
                f.write(
                    f"Seller Advantage: {m.get('seller_advantage_mean', 0):.2f} +/- {m.get('seller_advantage_std', 0):.2f}\n"
                )
                f.write(
                    f"Buyer Advantage: {m.get('buyer_advantage_mean', 0):.2f} +/- {m.get('buyer_advantage_std', 0):.2f}\n"
                )
                f.write(
                    f"Avg Negotiation Rounds: {m.get('avg_negotiation_rounds', 0):.1f} +/- {m.get('negotiation_rounds_std', 0):.1f}\n\n"
                )

        print(f"\nResults saved to: {self.results_dir}")

    def create_heatmap_data(self):
        models = sorted(
            {
                m
                for ck, met in self.summary.items()
                for m in [met["seller_model"], met["buyer_model"]]
            }
        )
        behaviors = sorted({met["behavior"] for met in self.summary.values()})
        heatmap_data = {}

        for behavior in behaviors:
            s_adv = {s: {b2: None for b2 in models} for s in models}
            b_adv = {s: {b2: None for b2 in models} for s in models}
            acc = {s: {b2: None for b2 in models} for s in models}
            for s in models:
                for bm in models:
                    ck = f"{s}_vs_{bm}_{behavior}"
                    if ck in self.summary:
                        m = self.summary[ck]
                        s_adv[s][bm] = m.get("seller_advantage_mean")
                        b_adv[s][bm] = m.get("buyer_advantage_mean")
                        acc[s][bm] = m.get("acceptance_rate_mean")
            heatmap_data[behavior] = {
                "seller_advantage": s_adv,
                "buyer_advantage": b_adv,
                "acceptance_rate": acc,
                "models": models,
            }

        jpath = self.results_dir / "all_heatmaps_buysell_data.json"
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(heatmap_data, f, indent=2)

        plots_dir = Path(self.results_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        nb = max(len(behaviors), 1)
        fig, axes = plt.subplots(nrows=nb, ncols=2, figsize=(12, 4 * nb), squeeze=False)
        type_cfgs = [
            ("seller_advantage", "Seller Advantage", "Oranges"),
            ("buyer_advantage", "Buyer Advantage", "Blues"),
        ]

        for bi, behavior in enumerate(behaviors):
            d = heatmap_data[behavior]
            mdls = d["models"]
            nm = len(mdls)
            for ti, (ht, title, cmap_name) in enumerate(type_cfgs):
                ax = axes[bi, ti]
                mat = np.full((nm, nm), np.nan)
                for i, a in enumerate(mdls):
                    for j, b in enumerate(mdls):
                        v = d[ht].get(a, {}).get(b)
                        if v is not None:
                            mat[i, j] = float(v)
                mm = np.ma.masked_invalid(mat)
                im = ax.imshow(mm, cmap=plt.cm.get_cmap(cmap_name))
                ax.set_xticks(range(nm))
                ax.set_yticks(range(nm))
                ax.set_xticklabels(mdls, rotation=45, ha="right")
                ax.set_yticklabels(mdls)
                if bi == 0:
                    ax.set_title(title, fontsize=13, pad=6)
                if ti == 0:
                    ax.set_ylabel(f"{behavior}\nSeller Model", fontsize=11)
                fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).ax.set_ylabel(
                    "ZUP Surplus", rotation=270, labelpad=15
                )
                for i in range(nm):
                    for j in range(nm):
                        if i == j:
                            ax.text(
                                j,
                                i,
                                "N/A",
                                ha="center",
                                va="center",
                                color="gray",
                                fontsize=8,
                            )
                        elif np.ma.is_masked(mm[i, j]):
                            ax.text(
                                j,
                                i,
                                "-",
                                ha="center",
                                va="center",
                                color="gray",
                                fontsize=8,
                            )
                        else:
                            v = mat[i, j]
                            tc = (
                                "white"
                                if (not np.isnan(mat.max()) and v > mat.max() * 0.7)
                                else "black"
                            )
                            ax.text(
                                j,
                                i,
                                f"{v:.1f}",
                                ha="center",
                                va="center",
                                color=tc,
                                fontsize=10,
                                fontweight="bold",
                            )

        plt.tight_layout(pad=0.8, w_pad=0.4, h_pad=0.6)
        hf = plots_dir / "all_heatmaps.png"
        fig.savefig(hf, dpi=150)
        plt.close(fig)
        print(f"  Heatmap JSON: {jpath}")
        print(f"  Heatmap image: {hf}")
        return heatmap_data

    # -------------------------------------------------------------------------
    # Statistical helpers (identical to trading file)
    # -------------------------------------------------------------------------

    def mann_whitney_test(self, g1, g2):
        g1 = np.array([x for x in g1 if x is not None and not np.isnan(x)])
        g2 = np.array([x for x in g2 if x is not None and not np.isnan(x)])
        if len(g1) < 3 or len(g2) < 3:
            return {"U": np.nan, "p_value": np.nan}
        try:
            u, p = mannwhitneyu(g1, g2, alternative="two-sided")
            return {"U": u, "p_value": p}
        except Exception as e:
            print(f"Mann-Whitney error: {e}")
            return {"U": np.nan, "p_value": np.nan}

    def kruskal_wallis_test(self, gd):
        groups = [
            np.array([x for x in v if x is not None and not np.isnan(x)])
            for v in gd.values()
        ]
        groups = [g for g in groups if len(g) >= 3]
        if len(groups) < 2:
            return {"H": np.nan, "p_value": np.nan, "df": np.nan}
        try:
            h, p = stats.kruskal(*groups)
            return {"H": h, "p_value": p, "df": len(groups) - 1}
        except Exception as e:
            print(f"Kruskal-Wallis error: {e}")
            return {"H": np.nan, "p_value": np.nan, "df": np.nan}

    def run_comprehensive_statistical_analysis(self):
        """Statistical analysis across language behaviors for the Buy-Sell Game."""
        out_dir = Path(self.results_dir) / "stats"
        out_dir.mkdir(exist_ok=True)
        log_file = out_dir / "statistical_analysis_log_buysell.txt"
        original_stdout = sys.stdout

        with open(log_file, "w", encoding="utf-8") as log_f:
            sys.stdout = log_f

            df = pd.DataFrame(self.raw_data)
            df_acc = df[df["outcome"] == "ACCEPT"]

            print("=" * 80)
            print("STATISTICAL ANALYSIS - BUY-SELL GAME (LANGUAGE COMPARISON)")
            print("=" * 80)
            print(f"\nTotal games: {len(df)}")
            print(
                f"Accepted: {len(df_acc)}  |  Rejected: {len(df[df['outcome'] == 'REJECT'])}"
            )

            results = []

            continuous = {
                "seller_advantage": ("Seller Advantage", df_acc),
                "buyer_advantage": ("Buyer Advantage", df_acc),
                "negotiation_rounds": ("Negotiation Rounds", df),
            }

            for metric, (label, subset) in continuous.items():
                print(f"\n{'=' * 70}\nMETRIC: {label}\n{'=' * 70}")
                gd, blist = {}, []
                for b in sorted(subset["behavior"].unique()):
                    vals = subset.loc[
                        (subset["behavior"] == b) & subset[metric].notna(), metric
                    ].values
                    if len(vals) >= 3:
                        gd[b] = vals
                        blist.append(b)
                        print(
                            f"  {b}: n={len(vals)}, mean={np.mean(vals):.2f}, median={np.median(vals):.2f}, std={np.std(vals, ddof=1):.2f}"
                        )
                if len(gd) < 2:
                    print(f"  Skipping {label} - insufficient groups")
                    continue

                kw = self.kruskal_wallis_test(gd)
                print(
                    f"\n  Kruskal-Wallis: H({kw['df']:.0f})={kw['H']:.4f}, p={kw['p_value']:.4e}  [{'YES' if kw['p_value'] < 0.05 else 'NO'}]"
                )
                results.append(
                    {
                        "metric": label,
                        "test": "Kruskal_Wallis",
                        "comparison": "overall",
                        "H": kw["H"],
                        "df": kw["df"],
                        "p_value": kw["p_value"],
                        "p_corrected": kw["p_value"],
                        "significant": kw["p_value"] < 0.05,
                    }
                )

                pairs = list(combinations(blist, 2))
                raw_p, ps_list = [], []
                for b1, b2 in pairs:
                    mw = self.mann_whitney_test(gd[b1], gd[b2])
                    raw_p.append(mw["p_value"])
                    ps_list.append(
                        {
                            "b1": b1,
                            "b2": b2,
                            "U": mw["U"],
                            "p_value": mw["p_value"],
                            "mean_diff": np.mean(gd[b1]) - np.mean(gd[b2]),
                        }
                    )

                vmask = [not np.isnan(p) for p in raw_p]
                pc_arr = np.full(len(raw_p), np.nan)
                if any(vmask):
                    _, corr, _, _ = multipletests(
                        [p for p, v in zip(raw_p, vmask) if v], method="fdr_bh"
                    )
                    vi = 0
                    for i, v in enumerate(vmask):
                        if v:
                            pc_arr[i] = corr[vi]
                            vi += 1

                print(f"\n  Pairwise Mann-Whitney U (BH FDR):")
                print(
                    f"    {'Comparison':<30} {'Mean Diff':>10} {'U':>10} {'p':>10} {'p_corr':>10} Sig"
                )
                print(f"    {'-' * 78}")
                for i, ps in enumerate(ps_list):
                    pc = pc_arr[i]
                    sig = (
                        "***"
                        if pc < 0.001
                        else "**"
                        if pc < 0.01
                        else "*"
                        if pc < 0.05
                        else "ns"
                    )
                    print(
                        f"    {ps['b1']} vs {ps['b2']:<20} {ps['mean_diff']:>10.2f} {ps['U']:>10.1f} {ps['p_value']:>10.4f} {pc:>10.4f} {sig}"
                    )
                    results.append(
                        {
                            "metric": label,
                            "test": "Mann_Whitney_U",
                            "comparison": f"{ps['b1']} vs {ps['b2']}",
                            "U": ps["U"],
                            "mean_diff": ps["mean_diff"],
                            "p_value": ps["p_value"],
                            "p_corrected": pc,
                            "significant": pc < 0.05,
                        }
                    )

            # Acceptance rate
            print(f"\n{'=' * 70}\nBINARY METRIC: Acceptance Rate\n{'=' * 70}")
            behaviors = sorted(df["behavior"].unique())
            contingency = []
            for b in behaviors:
                sub = df[df["behavior"] == b]
                acc_n = int((sub["outcome"] == "ACCEPT").sum())
                rej_n = int((sub["outcome"] == "REJECT").sum())
                tot = acc_n + rej_n
                print(
                    f"  {b}: {acc_n}/{tot} accepted ({acc_n / tot * 100 if tot else 0:.1f}%)"
                )
                contingency.append([acc_n, rej_n])
            contingency = np.array(contingency)
            degen = any(row[0] == 0 or row[1] == 0 for row in contingency)

            if degen:
                print("\n  WARNING: Degenerate case - chi-square skipped.")
                results.append(
                    {
                        "metric": "Acceptance Rate",
                        "test": "Chi_square",
                        "comparison": "overall",
                        "note": "Degenerate",
                        "p_value": np.nan,
                        "p_corrected": np.nan,
                        "significant": False,
                    }
                )
            else:
                chi2v, p_chi, dof, _ = stats.chi2_contingency(contingency)
                print(
                    f"\n  Chi-square: chi2({dof})={chi2v:.4f}, p={p_chi:.4e}  [{'YES' if p_chi < 0.05 else 'NO'}]"
                )
                results.append(
                    {
                        "metric": "Acceptance Rate",
                        "test": "Chi_square",
                        "comparison": "overall",
                        "chi2": chi2v,
                        "df": dof,
                        "p_value": p_chi,
                        "p_corrected": p_chi,
                        "significant": p_chi < 0.05,
                    }
                )

                pairs = [
                    (i, j)
                    for i in range(len(behaviors))
                    for j in range(i + 1, len(behaviors))
                ]
                raw_p, ps_list = [], []
                for i, j in pairs:
                    z, p = proportions_ztest(
                        np.array([contingency[i, 0], contingency[j, 0]]),
                        np.array([contingency[i].sum(), contingency[j].sum()]),
                    )
                    raw_p.append(p)
                    ps_list.append(
                        {
                            "b1": behaviors[i],
                            "b2": behaviors[j],
                            "z": z,
                            "p_value": p,
                            "rate_diff": contingency[i, 0] / contingency[i].sum()
                            - contingency[j, 0] / contingency[j].sum(),
                        }
                    )

                _, pc_corr, _, _ = multipletests(raw_p, method="fdr_bh")
                print(f"\n  Pairwise proportion z-tests (BH FDR):")
                print(
                    f"    {'Comparison':<30} {'Rate Diff':>10} {'z':>8} {'p':>10} {'p_corr':>10} Sig"
                )
                print(f"    {'-' * 75}")
                for i, ps in enumerate(ps_list):
                    pc = pc_corr[i]
                    sig = (
                        "***"
                        if pc < 0.001
                        else "**"
                        if pc < 0.01
                        else "*"
                        if pc < 0.05
                        else "ns"
                    )
                    print(
                        f"    {ps['b1']} vs {ps['b2']:<20} {ps['rate_diff']:>10.3f} {ps['z']:>8.3f} {ps['p_value']:>10.4f} {pc:>10.4f} {sig}"
                    )
                    results.append(
                        {
                            "metric": "Acceptance Rate",
                            "test": "Proportion_z_test",
                            "comparison": f"{ps['b1']} vs {ps['b2']}",
                            "z": ps["z"],
                            "rate_diff": ps["rate_diff"],
                            "p_value": ps["p_value"],
                            "p_corrected": pc,
                            "significant": pc < 0.05,
                        }
                    )

            rdf = pd.DataFrame(results)
            rdf.to_csv(out_dir / "statistical_tests_buysell.csv", index=False)
            rdf.to_json(
                out_dir / "statistical_tests_buysell.json", orient="records", indent=2
            )

            sfile = out_dir / "statistical_summary_buysell.txt"
            with open(sfile, "w", encoding="utf-8") as sf:
                sf.write(
                    "=" * 80
                    + "\nSTATISTICAL ANALYSIS SUMMARY - BUY-SELL GAME\n"
                    + "=" * 80
                    + "\n\n"
                )
                sf.write(
                    "Tests: Kruskal-Wallis | Mann-Whitney U | Chi-square | Proportion z-test\n"
                )
                sf.write(
                    "Correction: Benjamini-Hochberg FDR  |  * p<0.05  ** p<0.01  *** p<0.001\n\n"
                )
                sf.write("OVERALL TESTS\n" + "-" * 80 + "\n")
                for _, row in rdf[rdf["comparison"] == "overall"].iterrows():
                    sf.write(f"\n{row['metric']} ({row['test']}):\n")
                    if row["test"] == "Kruskal_Wallis":
                        sf.write(
                            f"  H({row['df']:.0f})={row['H']:.4f}, p={row['p_value']:.4e}\n"
                        )
                    elif row["test"] == "Chi_square":
                        cv = row.get("chi2", "NA")
                        cs = (
                            f"{cv:.4f}"
                            if isinstance(cv, float) and not np.isnan(cv)
                            else "NA"
                        )
                        sf.write(
                            f"  chi2({row.get('df', 'NA')})={cs}, p={row['p_value']:.4e}\n"
                        )
                    sf.write(
                        f"  Significant: {'YES' if row.get('significant', False) else 'NO'}\n"
                    )
                sf.write("\n\nPAIRWISE COMPARISONS\n" + "-" * 80 + "\n")
                pw = rdf[rdf["comparison"] != "overall"]
                for ml in pw["metric"].unique():
                    sf.write(
                        f"\n{ml}:\n  {'Comparison':<30} {'p':>10} {'p_corr':>10} Sig\n  {'-' * 55}\n"
                    )
                    for _, row in pw[pw["metric"] == ml].iterrows():
                        pc = row.get("p_corrected", np.nan)
                        sig = (
                            "***"
                            if pc < 0.001
                            else "**"
                            if pc < 0.01
                            else "*"
                            if pc < 0.05
                            else "ns"
                        )
                        sf.write(
                            f"  {row['comparison']:<30} {row['p_value']:>10.4f} {pc:>10.4f} {sig}\n"
                        )

            print(f"\nSaved: {out_dir}")

        sys.stdout = original_stdout
        print(f"\nStatistical analysis complete. Results in: {out_dir}")
        return rdf


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_buysell_results.py <results_directory>")
        return
    results_dir = sys.argv[1]
    if not os.path.exists(results_dir):
        print(f"Error: {results_dir} does not exist")
        return

    print(f"Analyzing buy-sell results in: {results_dir}")
    print("=" * 60)

    analyzer = BuySellResultsAnalyzer(results_dir)
    raw_data = analyzer.analyze_all_games()
    if not raw_data:
        print("No valid game data found!")
        return

    analyzer.calculate_metrics()
    analyzer.calculate_behavior_summary()
    analyzer.create_bar_plots_and_tables()
    analyzer.save_results()
    analyzer.create_heatmap_data()

    print("\n" + "=" * 60 + "\nANALYSIS COMPLETE!\n" + "=" * 60)
    print(f"Processed {len(raw_data)} games")
    print(f"Check output files in: {results_dir}")
    print("\n" + "=" * 60 + "\nRunning Statistical Analysis...\n" + "=" * 60)
    analyzer.run_comprehensive_statistical_analysis()


if __name__ == "__main__":
    main()
