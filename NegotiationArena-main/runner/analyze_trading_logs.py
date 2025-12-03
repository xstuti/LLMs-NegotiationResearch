import os
import json
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


"""
Custom analysis script for trading game logs stored in `.logs/trading`.

For each run (subdirectory with a `game_state.json`) we:
- Load the final summary state (current_iteration == "END")
- Group results by (model1, model2, social_behavior)
- Compute:
    - Win rate for Player 2 (excluding draws)
    - Draw rate
    - Payoff for Player 2 (average and list of all values)
    - Initial proposed value for Player 2 (average and list of all values)
      where the proposed value is the payoff implied by the final proposed trade,
      even if it was rejected.

Runs with malformed / incomplete JSON (e.g. missing END state) are skipped.

We also perform a light-weight qualitative analysis of the text content of the
game (languages/scripts used, code-switching, politeness, cultural markers)
and surface notable incidents.

Outputs:
- `trading_analysis/summary.json`  (quantitative + qualitative summary)
- `trading_analysis/figures/*.png` (heatmaps for win rate and payoff)
"""


ROOT_DIR = Path(".")
LOG_DIR = ROOT_DIR / ".logs" / "trading"
OUTPUT_DIR = ROOT_DIR / "trading_analysis"
FIGURES_DIR = OUTPUT_DIR / "figures"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"


# Map full model IDs to short readable names (fallback: use raw ID)
MODEL_NAME_MAP = {
    "openai/chatgpt-4o-latest": "GPT-4o",
    "openai/gpt-3.5-turbo": "GPT-3.5",
    "openai/gpt-oss-20b:free": "GPT-OSS-20B",
    "meta-llama/llama-3.3-70b-instruct:free": "Llama-3.3-70B",
    "gemini-2.5-pro": "gemini-2.5-pro",
    "gemini-3-pro-preview": "gemini-3-pro-preview",
}


def canonical_model(model_id: str) -> str:
    return MODEL_NAME_MAP.get(model_id, model_id)


def infer_social_behavior_label(behaviours: List[str]) -> str:
    """Infer a compact social behavior label from the prompt text."""
    if not behaviours:
        return "Unknown"
    text = " ".join(behaviours)
    lowered = text.lower()
    if "gujarati" in lowered:
        return "Gujarati"
    if "hindi" in lowered:
        return "Hindi"
    if "marwadi" in lowered or "marwadi businessman" in lowered:
        return "Marwadi"
    if "punjabi" in lowered:
        return "Punjabi"
    return "Unknown"


def extract_resource_dict(res_obj: Any) -> Dict[str, float]:
    """
    Extract the underlying resource dict from a JSON-serialised `Resources` object.
    Expected shape: {"_type": "resource", "_value": {"X": 25, "Y": 5}}
    """
    if isinstance(res_obj, dict) and "_value" in res_obj:
        inner = res_obj["_value"]
        if isinstance(inner, dict):
            return {k: float(v) for k, v in inner.items()}
    # Already a plain dict or unexpected format
    if isinstance(res_obj, dict):
        return {k: float(v) for k, v in res_obj.items()}
    return {}


def resources_value(res_dict: Dict[str, float]) -> float:
    return float(sum(res_dict.values()))


def add_resources(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    keys = set(a.keys()) | set(b.keys())
    return {k: a.get(k, 0.0) + b.get(k, 0.0) for k in keys}


def sub_resources(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    keys = set(a.keys()) | set(b.keys())
    return {k: a.get(k, 0.0) - b.get(k, 0.0) for k in keys}


def simulate_trade_effect(
    initial_resources: List[Dict[str, float]],
    proposed_trade: Dict[str, Any],
) -> List[Dict[str, float]]:
    """
    Re-implement `Trade.execute_trade` logic on JSON-serialised trade objects.

    proposed_trade example:
    {
        "_type": "trade",
        "_value": {
            "RED": {"_type": "resource", "_value": {"X": 10}},
            "BLUE": {"_type": "resource", "_value": {"Y": 10}}
        }
    }

    Player 0 is RED, Player 1 is BLUE.
    """
    if not proposed_trade or "_value" not in proposed_trade:
        return initial_resources

    trade_value = proposed_trade["_value"]
    if not isinstance(trade_value, dict) or len(trade_value) != 2:
        return initial_resources

    keys = sorted(list(trade_value.keys()), reverse=True)
    # In the original Trade class, `resources_from_first_agent` corresponds to keys[0]
    res_first = extract_resource_dict(trade_value[keys[0]])
    res_second = extract_resource_dict(trade_value[keys[1]])

    # Direction 0: net = second - first; Direction 1: net = first - second
    net_for_p0 = sub_resources(res_second, res_first)
    net_for_p1 = sub_resources(res_first, res_second)

    final0 = add_resources(initial_resources[0], net_for_p0)
    final1 = add_resources(initial_resources[1], net_for_p1)
    return [final0, final1]


def detect_scripts(text: str) -> Counter:
    """
    Very lightweight script detection based on Unicode ranges.
    Returns counts of characters per detected script.
    """
    counts: Counter = Counter()
    for ch in text:
        code = ord(ch)
        # Devanagari (Hindi, some Marathi)
        if 0x0900 <= code <= 0x097F:
            counts["Devanagari"] += 1
        # Gujarati
        elif 0x0A80 <= code <= 0x0AFF:
            counts["Gujarati"] += 1
        # Gurmukhi (Punjabi)
        elif 0x0A00 <= code <= 0x0A7F:
            counts["Gurmukhi"] += 1
        # Basic Latin letters
        elif ("A" <= ch <= "Z") or ("a" <= ch <= "z"):
            counts["Latin"] += 1
    return counts


POLITENESS_MARKERS = [
    "please",
    "thank you",
    "thanks",
    "kindly",
    "sorry",
]

CULTURAL_MARKERS = [
    "namaste",
    "salaam",
    "balle",
    "balle balle",
    "garba",
    "navratri",
    "diwali",
    "holi",
    "lassi",
    "chai",
]

STEREOTYPE_MARKERS = [
    "stereotype",
    "cheap",
    "miser",
    "money-minded",
    "greedy",
]


def analyse_text_incidents(text: str) -> Dict[str, Any]:
    """Analyse a single text chunk for qualitative markers."""
    scripts = detect_scripts(text)
    lowered = text.lower()

    politeness = [w for w in POLITENESS_MARKERS if w in lowered]
    cultural = [w for w in CULTURAL_MARKERS if w in lowered]
    stereotypes = [w for w in STEREOTYPE_MARKERS if w in lowered]

    incident = {
        "scripts": dict(scripts),
        "code_switching": len([s for s, c in scripts.items() if c > 0]) > 1,
        "politeness_markers": politeness,
        "cultural_markers": cultural,
        "stereotype_markers": stereotypes,
    }
    return incident


def extract_run_data(log_dir: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load and process a single game_state.json.

    Returns:
        quantitative_result, qualitative_result
    or (None, None) if the run should be skipped.
    """
    game_state_path = log_dir / "game_state.json"
    try:
        with game_state_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[SKIP] Failed to load JSON from {game_state_path}: {e}")
        return None, None

    try:
        players = data.get("players", [])
        if len(players) < 2:
            raise ValueError("Expected at least 2 players")

        model1_id = players[0].get("model", "unknown")
        model2_id = players[1].get("model", "unknown")

        model1 = canonical_model(model1_id)
        model2 = canonical_model(model2_id)

        social_behaviour_prompts = data.get("player_social_behaviour") or data.get(
            "game_state", [{}]
        )[0].get("settings", {}).get("player_social_behaviour", [])
        social_label = infer_social_behavior_label(social_behaviour_prompts)

        game_states = data.get("game_state", [])
        if not game_states:
            raise ValueError("Missing game_state array")

        # Find END state
        end_state = None
        for st in reversed(game_states):
            if st.get("current_iteration") == "END":
                end_state = st
                break

        if not end_state:
            raise ValueError("No END state found (likely incomplete run)")

        summary = end_state.get("summary", {})
        if not summary:
            raise ValueError("Missing summary in END state")

        initial_resources_raw = summary.get("initial_resources")
        final_resources_raw = summary.get("final_resources")
        proposed_trade = summary.get("proposed_trade")
        final_response = summary.get("final_response", "")

        if not initial_resources_raw or not final_resources_raw:
            raise ValueError("Missing initial or final resources in summary")

        initial_resources = [
            extract_resource_dict(r) for r in initial_resources_raw
        ]
        final_resources = [extract_resource_dict(r) for r in final_resources_raw]

        if len(initial_resources) != 2 or len(final_resources) != 2:
            raise ValueError("Expected resources for exactly 2 players")

        # Quantitative metrics
        init_vals = [resources_value(r) for r in initial_resources]
        final_vals = [resources_value(r) for r in final_resources]

        payoffs = [f - i for f, i in zip(final_vals, init_vals)]

        # Simulated value of the final proposed trade ("initial offer") even if rejected
        offered_final_resources = simulate_trade_effect(
            initial_resources, proposed_trade
        )
        offered_final_vals = [
            resources_value(r) for r in offered_final_resources
        ]
        offer_values = [of - iv for of, iv in zip(offered_final_vals, init_vals)]

        # Win / draw from final resources
        if abs(final_vals[0] - final_vals[1]) < 1e-9:
            winner = None
            is_draw = True
        elif final_vals[1] > final_vals[0]:
            winner = "player2"
            is_draw = False
        else:
            winner = "player1"
            is_draw = False

        quantitative = {
            "model1": model1,
            "model2": model2,
            "model1_id": model1_id,
            "model2_id": model2_id,
            "social_behavior": social_label,
            "social_behavior_prompts": social_behaviour_prompts,
            "final_response": final_response,
            "initial_resources_value": init_vals,
            "final_resources_value": final_vals,
            "payoffs": payoffs,
            "offer_values": offer_values,
            "winner": winner,
            "is_draw": is_draw,
            "log_dir": str(log_dir),
        }

        # Qualitative: look at player conversations and final answers
        texts: List[str] = []
        for p in players:
            for msg in p.get("conversation", []):
                content = msg.get("content")
                if isinstance(content, str):
                    texts.append(content)
        for st in game_states:
            ans = st.get("player_complete_answer")
            if isinstance(ans, str):
                texts.append(ans)

        scripts_counter: Counter = Counter()
        has_code_switching = False
        politeness_total: Counter = Counter()
        cultural_total: Counter = Counter()
        stereotype_total: Counter = Counter()
        notable_examples: List[Dict[str, Any]] = []

        for t in texts:
            inc = analyse_text_incidents(t)
            scripts_counter.update(inc["scripts"])
            if inc["code_switching"]:
                has_code_switching = True
            for w in inc["politeness_markers"]:
                politeness_total[w] += 1
            for w in inc["cultural_markers"]:
                cultural_total[w] += 1
            for w in inc["stereotype_markers"]:
                stereotype_total[w] += 1

            if (
                inc["code_switching"]
                or inc["cultural_markers"]
                or inc["stereotype_markers"]
            ):
                # Keep a short sample (truncated)
                snippet = t
                if len(snippet) > 400:
                    snippet = snippet[:400] + "..."
                notable_examples.append(
                    {
                        "incident": inc,
                        "snippet": snippet,
                    }
                )

        qualitative = {
            "model1": model1,
            "model2": model2,
            "social_behavior": social_label,
            "log_dir": str(log_dir),
            "scripts_used": dict(scripts_counter),
            "code_switching": has_code_switching,
            "politeness_markers": {k: int(v) for k, v in politeness_total.items()},
            "cultural_markers": {k: int(v) for k, v in cultural_total.items()},
            "stereotype_markers": {k: int(v) for k, v in stereotype_total.items()},
            "notable_examples": notable_examples[:5],  # limit per run
        }

        return quantitative, qualitative

    except Exception as e:
        print(f"[SKIP] Problem parsing {game_state_path}: {e}")
        return None, None


def aggregate_results(
    quantitative_results: List[Dict[str, Any]],
    qualitative_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Aggregate per-run data into combo-level statistics."""
    combo_stats: Dict[Tuple[str, str, str], Dict[str, Any]] = defaultdict(
        lambda: {
            "games": 0,
            "draws": 0,
            "wins_player2": 0,
            "losses_player2": 0,
            "payoffs_player2": [],
            "offer_values_player2": [],
        }
    )

    qualitative_by_combo: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(
        list
    )

    for q in quantitative_results:
        key = (q["model1"], q["model2"], q["social_behavior"])
        stats = combo_stats[key]
        stats["games"] += 1
        if q["is_draw"]:
            stats["draws"] += 1
        else:
            if q["winner"] == "player2":
                stats["wins_player2"] += 1
            elif q["winner"] == "player1":
                stats["losses_player2"] += 1

        # Player 2 payoff / offer value (index 1)
        if isinstance(q.get("payoffs"), list) and len(q["payoffs"]) >= 2:
            stats["payoffs_player2"].append(q["payoffs"][1])
        if isinstance(q.get("offer_values"), list) and len(q["offer_values"]) >= 2:
            stats["offer_values_player2"].append(q["offer_values"][1])

    for qual in qualitative_results:
        key = (qual["model1"], qual["model2"], qual["social_behavior"])
        qualitative_by_combo[key].append(qual)

    # Build final summary structure
    combos_summary: Dict[str, Any] = {}
    all_models = set()
    all_behaviors = set()

    for (m1, m2, beh), stats in combo_stats.items():
        games = stats["games"]
        draws = stats["draws"]
        decisive_non_draw = games - draws

        if decisive_non_draw > 0:
            win_rate_p2 = stats["wins_player2"] / decisive_non_draw
        else:
            win_rate_p2 = None

        draw_rate = draws / games if games > 0 else None

        payoffs_arr = np.array(stats["payoffs_player2"], dtype=float) if stats[
            "payoffs_player2"
        ] else np.array([])
        offers_arr = np.array(
            stats["offer_values_player2"], dtype=float
        ) if stats["offer_values_player2"] else np.array([])

        combo_key = f"{m1}__vs__{m2}__{beh}"

        combos_summary[combo_key] = {
            "model1": m1,
            "model2": m2,
            "social_behavior": beh,
            "games": games,
            "draws": draws,
            "wins_player2": stats["wins_player2"],
            "losses_player2": stats["losses_player2"],
            "win_rate_player2_excl_draws": win_rate_p2,
            "draw_rate": draw_rate,
            "payoff_player2": {
                "average": float(payoffs_arr.mean()) if payoffs_arr.size > 0 else None,
                "values": stats["payoffs_player2"],
            },
            "initial_offer_value_player2": {
                "average": float(offers_arr.mean()) if offers_arr.size > 0 else None,
                "values": stats["offer_values_player2"],
            },
            "qualitative_runs": qualitative_by_combo.get((m1, m2, beh), []),
        }

        all_models.add(m1)
        all_models.add(m2)
        all_behaviors.add(beh)

    summary = {
        "meta": {
            "log_dir": str(LOG_DIR),
            "total_runs": len(quantitative_results),
            "models": sorted(all_models),
            "social_behaviors": sorted(all_behaviors),
        },
        "combos": combos_summary,
    }
    return summary


def create_heatmap_matrix(
    summary: Dict[str, Any],
    behavior: str = None,
    metric_key: str = "win_rate_player2_excl_draws",
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a matrix [model2 (rows) x model1 (cols)] for the given metric.
    """
    combos = summary["combos"]

    # Determine model set
    models = sorted(summary["meta"]["models"])
    idx = {m: i for i, m in enumerate(models)}

    mat = np.full((len(models), len(models)), np.nan)

    for combo in combos.values():
        if behavior and combo["social_behavior"] != behavior:
            continue
        m1 = combo["model1"]
        m2 = combo["model2"]
        if m1 not in idx or m2 not in idx:
            continue
        val = combo.get(metric_key)
        if val is None:
            continue
        i = idx[m2]  # Player 2 on rows
        j = idx[m1]  # Player 1 on cols
        mat[i, j] = val

    return mat, models


def plot_heatmaps(summary: Dict[str, Any]) -> None:
    """Generate heatmaps for win rate and payoff, overall and per behavior."""
    os.makedirs(FIGURES_DIR, exist_ok=True)

    behaviors = summary["meta"]["social_behaviors"]

    def _plot_for_behavior(behavior: str = None):
        label = behavior if behavior else "ALL"

        win_mat, models = create_heatmap_matrix(
            summary, behavior=behavior, metric_key="win_rate_player2_excl_draws"
        )
        payoff_mat, _ = create_heatmap_matrix(
            summary, behavior=behavior, metric_key="payoff_player2"
        )

        # payoff_player2 in summary is a dict; for heatmap we want averages
        if payoff_mat.dtype == object:
            # Rebuild payoff matrix using averages explicitly
            payoff_mat, _ = create_heatmap_matrix(
                summary,
                behavior=behavior,
                metric_key="payoff_player2",  # placeholder, we will fix below
            )

        # Instead, recompute payoff matrix with averages
        payoff_mat = np.full_like(win_mat, np.nan, dtype=float)
        combos = summary["combos"]
        idx = {m: i for i, m in enumerate(models)}
        for combo in combos.values():
            if behavior and combo["social_behavior"] != behavior:
                continue
            m1 = combo["model1"]
            m2 = combo["model2"]
            if m1 not in idx or m2 not in idx:
                continue
            avg_payoff = combo["payoff_player2"]["average"]
            if avg_payoff is None:
                continue
            i = idx[m2]
            j = idx[m1]
            payoff_mat[i, j] = avg_payoff

        # Figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        sns.heatmap(
            win_mat,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=models,
            yticklabels=models,
            cbar_kws={"label": "Win Rate (Player 2, excl. draws)"},
            ax=axes[0],
            vmin=0.0,
            vmax=1.0,
            mask=np.isnan(win_mat),
        )
        axes[0].set_title(
            "Win Rate\n(Player 2, decisive games only)", fontsize=11, fontweight="bold"
        )
        axes[0].set_xlabel("Player 1", fontsize=10)
        axes[0].set_ylabel("Player 2", fontsize=10)

        if not np.all(np.isnan(payoff_mat)):
            pmin = float(np.nanmin(payoff_mat))
            pmax = float(np.nanmax(payoff_mat))
        else:
            pmin, pmax = -5.0, 5.0

        sns.heatmap(
            payoff_mat,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            xticklabels=models,
            yticklabels=models,
            cbar_kws={"label": "Average Payoff (Player 2)"},
            ax=axes[1],
            vmin=pmin,
            vmax=pmax,
            mask=np.isnan(payoff_mat),
        )
        axes[1].set_title(
            "Payoff\n(Player 2 average payoff in all games)",
            fontsize=11,
            fontweight="bold",
        )
        axes[1].set_xlabel("Player 1", fontsize=10)
        axes[1].set_ylabel("Player 2", fontsize=10)

        fig.suptitle(
            f"Resource Exchange Game Performance Metrics\nSocial Behavior: {label}",
            fontsize=13,
            fontweight="bold",
        )
        fig.tight_layout()

        fname = f"trading_heatmaps_{label.replace(' ', '_')}.png"
        out_path = FIGURES_DIR / fname
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved heatmaps to: {out_path}")

    # Overall heatmaps
    _plot_for_behavior(None)
    # Per-behavior heatmaps
    for beh in behaviors:
        _plot_for_behavior(beh)


def main():
    if not LOG_DIR.exists():
        print(f"Log directory not found: {LOG_DIR}")
        sys.exit(1)

    quantitative_results: List[Dict[str, Any]] = []
    qualitative_results: List[Dict[str, Any]] = []

    print(f"Scanning log directory: {LOG_DIR}")
    for entry in sorted(LOG_DIR.iterdir()):
        if not entry.is_dir():
            continue
        game_state_path = entry / "game_state.json"
        if not game_state_path.exists():
            continue

        q, qual = extract_run_data(entry)
        if q is None:
            continue
        quantitative_results.append(q)
        if qual is not None:
            qualitative_results.append(qual)

    print(f"Loaded {len(quantitative_results)} valid runs")

    if not quantitative_results:
        print("No valid runs found; exiting.")
        sys.exit(0)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    summary = aggregate_results(quantitative_results, qualitative_results)

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {SUMMARY_PATH}")

    # Generate heatmaps
    plot_heatmaps(summary)


if __name__ == "__main__":
    main()


