#!/usr/bin/env python3
"""
Complete analysis pipeline for ultimatum game experiment (Punjabi behavior).
Steps:
1. Backfill all_results.json with real resource values from game_state.json files
2. Save updated all_results.json
3. Compute per-pair statistics
4. Generate visualizations
5. Print summary table
"""

import json
import glob
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
BASE_DIR = Path("/Users/rakshitsakhuja/personal/LLMs-NegotiationResearch/NegotiationArena-main")
LOGS_DIR = BASE_DIR / ".logs" / "ultimatum_prompt_native"
ALL_RESULTS_PATH = LOGS_DIR / "all_results.json"
ANALYSIS_DIR = LOGS_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["GPT-4o", "GPT-3.5", "Claude-3-Haiku", "Claude-3.5-Haiku"]
BEHAVIOR = "Punjabi"

# ─────────────────────────────────────────────────────────────
# Helper: effective payoff = sum of positive values in resource dict
# ─────────────────────────────────────────────────────────────
def effective_payoff(resource_value: dict) -> float:
    return sum(v for v in resource_value.values() if v > 0)


# ─────────────────────────────────────────────────────────────
# Helper: find latest subdirectory in a game dir
# ─────────────────────────────────────────────────────────────
def latest_subdir(game_dir: Path) -> Path | None:
    subdirs = sorted([d for d in game_dir.iterdir() if d.is_dir()])
    return subdirs[-1] if subdirs else None


# ─────────────────────────────────────────────────────────────
# Helper: parse game_state.json -> (final_response, p1_payoff, p2_payoff)
# ─────────────────────────────────────────────────────────────
def parse_game_state(gs_path: Path):
    with open(gs_path) as f:
        data = json.load(f)

    states = data.get("game_state", [])
    end_states = [s for s in states if s.get("current_iteration") == "END"]

    if not end_states:
        return "NONE", 0.0, 0.0

    summary = end_states[-1]["summary"]
    final_response = summary.get("final_response", "NONE")
    final_resources = summary.get("final_resources", [])

    if len(final_resources) < 2:
        return final_response, 0.0, 0.0

    p1_payoff = effective_payoff(final_resources[0]["_value"])
    p2_payoff = effective_payoff(final_resources[1]["_value"])

    return final_response, p1_payoff, p2_payoff


# ─────────────────────────────────────────────────────────────
# Step 1 & 2: Backfill all_results.json
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1-2: Backfilling all_results.json")
print("=" * 60)

with open(ALL_RESULTS_PATH) as f:
    all_results = json.load(f)

punjabi_results = [r for r in all_results if r.get("behavior") == BEHAVIOR]
print(f"Total results in file : {len(all_results)}")
print(f"Punjabi results       : {len(punjabi_results)}")

backfilled = 0
missing = 0

for r in all_results:
    if r.get("behavior") != BEHAVIOR:
        continue

    model1 = r["model1"]
    model2 = r["model2"]
    iteration = r["iteration"]
    game_dir_name = f"{model1}_vs_{model2}_{BEHAVIOR}_iter_{iteration}"
    game_dir = LOGS_DIR / game_dir_name

    if not game_dir.exists():
        print(f"  [MISSING DIR] {game_dir_name}")
        r["final_response"] = "NONE"
        r["player1_final_resources"] = 0.0
        r["player2_final_resources"] = 0.0
        missing += 1
        continue

    subdir = latest_subdir(game_dir)
    if subdir is None:
        print(f"  [NO SUBDIR]   {game_dir_name}")
        r["final_response"] = "NONE"
        r["player1_final_resources"] = 0.0
        r["player2_final_resources"] = 0.0
        missing += 1
        continue

    gs_path = subdir / "game_state.json"
    if not gs_path.exists():
        print(f"  [NO GS FILE]  {game_dir_name}/{subdir.name}")
        r["final_response"] = "NONE"
        r["player1_final_resources"] = 0.0
        r["player2_final_resources"] = 0.0
        missing += 1
        continue

    final_response, p1, p2 = parse_game_state(gs_path)
    r["final_response"] = final_response
    r["player1_final_resources"] = p1
    r["player2_final_resources"] = p2
    backfilled += 1

print(f"\nBackfilled: {backfilled}  |  Missing/error: {missing}")

# Save updated all_results.json
with open(ALL_RESULTS_PATH, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"Saved updated all_results.json -> {ALL_RESULTS_PATH}")


# ─────────────────────────────────────────────────────────────
# Step 3: Compute statistics
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Computing statistics")
print("=" * 60)

# Build DataFrame from Punjabi results
rows = []
for r in all_results:
    if r.get("behavior") != BEHAVIOR:
        continue
    rows.append({
        "model1": r["model1"],
        "model2": r["model2"],
        "iteration": r["iteration"],
        "final_response": r.get("final_response", "NONE"),
        "p1_payoff": r.get("player1_final_resources", 0.0),
        "p2_payoff": r.get("player2_final_resources", 0.0),
    })

df = pd.DataFrame(rows)
print(f"DataFrame shape: {df.shape}")
print(f"\nResponse distribution:\n{df['final_response'].value_counts()}")

# Per-pair statistics
stats_rows = []
for m1 in MODELS:
    for m2 in MODELS:
        sub = df[(df["model1"] == m1) & (df["model2"] == m2)]
        n = len(sub)
        if n == 0:
            stats_rows.append({
                "model1": m1, "model2": m2,
                "n": 0,
                "acceptance_rate": np.nan,
                "rejection_rate": np.nan,
                "timeout_rate": np.nan,
                "avg_p1_payoff_accept": np.nan,
                "avg_p2_payoff_accept": np.nan,
                "avg_p1_payoff_overall": np.nan,
                "avg_p2_payoff_overall": np.nan,
            })
            continue

        accepted = sub[sub["final_response"] == "ACCEPT"]
        rejected = sub[sub["final_response"] == "REJECT"]
        timed_out = sub[~sub["final_response"].isin(["ACCEPT", "REJECT"])]

        stats_rows.append({
            "model1": m1,
            "model2": m2,
            "n": n,
            "acceptance_rate": len(accepted) / n,
            "rejection_rate": len(rejected) / n,
            "timeout_rate": len(timed_out) / n,
            "avg_p1_payoff_accept": accepted["p1_payoff"].mean() if len(accepted) > 0 else np.nan,
            "avg_p2_payoff_accept": accepted["p2_payoff"].mean() if len(accepted) > 0 else np.nan,
            "avg_p1_payoff_overall": sub["p1_payoff"].mean(),
            "avg_p2_payoff_overall": sub["p2_payoff"].mean(),
        })

stats_df = pd.DataFrame(stats_rows)
stats_df.to_csv(ANALYSIS_DIR / "per_pair_stats.csv", index=False)
print(f"\nPer-pair stats saved -> {ANALYSIS_DIR / 'per_pair_stats.csv'}")


# ─────────────────────────────────────────────────────────────
# Helper: pivot to matrix
# ─────────────────────────────────────────────────────────────
def make_matrix(stats_df, value_col):
    pivot = stats_df.pivot(index="model1", columns="model2", values=value_col)
    # Reindex to consistent order
    pivot = pivot.reindex(index=MODELS, columns=MODELS)
    return pivot


# ─────────────────────────────────────────────────────────────
# Step 4: Visualizations
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Creating visualizations")
print("=" * 60)

MODEL_SHORT = {
    "GPT-4o": "GPT-4o",
    "GPT-3.5": "GPT-3.5",
    "Claude-3-Haiku": "C3-Haiku",
    "Claude-3.5-Haiku": "C3.5-Haiku",
}
SHORT_MODELS = [MODEL_SHORT[m] for m in MODELS]


def shorten_index(matrix):
    matrix = matrix.copy()
    matrix.index = [MODEL_SHORT.get(i, i) for i in matrix.index]
    matrix.columns = [MODEL_SHORT.get(c, c) for c in matrix.columns]
    return matrix


# ── 4a: Heatmap – acceptance rate ──────────────────────────
acc_matrix = shorten_index(make_matrix(stats_df, "acceptance_rate"))

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    acc_matrix.astype(float),
    annot=True, fmt=".2f", cmap="YlGn",
    vmin=0, vmax=1,
    linewidths=0.5, ax=ax,
    cbar_kws={"label": "Acceptance Rate"},
)
ax.set_title("Acceptance Rate\n(rows = proposer/Player1, cols = responder/Player2)", fontsize=12)
ax.set_xlabel("Player 2 (Responder)")
ax.set_ylabel("Player 1 (Proposer)")
plt.tight_layout()
plt.savefig(ANALYSIS_DIR / "heatmap_acceptance_rate.png", dpi=150)
plt.close()
print("  Saved heatmap_acceptance_rate.png")

# ── 4b: Heatmap – avg p1 payoff overall ────────────────────
p1_matrix = shorten_index(make_matrix(stats_df, "avg_p1_payoff_overall"))

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    p1_matrix.astype(float),
    annot=True, fmt=".1f", cmap="Blues",
    vmin=0, vmax=100,
    linewidths=0.5, ax=ax,
    cbar_kws={"label": "Avg Payoff"},
)
ax.set_title("Average Player 1 (Proposer) Payoff Overall\n(rows = proposer, cols = responder)", fontsize=12)
ax.set_xlabel("Player 2 (Responder)")
ax.set_ylabel("Player 1 (Proposer)")
plt.tight_layout()
plt.savefig(ANALYSIS_DIR / "heatmap_p1_payoff_overall.png", dpi=150)
plt.close()
print("  Saved heatmap_p1_payoff_overall.png")

# ── 4c: Heatmap – avg p2 payoff overall ────────────────────
p2_matrix = shorten_index(make_matrix(stats_df, "avg_p2_payoff_overall"))

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    p2_matrix.astype(float),
    annot=True, fmt=".1f", cmap="Oranges",
    vmin=0, vmax=100,
    linewidths=0.5, ax=ax,
    cbar_kws={"label": "Avg Payoff"},
)
ax.set_title("Average Player 2 (Responder) Payoff Overall\n(rows = proposer, cols = responder)", fontsize=12)
ax.set_xlabel("Player 2 (Responder)")
ax.set_ylabel("Player 1 (Proposer)")
plt.tight_layout()
plt.savefig(ANALYSIS_DIR / "heatmap_p2_payoff_overall.png", dpi=150)
plt.close()
print("  Saved heatmap_p2_payoff_overall.png")

# ── 4d: Bar chart – per-model acceptance rate as proposer vs responder ──
proposer_acc = (
    df[df["final_response"] == "ACCEPT"]
    .groupby("model1").size()
    / df.groupby("model1").size()
).reindex(MODELS).rename("As Proposer (P1)")

responder_acc = (
    df[df["final_response"] == "ACCEPT"]
    .groupby("model2").size()
    / df.groupby("model2").size()
).reindex(MODELS).rename("As Responder (P2)")

acc_role_df = pd.concat([proposer_acc, responder_acc], axis=1)
acc_role_df.index = SHORT_MODELS

fig, ax = plt.subplots(figsize=(8, 5))
acc_role_df.plot(kind="bar", ax=ax, color=["steelblue", "coral"], width=0.6)
ax.set_title("Acceptance Rate per Model by Role (Punjabi)", fontsize=13)
ax.set_ylabel("Acceptance Rate")
ax.set_xlabel("Model")
ax.set_ylim(0, 1.15)
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
ax.legend(loc="upper right")
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=3, fontsize=9)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(ANALYSIS_DIR / "bar_acceptance_by_role.png", dpi=150)
plt.close()
print("  Saved bar_acceptance_by_role.png")

# ── 4e: Bar chart – avg payoff per model as proposer and responder ──
proposer_payoff = df.groupby("model1")["p1_payoff"].mean().reindex(MODELS).rename("Proposer Payoff")
responder_payoff = df.groupby("model2")["p2_payoff"].mean().reindex(MODELS).rename("Responder Payoff")

payoff_role_df = pd.concat([proposer_payoff, responder_payoff], axis=1)
payoff_role_df.index = SHORT_MODELS

fig, ax = plt.subplots(figsize=(8, 5))
payoff_role_df.plot(kind="bar", ax=ax, color=["steelblue", "coral"], width=0.6)
ax.set_title("Average Payoff per Model by Role (Punjabi)", fontsize=13)
ax.set_ylabel("Average Payoff (Dollars)")
ax.set_xlabel("Model")
ax.set_ylim(0, 125)
ax.axhline(50, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="Equal split (50)")
ax.legend(loc="upper right")
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f", padding=3, fontsize=9)
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(ANALYSIS_DIR / "bar_payoff_by_role.png", dpi=150)
plt.close()
print("  Saved bar_payoff_by_role.png")


# ─────────────────────────────────────────────────────────────
# Step 5: Print summary table
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Summary Table")
print("=" * 60)

summary_cols = [
    "model1", "model2", "n",
    "acceptance_rate", "rejection_rate", "timeout_rate",
    "avg_p1_payoff_accept", "avg_p2_payoff_accept",
    "avg_p1_payoff_overall", "avg_p2_payoff_overall",
]

print_df = stats_df[summary_cols].copy()
print_df = print_df[print_df["n"] > 0].sort_values(["model1", "model2"])
print_df["model1_s"] = print_df["model1"].map(MODEL_SHORT)
print_df["model2_s"] = print_df["model2"].map(MODEL_SHORT)

# Format floats
float_cols = [
    "acceptance_rate", "rejection_rate", "timeout_rate",
    "avg_p1_payoff_accept", "avg_p2_payoff_accept",
    "avg_p1_payoff_overall", "avg_p2_payoff_overall",
]
for col in float_cols:
    print_df[col] = print_df[col].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")

display = print_df[[
    "model1_s", "model2_s", "n",
    "acceptance_rate", "rejection_rate", "timeout_rate",
    "avg_p1_payoff_accept", "avg_p2_payoff_accept",
    "avg_p1_payoff_overall", "avg_p2_payoff_overall",
]].rename(columns={
    "model1_s": "Proposer",
    "model2_s": "Responder",
    "n": "N",
    "acceptance_rate": "AccRate",
    "rejection_rate": "RejRate",
    "timeout_rate": "TMORate",
    "avg_p1_payoff_accept": "P1$|Acc",
    "avg_p2_payoff_accept": "P2$|Acc",
    "avg_p1_payoff_overall": "P1$All",
    "avg_p2_payoff_overall": "P2$All",
})

pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 160)
pd.set_option("display.max_colwidth", 15)
print(display.to_string(index=False))

# Also print aggregate per-model stats
print("\n" + "─" * 60)
print("Per-Model Aggregate Stats (Punjabi)\n")

agg_rows = []
for m in MODELS:
    as_prop = df[df["model1"] == m]
    as_resp = df[df["model2"] == m]
    agg_rows.append({
        "Model": MODEL_SHORT[m],
        "As Proposer - AccRate": f"{(as_prop['final_response']=='ACCEPT').mean():.2f}",
        "As Proposer - Avg$": f"{as_prop['p1_payoff'].mean():.1f}",
        "As Responder - AccRate": f"{(as_resp['final_response']=='ACCEPT').mean():.2f}",
        "As Responder - Avg$": f"{as_resp['p2_payoff'].mean():.1f}",
    })

agg_df = pd.DataFrame(agg_rows)
print(agg_df.to_string(index=False))

print("\n" + "=" * 60)
print(f"Analysis complete. Outputs in: {ANALYSIS_DIR}")
print("=" * 60)
