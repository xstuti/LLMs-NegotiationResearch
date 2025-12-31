import pandas as pd
import numpy as np

# ==============================
# Load data
# ==============================
df = pd.read_csv("behavior_metrics_summary.csv")

BEHAVIOR_COL = "behavior"

METRICS = {
    "acceptance_rate_mean": "Acceptance Rate",
    "avg_negotiation_rounds": "Negotiation Rounds",
    "seller_advantage_mean": "Seller Advantage",
    "buyer_advantage_mean": "Buyer Advantage",
    "win_rate_player1": "Player-1 Win Rate"
}

N_PERMUTATIONS = 10000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ==============================
# Permutation ANOVA
# ==============================
def permutation_anova(df, metric, group_col, n_perm=10000):
    groups = df[group_col].unique()
    group_means = df.groupby(group_col)[metric].mean()

    grand_mean = df[metric].mean()

    ss_between = sum(
        len(df[df[group_col] == g]) * (group_means[g] - grand_mean) ** 2
        for g in groups
    )

    permuted_stats = []
    values = df[metric].values.copy()

    for _ in range(n_perm):
        np.random.shuffle(values)
        perm_df = df.copy()
        perm_df[metric] = values

        perm_means = perm_df.groupby(group_col)[metric].mean()
        perm_grand = perm_df[metric].mean()

        ss_perm = sum(
            len(perm_df[perm_df[group_col] == g]) * (perm_means[g] - perm_grand) ** 2
            for g in groups
        )
        permuted_stats.append(ss_perm)

    p_value = np.mean(np.array(permuted_stats) >= ss_between)

    # Effect size: eta-squared
    ss_total = np.sum((df[metric] - grand_mean) ** 2)
    eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

    return {
        "SS_between": ss_between,
        "eta_squared": eta_sq,
        "p_value": p_value
    }

# ==============================
# Run tests
# ==============================
results = []

for metric, pretty_name in METRICS.items():
    stats = permutation_anova(df, metric, BEHAVIOR_COL, N_PERMUTATIONS)

    results.append({
        "Metric": pretty_name,
        "SS_between": round(stats["SS_between"], 4),
        "Eta_squared": round(stats["eta_squared"], 4),
        "p_value": round(stats["p_value"], 5),
        "Significant (p<0.05)": stats["p_value"] < 0.05
    })

results_df = pd.DataFrame(results)

# ==============================
# Output
# ==============================
print("\n=== Permutation-Based ANOVA Results ===")
print(results_df.to_string(index=False))

results_df.to_csv("behavior_statistical_tests.csv", index=False)

# ==============================
# Descriptive table (paper-ready)
# ==============================
desc_cols = [
    "behavior",
    "total_games",
    "acceptance_rate_mean",
    "avg_negotiation_rounds",
    "seller_advantage_mean",
    "buyer_advantage_mean",
    "win_rate_player1"
]

desc_table = df[desc_cols].round(4)
desc_table.to_csv("behavior_descriptive_table.csv", index=False)

print("\nSaved:")
print(" - behavior_statistical_tests.csv")
print(" - behavior_descriptive_table.csv")