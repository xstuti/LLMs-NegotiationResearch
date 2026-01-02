import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f_oneway, levene
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# Set styling
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ============================================================================
# LOAD DATA
# ============================================================================
df = pd.read_csv('behavior_metrics_summary.csv')  # Replace with your actual filename

print("="*80)
print("DATA SUMMARY")
print("="*80)
print(df)
print("\n")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def cohens_d(mean1, mean2, std1, std2, n1, n2):
    """
    Calculate Cohen's d effect size for two groups
    Uses pooled standard deviation
    """
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return np.nan
    return (mean1 - mean2) / pooled_std

def interpret_cohens_d(d):
    """Interpret Cohen's d magnitude"""
    abs_d = abs(d)
    if pd.isna(d):
        return "undefined"
    elif abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"

def interpret_eta_squared(eta_sq):
    """Interpret eta-squared magnitude"""
    if eta_sq < 0.01:
        return "negligible"
    elif eta_sq < 0.06:
        return "small"
    elif eta_sq < 0.14:
        return "medium"
    else:
        return "large"

def reconstruct_data(mean, std, n, is_proportion=False):
    """
    Reconstruct approximate individual data points from summary statistics
    Assumes normal distribution
    """
    if is_proportion and std == 0:
        # For proportions with std=0 (e.g., 100% acceptance in Hindi)
        # All values are identical
        return np.full(n, mean)
    
    # Generate from normal distribution
    data = np.random.normal(mean, std, n)
    
    # For proportions, clip to [0, 1]
    if is_proportion:
        data = np.clip(data, 0, 1)
    
    return data

# ============================================================================
# METRICS TO ANALYZE
# ============================================================================

# Define metrics and whether they are proportions
metrics = {
    'acceptance_rate_mean': {
        'std_col': 'acceptance_rate_std',
        'is_proportion': True,
        'description': 'Acceptance Rate'
    },
    'avg_negotiation_rounds': {
        'std_col': 'negotiation_rounds_std',
        'is_proportion': False,
        'description': 'Average Negotiation Rounds'
    },
    'seller_advantage_mean': {
        'std_col': 'seller_advantage_std',
        'is_proportion': False,
        'description': 'Seller Advantage'
    },
    'buyer_advantage_mean': {
        'std_col': 'buyer_advantage_std',
        'is_proportion': False,
        'description': 'Buyer Advantage'
    },
    'win_rate_player1': {
        'std_col': None,  # No std provided for win rate
        'is_proportion': True,
        'description': 'Win Rate (Player 1)'
    },
    'draw_rate': {
        'std_col': None,  # No std provided for draw rate
        'is_proportion': True,
        'description': 'Draw Rate'
    }
}

languages = df['behavior'].tolist()
n_games = df['total_games'].tolist()

# ============================================================================
# RUN ANALYSIS FOR EACH METRIC
# ============================================================================

results_summary = []
all_pairwise_results = []

for metric, info in metrics.items():
    print("="*80)
    print(f"ANALYSIS: {info['description']}")
    print("="*80)
    
    means = df[metric].tolist()
    std_col = info['std_col']
    
    # Skip if metric not in dataframe
    if metric not in df.columns:
        print(f"⚠️  Metric {metric} not found in data, skipping...\n")
        continue
    
    # Get standard deviations (use 0 if not provided)
    if std_col and std_col in df.columns:
        stds = df[std_col].tolist()
    else:
        stds = [0] * len(means)
    
    # Print descriptive statistics
    print("\nDescriptive Statistics:")
    print("-" * 80)
    for lang, mean, std, n in zip(languages, means, stds, n_games):
        print(f"{lang:15s} | Mean: {mean:8.4f} | Std: {std:8.4f} | N: {n:4d}")
    
    # ========================================================================
    # 1. LEVENE'S TEST (Test for equal variances)
    # ========================================================================
    print("\n1. Levene's Test for Equal Variances")
    print("-" * 80)
    
    # Reconstruct data for each language
    reconstructed_groups = []
    for mean, std, n, lang in zip(means, stds, n_games, languages):
        data = reconstruct_data(mean, std, n, info['is_proportion'])
        reconstructed_groups.append(data)
    
    levene_stat, levene_p = levene(*reconstructed_groups)
    print(f"Levene statistic: {levene_stat:.4f}")
    print(f"p-value: {levene_p:.4f}")
    
    if levene_p < 0.05:
        print("⚠️  WARNING: Unequal variances detected (p < 0.05)")
        print("   Consider using Welch's ANOVA or non-parametric tests")
        use_welch = True
    else:
        print("✓ Variances are approximately equal (p >= 0.05)")
        use_welch = False
    
    # ========================================================================
    # 2. ONE-WAY ANOVA / WELCH'S ANOVA
    # ========================================================================
    print("\n2. One-Way ANOVA")
    print("-" * 80)
    
    f_stat, anova_p = f_oneway(*reconstructed_groups)
    
    # Calculate eta-squared (effect size for ANOVA)
    # η² = SS_between / SS_total
    all_data = np.concatenate(reconstructed_groups)
    grand_mean = np.mean(all_data)
    
    ss_between = sum([len(group) * (np.mean(group) - grand_mean)**2 
                      for group in reconstructed_groups])
    ss_total = sum((all_data - grand_mean)**2)
    eta_squared = ss_between / ss_total if ss_total > 0 else 0
    
    print(f"F-statistic: {f_stat:.4f}")
    print(f"p-value: {anova_p:.6f}")
    print(f"Eta-squared (η²): {eta_squared:.4f} ({interpret_eta_squared(eta_squared)} effect)")
    
    if anova_p < 0.05:
        print("✓ SIGNIFICANT: At least one language differs (p < 0.05)")
        is_significant = True
    else:
        print("✗ NOT SIGNIFICANT: No significant differences detected (p >= 0.05)")
        is_significant = False
    
    # Store result
    results_summary.append({
        'Metric': info['description'],
        'F-statistic': f"{f_stat:.4f}",
        'p-value': f"{anova_p:.6f}",
        'Eta²': f"{eta_squared:.4f}",
        'Effect Size': interpret_eta_squared(eta_squared),
        'Significant': 'Yes' if is_significant else 'No'
    })
    
    # ========================================================================
    # 3. POST-HOC PAIRWISE COMPARISONS (if ANOVA significant)
    # ========================================================================
    if is_significant:
        print("\n3. Post-hoc Pairwise Comparisons (Tukey HSD)")
        print("-" * 80)
        
        # Prepare data for Tukey HSD
        data_for_tukey = []
        groups_for_tukey = []
        
        for lang, group in zip(languages, reconstructed_groups):
            data_for_tukey.extend(group)
            groups_for_tukey.extend([lang] * len(group))
        
        # Run Tukey HSD
        tukey = pairwise_tukeyhsd(endog=data_for_tukey, 
                                  groups=groups_for_tukey, 
                                  alpha=0.05)
        print(tukey)
        
        # Calculate Cohen's d for each pair
        print("\n4. Effect Sizes (Cohen's d) for Significant Pairs")
        print("-" * 80)
        print(f"{'Comparison':<25} {'Mean Diff':>10} {'Cohens d':>10} {'Magnitude':>12} {'p-value':>10}")
        print("-" * 80)
        
        tukey_results = tukey.summary().data[1:]  # Skip header
        
        for i, (lang1, lang2) in enumerate(combinations(languages, 2)):
            idx1 = languages.index(lang1)
            idx2 = languages.index(lang2)
            
            mean1, mean2 = means[idx1], means[idx2]
            std1, std2 = stds[idx1], stds[idx2]
            n1, n2 = n_games[idx1], n_games[idx2]
            
            mean_diff = mean1 - mean2
            d = cohens_d(mean1, mean2, std1, std2, n1, n2)
            
            # Find corresponding p-value from Tukey results
            tukey_p = None
            for row in tukey_results:
                if (row[0] == lang1 and row[1] == lang2) or \
                   (row[0] == lang2 and row[1] == lang1):
                    tukey_p = row[3]  # p-value is in 4th column
                    break
            
            comparison = f"{lang1} vs {lang2}"
            print(f"{comparison:<25} {mean_diff:>10.4f} {d:>10.3f} {interpret_cohens_d(d):>12} {tukey_p if tukey_p else 'N/A':>10}")
            
            # Store for output
            all_pairwise_results.append({
                'Metric': info['description'],
                'Comparison': comparison,
                'Mean_Diff': mean_diff,
                'Cohens_d': d,
                'Effect_Size': interpret_cohens_d(d),
                'p_value': tukey_p if tukey_p else np.nan
            })
    
    print("\n")

# ============================================================================
# SUMMARY TABLE
# ============================================================================
print("="*80)
print("SUMMARY: ANOVA RESULTS FOR ALL METRICS")
print("="*80)

summary_df = pd.DataFrame(results_summary)
print(summary_df.to_string(index=False))
print("\n")

# Save summary
summary_df.to_csv('anova_summary.csv', index=False)
print("✓ Saved summary to 'anova_summary.csv'\n")

# ============================================================================
# PAIRWISE COMPARISONS TABLE
# ============================================================================
if all_pairwise_results:
    print("="*80)
    print("SUMMARY: SIGNIFICANT PAIRWISE COMPARISONS")
    print("="*80)
    
    pairwise_df = pd.DataFrame(all_pairwise_results)
    
    # Filter to only significant pairs (p < 0.05)
    pairwise_df_sig = pairwise_df[pairwise_df['p_value'] < 0.05].copy()
    
    if len(pairwise_df_sig) > 0:
        print(pairwise_df_sig.to_string(index=False))
        pairwise_df_sig.to_csv('pairwise_comparisons_significant.csv', index=False)
        print("\n✓ Saved significant pairs to 'pairwise_comparisons_significant.csv'\n")
    else:
        print("No significant pairwise differences found.\n")
    
    # Save all pairwise comparisons
    pairwise_df.to_csv('pairwise_comparisons_all.csv', index=False)
    print("✓ Saved all pairwise comparisons to 'pairwise_comparisons_all.csv'\n")

# ============================================================================
# VISUALIZATION: COMPARISON ACROSS LANGUAGES
# ============================================================================
print("="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for idx, (metric, info) in enumerate(metrics.items()):
    if metric not in df.columns:
        continue
    
    ax = axes[idx]
    
    means = df[metric].tolist()
    std_col = info['std_col']
    
    if std_col and std_col in df.columns:
        stds = df[std_col].tolist()
    else:
        stds = [0] * len(means)
    
    # Bar plot with error bars
    x_pos = np.arange(len(languages))
    bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                  alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    
    ax.set_xlabel('Language', fontsize=12, fontweight='bold')
    ax.set_ylabel(info['description'], fontsize=12, fontweight='bold')
    ax.set_title(info['description'], fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(languages, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add significance indicator if ANOVA was significant
    result = next((r for r in results_summary if r['Metric'] == info['description']), None)
    if result and result['Significant'] == 'Yes':
        ax.text(0.5, 0.95, '* Significant (p < 0.05)', 
                transform=ax.transAxes, fontsize=10, 
                verticalalignment='top', horizontalalignment='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

plt.tight_layout()
plt.savefig('language_comparison_all_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization to 'language_comparison_all_metrics.png'\n")

plt.show()

print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)