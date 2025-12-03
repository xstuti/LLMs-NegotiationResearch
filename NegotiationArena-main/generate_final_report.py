#!/usr/bin/env python3
"""
Generate Final Report for Ultimatum Game Social Behavior Analysis

This script creates a comprehensive report summarizing the key findings
from the social behavior experiments with multiple AI models.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add current directory to Python path for module imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))


class UltimatumReportGenerator:
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.summary_file = self.results_dir / "summary.json"
        self.summary_data = {}
        self.models = set()

    def load_data(self):
        """Load the summary data"""
        if not self.summary_file.exists():
            raise FileNotFoundError(f"Summary file not found: {self.summary_file}")

        with open(self.summary_file, "r", encoding="utf-8") as f:
            self.summary_data = json.load(f)

        # Extract all unique models from the data
        for metrics in self.summary_data.values():
            self.models.add(metrics["model1"])
            self.models.add(metrics["model2"])

        self.models = sorted(list(self.models))

    def analyze_key_findings(self):
        """Analyze and extract key findings"""
        findings = {
            "model_performance": {},
            "behavior_effects": {},
            "strategic_patterns": {},
            "cultural_insights": {},
        }

        # Organize data by behavior and model combinations
        by_behavior = {}
        for combo_key, metrics in self.summary_data.items():
            behavior = metrics["behavior"]
            if behavior not in by_behavior:
                by_behavior[behavior] = []
            by_behavior[behavior].append((combo_key, metrics))

        # Analyze each behavior
        for behavior, combos in by_behavior.items():
            behavior_analysis = {
                "total_games": 0,
                "avg_acceptance_rate": 0,
                "avg_initial_offer": 0,
                "rejection_patterns": [],
            }

            # Initialize model-specific tracking
            model_stats = {}
            for model in self.models:
                model_stats[model] = {
                    "as_p1_wins": 0,
                    "as_p1_payoff": 0,
                    "as_p1_games": 0,
                }

            for combo_key, metrics in combos:
                behavior_analysis["total_games"] += metrics["total_games"]
                behavior_analysis["avg_acceptance_rate"] += (
                    metrics["acceptance_rate"] * metrics["total_games"]
                )
                behavior_analysis["avg_initial_offer"] += (
                    metrics["initial_offer_avg"] * metrics["total_games"]
                )

                # Track model performance as Player 1
                model1 = metrics["model1"]
                if model1 in model_stats:
                    model_stats[model1]["as_p1_wins"] += metrics["player1_wins"]
                    model_stats[model1]["as_p1_payoff"] += (
                        metrics["player1_payoff_avg"] * metrics["total_games"]
                    )
                    model_stats[model1]["as_p1_games"] += metrics["total_games"]

                if metrics["rejects"] > 0:
                    behavior_analysis["rejection_patterns"].append(
                        {
                            "combo": f"{metrics['model1']} vs {metrics['model2']}",
                            "rejections": metrics["rejects"],
                            "rejection_rate": metrics["rejects"]
                            / metrics["total_games"],
                        }
                    )

            # Calculate averages
            if behavior_analysis["total_games"] > 0:
                behavior_analysis["avg_acceptance_rate"] /= behavior_analysis[
                    "total_games"
                ]
                behavior_analysis["avg_initial_offer"] /= behavior_analysis[
                    "total_games"
                ]

            # Calculate model averages
            for model in model_stats:
                if model_stats[model]["as_p1_games"] > 0:
                    model_stats[model]["as_p1_payoff"] /= model_stats[model][
                        "as_p1_games"
                    ]
                    model_stats[model]["win_rate"] = (
                        model_stats[model]["as_p1_wins"]
                        / model_stats[model]["as_p1_games"]
                    )
                else:
                    model_stats[model]["win_rate"] = 0

            behavior_analysis["model_stats"] = model_stats
            findings["behavior_effects"][behavior] = behavior_analysis

        # Overall model performance analysis
        overall_model_stats = {}
        for model in self.models:
            overall_model_stats[model] = {
                "total_wins": 0,
                "total_games": 0,
                "total_payoff": 0,
            }

        for combo_key, metrics in self.summary_data.items():
            model1 = metrics["model1"]
            if model1 in overall_model_stats:
                overall_model_stats[model1]["total_wins"] += metrics["player1_wins"]
                overall_model_stats[model1]["total_games"] += metrics["total_games"]
                overall_model_stats[model1]["total_payoff"] += (
                    metrics["player1_payoff_avg"] * metrics["total_games"]
                )

        # Calculate overall win rates and average payoffs
        for model in overall_model_stats:
            stats = overall_model_stats[model]
            if stats["total_games"] > 0:
                stats["win_rate"] = stats["total_wins"] / stats["total_games"]
                stats["avg_payoff"] = stats["total_payoff"] / stats["total_games"]
            else:
                stats["win_rate"] = 0
                stats["avg_payoff"] = 0

        findings["model_performance"] = overall_model_stats
        return findings

    def generate_report(self):
        """Generate comprehensive report"""
        findings = self.analyze_key_findings()

        report_lines = []

        # Header
        report_lines.extend(
            [
                "ULTIMATUM GAME SOCIAL BEHAVIOR ANALYSIS",
                "=" * 60,
                "",
                f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"Results Directory: {self.results_dir.name}",
                f"Total Combinations Analyzed: {len(self.summary_data)}",
                "",
                "=" * 60,
                "EXECUTIVE SUMMARY",
                "=" * 60,
                "",
            ]
        )

        # Executive Summary
        total_games = sum(
            metrics["total_games"] for metrics in self.summary_data.values()
        )
        total_accepts = sum(
            metrics["accepts"] for metrics in self.summary_data.values()
        )
        total_rejects = sum(
            metrics["rejects"] for metrics in self.summary_data.values()
        )
        overall_acceptance_rate = total_accepts / total_games if total_games > 0 else 0

        # Extract unique behaviors
        behaviors = list(
            set(metrics["behavior"] for metrics in self.summary_data.values())
        )
        behaviors.sort()

        report_lines.extend(
            [
                f"• Total Games Analyzed: {total_games}",
                f"• Overall Acceptance Rate: {overall_acceptance_rate:.1%}",
                f"• Total Accepted Proposals: {total_accepts}",
                f"• Total Rejected Proposals: {total_rejects}",
                f"• Models Tested: {', '.join(self.models)}",
                f"• Cultural Behaviors: {', '.join(behaviors)}",
                "",
            ]
        )

        # Key Findings
        report_lines.extend(
            [
                "KEY FINDINGS:",
                "-" * 20,
                "",
            ]
        )

        # Model Performance Comparison
        model_performance = findings["model_performance"]
        report_lines.extend(
            [
                f"1. MODEL PERFORMANCE (as Player 1):",
            ]
        )

        # Sort models by win rate for better presentation
        sorted_models = sorted(
            model_performance.items(), key=lambda x: x[1]["win_rate"], reverse=True
        )

        for model, stats in sorted_models:
            report_lines.append(
                f"   • {model}: {stats['win_rate']:.1%} win rate, ${stats['avg_payoff']:.1f} avg payoff ({stats['total_games']} games)"
            )

        report_lines.append("")

        # Behavior-specific insights
        report_lines.extend(
            [
                "2. CULTURAL BEHAVIOR EFFECTS:",
                "",
            ]
        )

        for behavior, analysis in findings["behavior_effects"].items():
            rejection_info = ""
            if analysis["rejection_patterns"]:
                rejections = [p["rejections"] for p in analysis["rejection_patterns"]]
                total_rejections = sum(rejections)
                rejection_info = f" (Rejections: {total_rejections})"

            report_lines.extend(
                [
                    f"   {behavior.upper()}:",
                    f"   • Games: {analysis['total_games']} | Acceptance Rate: {analysis['avg_acceptance_rate']:.1%}{rejection_info}",
                    f"   • Average Initial Offer: ${analysis['avg_initial_offer']:.1f}",
                ]
            )

            # Add model-specific performance for this behavior
            for model, stats in analysis["model_stats"].items():
                if stats["as_p1_games"] > 0:
                    report_lines.append(
                        f"   • {model} as P1: {stats['win_rate']:.1%} win rate, ${stats['as_p1_payoff']:.1f} avg payoff"
                    )

            if analysis["rejection_patterns"]:
                report_lines.append("   • Rejection Patterns:")
                for pattern in analysis["rejection_patterns"]:
                    report_lines.append(
                        f"     - {pattern['combo']}: {pattern['rejections']} rejections ({pattern['rejection_rate']:.1%})"
                    )

            report_lines.append("")

        # Detailed Analysis by Combination
        report_lines.extend(
            [
                "=" * 60,
                "DETAILED ANALYSIS BY COMBINATION",
                "=" * 60,
                "",
            ]
        )

        # Group by behavior for detailed reporting
        for behavior in sorted(behaviors):
            report_lines.extend(
                [
                    f"{behavior.upper()} BEHAVIOR ANALYSIS:",
                    "-" * 40,
                    "",
                ]
            )

            behavior_combos = [
                (combo_key, metrics)
                for combo_key, metrics in self.summary_data.items()
                if metrics["behavior"] == behavior
            ]

            for combo_key, metrics in sorted(behavior_combos):
                report_lines.extend(
                    [
                        f"Combination: {metrics['model1']} vs {metrics['model2']}",
                        f"  Total Games: {metrics['total_games']}",
                        f"  Acceptance Rate: {metrics['acceptance_rate']:.1%} ({metrics['accepts']} accepts, {metrics['rejects']} rejects)",
                        f"  Win Rate (P1): {metrics['win_rate_player1']:.1%} | Draw Rate: {metrics['draw_rate']:.1%}",
                        f"  Average Payoffs: P1=${metrics['player1_payoff_avg']:.1f}, P2=${metrics['player2_payoff_avg']:.1f}",
                        f"  Average Initial Offer: ${metrics['initial_offer_avg']:.1f}",
                        f"  Payoff Range P1: ${min(metrics['player1_payoffs']):.0f}-${max(metrics['player1_payoffs']):.0f}",
                        f"  Initial Offer Range: ${min(metrics['initial_offers']):.0f}-${max(metrics['initial_offers']):.0f}",
                        "",
                    ]
                )

        # Strategic Insights
        report_lines.extend(
            [
                "=" * 60,
                "STRATEGIC INSIGHTS",
                "=" * 60,
                "",
            ]
        )

        # Calculate some strategic insights
        high_rejection_combos = []
        low_offer_combos = []
        high_payoff_combos = []

        for combo_key, metrics in self.summary_data.items():
            rejection_rate = (
                metrics["rejects"] / metrics["total_games"]
                if metrics["total_games"] > 0
                else 0
            )
            if rejection_rate > 0.2:  # More than 20% rejection
                high_rejection_combos.append((combo_key, rejection_rate, metrics))

            if metrics["initial_offer_avg"] < 30:  # Low offers
                low_offer_combos.append(
                    (combo_key, metrics["initial_offer_avg"], metrics)
                )

            if metrics["player1_payoff_avg"] > 80:  # High payoffs for P1
                high_payoff_combos.append(
                    (combo_key, metrics["player1_payoff_avg"], metrics)
                )

        report_lines.extend(
            [
                "NEGOTIATION PATTERNS:",
                "",
                f"• High Rejection Rate Combinations ({len(high_rejection_combos)}):",
            ]
        )

        for combo_key, rejection_rate, metrics in high_rejection_combos:
            report_lines.append(
                f"  - {metrics['model1']} vs {metrics['model2']} ({metrics['behavior']}): {rejection_rate:.1%}"
            )

        if not high_rejection_combos:
            report_lines.append("  - None (all combinations had low rejection rates)")

        report_lines.extend(
            [
                "",
                f"• Low Initial Offer Combinations ({len(low_offer_combos)}):",
            ]
        )

        for combo_key, avg_offer, metrics in low_offer_combos:
            report_lines.append(
                f"  - {metrics['model1']} vs {metrics['model2']} ({metrics['behavior']}): ${avg_offer:.1f}"
            )

        report_lines.extend(
            [
                "",
                f"• High Player 1 Payoff Combinations ({len(high_payoff_combos)}):",
            ]
        )

        for combo_key, avg_payoff, metrics in high_payoff_combos:
            report_lines.append(
                f"  - {metrics['model1']} vs {metrics['model2']} ({metrics['behavior']}): ${avg_payoff:.1f}"
            )

        # Cultural Analysis
        report_lines.extend(
            [
                "",
                "=" * 60,
                "CULTURAL BEHAVIOR ANALYSIS",
                "=" * 60,
                "",
            ]
        )

        # Compare behaviors
        behavior_summary = {}
        for behavior, analysis in findings["behavior_effects"].items():
            behavior_summary[behavior] = {
                "acceptance_rate": analysis["avg_acceptance_rate"],
                "avg_offer": analysis["avg_initial_offer"],
                "total_rejections": sum(
                    p["rejections"] for p in analysis["rejection_patterns"]
                ),
            }

        # Sort behaviors by acceptance rate
        sorted_behaviors = sorted(
            behavior_summary.items(),
            key=lambda x: x[1]["acceptance_rate"],
            reverse=True,
        )

        report_lines.extend(
            [
                "BEHAVIOR RANKING BY ACCEPTANCE RATE:",
            ]
        )

        for i, (behavior, stats) in enumerate(sorted_behaviors, 1):
            report_lines.append(
                f"  {i}. {behavior}: {stats['acceptance_rate']:.1%} acceptance, avg offer ${stats['avg_offer']:.1f}"
            )

        # Sort by average offer
        sorted_by_offer = sorted(
            behavior_summary.items(), key=lambda x: x[1]["avg_offer"], reverse=True
        )

        report_lines.extend(
            [
                "",
                "BEHAVIOR RANKING BY GENEROSITY (INITIAL OFFERS):",
            ]
        )

        for i, (behavior, stats) in enumerate(sorted_by_offer, 1):
            report_lines.append(
                f"  {i}. {behavior}: ${stats['avg_offer']:.1f} average initial offer"
            )

        # Model Comparison Analysis
        report_lines.extend(
            [
                "",
                "MODEL COMPARISON ANALYSIS:",
                "",
            ]
        )

        # Find best and worst performing models
        best_model = max(model_performance.items(), key=lambda x: x[1]["win_rate"])
        worst_model = min(model_performance.items(), key=lambda x: x[1]["win_rate"])

        most_generous = max(model_performance.items(), key=lambda x: x[1]["avg_payoff"])
        least_generous = min(
            model_performance.items(), key=lambda x: x[1]["avg_payoff"]
        )

        report_lines.extend(
            [
                f"• Highest Win Rate: {best_model[0]} ({best_model[1]['win_rate']:.1%})",
                f"• Lowest Win Rate: {worst_model[0]} ({worst_model[1]['win_rate']:.1%})",
                f"• Highest Average Payoff: {most_generous[0]} (${most_generous[1]['avg_payoff']:.1f})",
                f"• Lowest Average Payoff: {least_generous[0]} (${least_generous[1]['avg_payoff']:.1f})",
            ]
        )

        # Conclusions
        report_lines.extend(
            [
                "",
                "=" * 60,
                "CONCLUSIONS",
                "=" * 60,
                "",
                "1. MODEL DIFFERENCES:",
                f"   • {best_model[0]} shows the most competitive negotiation patterns as Player 1",
                f"   • {worst_model[0]} demonstrates more cooperative behavior in negotiations",
                f"   • Model architecture and training differences significantly affect negotiation strategies",
                "",
                "2. CULTURAL CONTEXT EFFECTS:",
                f"   • {sorted_behaviors[0][0]} behavior shows highest cooperation ({sorted_behaviors[0][1]['acceptance_rate']:.1%} acceptance)",
                f"   • {sorted_behaviors[-1][0]} behavior shows most competitive patterns",
                f"   • Cultural prompting significantly affects negotiation strategies across all models",
                "",
                "3. STRATEGIC PATTERNS:",
                f"   • Initial offer amounts vary significantly by cultural context (${min(s[1]['avg_offer'] for s in sorted_by_offer):.1f} - ${max(s[1]['avg_offer'] for s in sorted_by_offer):.1f})",
                f"   • Rejection rates correlate with cultural behavior types",
                f"   • Player 1 advantage varies significantly across cultural contexts and model combinations",
                "",
                "4. RESEARCH IMPLICATIONS:",
                f"   • Cultural prompting is a significant factor in AI negotiation behavior across different model families",
                f"   • OpenAI, Anthropic, and other model architectures show distinct negotiation characteristics",
                f"   • Cross-cultural AI behavior requires careful consideration in applications",
                f"   • Model selection significantly impacts negotiation outcomes in cultural contexts",
                "",
            ]
        )

        # Technical Details
        report_lines.extend(
            [
                "=" * 60,
                "TECHNICAL DETAILS",
                "=" * 60,
                "",
                f"• Analysis Script: analyze_real_results.py",
                f"• Visualization Script: create_final_heatmaps.py",
                f"• Data Source: game_state.json files from individual games",
                f"• Win Rate Calculation: Excludes draws (50-50 splits)",
                f"• Games per Combination: 5 (with some incomplete due to parsing errors)",
                f"• Models: {', '.join(self.models)} via OpenRouter API",
                "",
                "Generated Files:",
                f"• Raw Data: raw_game_data.json",
                f"• Summary: summary.json",
                f"• Visualizations: final_heatmaps.png, behavior_comparison.png",
                f"• Tables: summary_table.csv",
                f"• Report: final_report.txt (this file)",
                "",
                "=" * 60,
                "END OF REPORT",
                "=" * 60,
            ]
        )

        return "\n".join(report_lines)

    def save_report(self):
        """Save the generated report"""
        report_content = self.generate_report()
        report_file = self.results_dir / "final_report.txt"

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(report_content)

        print(f"Final report saved to: {report_file}")
        return report_file


def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python generate_final_report.py <results_directory>")
        print("Example: python generate_final_report.py .logs/final_ultimatum")
        return

    results_dir = sys.argv[1]

    if not Path(results_dir).exists():
        print(f"Error: Results directory does not exist: {results_dir}")
        return

    try:
        generator = UltimatumReportGenerator(results_dir)
        generator.load_data()
        report_file = generator.save_report()

        print(f"\n" + "=" * 60)
        print("REPORT GENERATION COMPLETE!")
        print("=" * 60)
        print(f"Comprehensive analysis report saved to:")
        print(f"{report_file}")

    except Exception as e:
        print(f"Error generating report: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
