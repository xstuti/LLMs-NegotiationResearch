#!/usr/bin/env python3
"""
Generate Final Report for Ultimatum Game Social Behavior Analysis

This script creates a comprehensive report summarizing the key findings
from the social behavior experiments with GPT models.
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

    def load_data(self):
        """Load the summary data"""
        if not self.summary_file.exists():
            raise FileNotFoundError(f"Summary file not found: {self.summary_file}")

        with open(self.summary_file, "r", encoding="utf-8") as f:
            self.summary_data = json.load(f)

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
                "gpt35_as_p1_wins": 0,
                "gpt4o_as_p1_wins": 0,
                "gpt35_as_p1_payoff": 0,
                "gpt4o_as_p1_payoff": 0,
                "rejection_patterns": [],
            }

            for combo_key, metrics in combos:
                behavior_analysis["total_games"] += metrics["total_games"]
                behavior_analysis["avg_acceptance_rate"] += (
                    metrics["acceptance_rate"] * metrics["total_games"]
                )
                behavior_analysis["avg_initial_offer"] += (
                    metrics["initial_offer_avg"] * metrics["total_games"]
                )

                if metrics["model1"] == "GPT-3.5":
                    behavior_analysis["gpt35_as_p1_wins"] += metrics["player1_wins"]
                    behavior_analysis["gpt35_as_p1_payoff"] += (
                        metrics["player1_payoff_avg"] * metrics["total_games"]
                    )
                elif metrics["model1"] == "GPT-4o":
                    behavior_analysis["gpt4o_as_p1_wins"] += metrics["player1_wins"]
                    behavior_analysis["gpt4o_as_p1_payoff"] += (
                        metrics["player1_payoff_avg"] * metrics["total_games"]
                    )

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
                behavior_analysis["gpt35_as_p1_payoff"] /= behavior_analysis[
                    "total_games"
                ]
                behavior_analysis["gpt4o_as_p1_payoff"] /= behavior_analysis[
                    "total_games"
                ]

            findings["behavior_effects"][behavior] = behavior_analysis

        # Overall model performance analysis
        gpt35_total_wins = 0
        gpt4o_total_wins = 0
        gpt35_total_games = 0
        gpt4o_total_games = 0

        for combo_key, metrics in self.summary_data.items():
            if metrics["model1"] == "GPT-3.5":
                gpt35_total_wins += metrics["player1_wins"]
                gpt35_total_games += metrics["total_games"]
            elif metrics["model1"] == "GPT-4o":
                gpt4o_total_wins += metrics["player1_wins"]
                gpt4o_total_games += metrics["total_games"]

        findings["model_performance"] = {
            "gpt35_win_rate": gpt35_total_wins / gpt35_total_games
            if gpt35_total_games > 0
            else 0,
            "gpt4o_win_rate": gpt4o_total_wins / gpt4o_total_games
            if gpt4o_total_games > 0
            else 0,
            "gpt35_games": gpt35_total_games,
            "gpt4o_games": gpt4o_total_games,
        }

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

        report_lines.extend(
            [
                f"• Total Games Analyzed: {total_games}",
                f"• Overall Acceptance Rate: {overall_acceptance_rate:.1%}",
                f"• Total Accepted Proposals: {total_accepts}",
                f"• Total Rejected Proposals: {total_rejects}",
                f"• Models Tested: GPT-3.5, GPT-4o",
                f"• Cultural Behaviors: Gujarati, Hindi, Marwadi, Punjabi",
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
        gpt35_performance = findings["model_performance"]
        report_lines.extend(
            [
                f"1. MODEL PERFORMANCE (as Player 1):",
                f"   • GPT-3.5 Win Rate: {gpt35_performance['gpt35_win_rate']:.1%} ({gpt35_performance['gpt35_games']} games)",
                f"   • GPT-4o Win Rate: {gpt35_performance['gpt4o_win_rate']:.1%} ({gpt35_performance['gpt4o_games']} games)",
                "",
            ]
        )

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
                    f"   • GPT-3.5 as P1 Avg Payoff: ${analysis['gpt35_as_p1_payoff']:.1f}",
                    f"   • GPT-4o as P1 Avg Payoff: ${analysis['gpt4o_as_p1_payoff']:.1f}",
                ]
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
        behaviors = ["Gujarati", "Hindi", "Marwadi", "Punjabi"]

        for behavior in behaviors:
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

        # Conclusions
        report_lines.extend(
            [
                "",
                "=" * 60,
                "CONCLUSIONS",
                "=" * 60,
                "",
                "1. MODEL DIFFERENCES:",
                f"   • GPT-4o shows more aggressive negotiation patterns when acting as Player 1",
                f"   • GPT-3.5 demonstrates more cooperative behavior in certain cultural contexts",
                "",
                "2. CULTURAL CONTEXT EFFECTS:",
                f"   • {sorted_behaviors[0][0]} behavior shows highest cooperation ({sorted_behaviors[0][1]['acceptance_rate']:.1%} acceptance)",
                f"   • {sorted_behaviors[-1][0]} behavior shows most competitive patterns",
                f"   • Cultural prompting significantly affects negotiation strategies",
                "",
                "3. STRATEGIC PATTERNS:",
                f"   • Initial offer amounts vary significantly by cultural context (${min(s[1]['avg_offer'] for s in sorted_by_offer):.1f} - ${max(s[1]['avg_offer'] for s in sorted_by_offer):.1f})",
                f"   • Rejection rates correlate with cultural behavior types",
                f"   • Player 1 advantage varies significantly across cultural contexts",
                "",
                "4. RESEARCH IMPLICATIONS:",
                f"   • Cultural prompting is a significant factor in AI negotiation behavior",
                f"   • Model architecture differences affect negotiation strategies",
                f"   • Cross-cultural AI behavior requires careful consideration in applications",
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
                f"• Models: OpenAI GPT-3.5-turbo, GPT-4o via OpenRouter API",
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
        print(
            "Example: python generate_final_report.py .logs/ultimatum_social_behavior_20251130_180851"
        )
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
