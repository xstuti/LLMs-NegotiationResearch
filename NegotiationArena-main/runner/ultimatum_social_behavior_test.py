import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add current directory to Python path for module imports
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

from games.ultimatum.game import MultiTurnUltimatumGame
from negotiationarena.agents.openrouter_agent import OpenRouterAgent
from negotiationarena.constants import *
from negotiationarena.game_objects.goal import UltimatumGoal
from negotiationarena.game_objects.resource import Resources

load_dotenv(".env.local")

# Test configurations
MODELS = {
    "GPT-4o": "openai/gpt-4o",
    "GPT-3.5": "openai/gpt-3.5-turbo",
    "Claude-3-Haiku": "anthropic/claude-3-haiku",
    "Claude-3.5-Haiku": "anthropic/claude-3.5-haiku",
    # "Llama-3.3-70B": "meta-llama/llama-3.3-70b-instruct",
    # "GPT-oss": "openai/gpt-oss-20b",
    # "GPT-oss-120": "openai/gpt-oss-120b",
    # "Deepseek-3.2": "deepseek/deepseek-v3.2",
}

SOCIAL_BEHAVIORS = [
    # {
    #     "name": "Hindi",
    #     "player1": "केवल हिंदी में बात करें।",
    #     "player2": "केवल हिंदी में बात करें।",
    # },
    # {
    #     "name": "Gujarati",
    #     "player1": "ફક્ત ગુજરાતીમાં જ વાત કરો.",
    #     "player2": "ફક્ત ગુજરાતીમાં જ વાત કરો.",
    # },
    # {
    #     "name": "Punjabi",
    #     "player1": "ਸਿਰਫ਼ ਪੰਜਾਬੀ ਵਿੱਚ ਗੱਲ ਕਰੋ।",
    #     "player2": "ਸਿਰਫ਼ ਪੰਜਾਬੀ ਵਿੱਚ ਗੱਲ ਕਰੋ।",
    # },
    # # {
    # #     "name": "Marwadi",
    # #     "player1": "You speak and negotiate only in Marwadi language. Negotiate accordingly.",
    # #     "player2": "The other player speaks and negotiates only in Marwadi language. Negotiate accordingly.",
    # # },
    {
        "name": "English",
        "player1": "Talk only in English.",
        "player2": "Talk only in English.",
    },
    # {
    #     "name": "Tamil",
    #     "player1": "You speak and negotiate only in Tamil language. Negotiate accordingly.",
    #     "player2": "You speak and negotiate only in Tamil language. Negotiate accordingly.",
    # },
]

ITERATIONS_PER_TEST = 30


class UltimatumTestSuite:
    def __init__(self, resume_from=None, complete_mode=False):
        self.results = []
        self.completed_tests = set()
        self.complete_mode = complete_mode

        if resume_from:
            # Resume from existing directory
            self.log_base_dir = resume_from
            self.timestamp = Path(resume_from).name.replace(
                "ultimatum_social_behavior_", ""
            )
            print(f"Resuming from: {self.log_base_dir}")
            self.load_existing_results()
        else:
            # Start fresh
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_base_dir = f"./.logs/ultimatum_social_behavior_promptnative20to30"
            os.makedirs(self.log_base_dir, exist_ok=True)
            print(f"Starting new test suite: {self.log_base_dir}")

    def load_existing_results(self):
        """Load existing results by scanning the log directory for completed tests"""
        import re

        self.results = []
        self.completed_tests = set()

        if not os.path.exists(self.log_base_dir):
            print("Starting fresh...")
            return

        count = 0
        for item in os.listdir(self.log_base_dir):
            full_path = os.path.join(self.log_base_dir, item)
            if not os.path.isdir(full_path):
                continue

            match = re.search(r"^(.*)_vs_(.*)_(.*)_iter_(\d+)$", item)
            if match:
                model1 = match.group(1)
                model2 = match.group(2)
                behavior = match.group(3)
                iteration = int(match.group(4))

                self.results.append(
                    {
                        "model1": model1,
                        "model2": model2,
                        "behavior": behavior,
                        "iteration": iteration,
                    }
                )
                self.completed_tests.add((model1, model2, behavior, iteration))
                count += 1

        print(f"Found {count} completed tests from existing directories")

    def is_test_completed(self, model1_name, model2_name, behavior_name, iteration):
        """Check if a specific test has already been completed"""
        test_key = (model1_name, model2_name, behavior_name, iteration)
        return test_key in self.completed_tests

    def create_agent(self, model_name, agent_id):
        """Create an OpenRouter agent with the specified model"""
        return OpenRouterAgent(
            agent_name=agent_id,
            model=MODELS[model_name],
            temperature=0.7,
            max_tokens=5000,
        )

    def run_single_game(self, model1_name, model2_name, behavior, iteration):
        """Run a single ultimatum game with specified models and behavior"""

        # Create agents
        agent1 = self.create_agent(model1_name, AGENT_ONE)
        agent2 = self.create_agent(model2_name, AGENT_TWO)

        # Setup log directory for this specific test
        log_dir = os.path.join(
            self.log_base_dir,
            f"{model1_name}_vs_{model2_name}_{behavior['name']}_iter_{iteration}",
        )

        # Create the game
        game = MultiTurnUltimatumGame(
            players=[agent1, agent2],
            iterations=6,  # Same as original script
            resources_support_set=Resources({"Dollars": 0}),
            player_goals=[
                UltimatumGoal(),
                UltimatumGoal(),
            ],
            player_initial_resources=[
                Resources({"Dollars": 100}),
                Resources({"Dollars": 0}),
            ],
            player_social_behaviour=[behavior["player1"], behavior["player2"]],
            player_roles=[
                f"You are {AGENT_ONE}.",
                f"You are {AGENT_TWO}.",
            ],
            log_dir=log_dir,
        )

        # Run the game
        try:
            result = game.run()

            # Extract results (you may need to adjust this based on actual game return structure)
            game_result = {
                "model1": model1_name,
                "model2": model2_name,
                "behavior": behavior["name"],
                "iteration": iteration,
                "player1_final_resources": result.get("player1_resources", 0)
                if result
                else 0,
                "player2_final_resources": result.get("player2_resources", 0)
                if result
                else 0,
                "game_completed": result is not None,
                "log_dir": log_dir,
            }

            return game_result

        except Exception as e:
            print(
                f"Error in game {model1_name} vs {model2_name}, {behavior['name']}, iter {iteration}: {str(e)}"
            )
            return {
                "model1": model1_name,
                "model2": model2_name,
                "behavior": behavior["name"],
                "iteration": iteration,
                "player1_final_resources": 0,
                "player2_final_resources": 0,
                "game_completed": False,
                "error": str(e),
                "log_dir": log_dir,
            }

    def get_incomplete_combinations(self):
        """Identify combinations that have fewer than ITERATIONS_PER_TEST iterations
        by checking results already loaded from all_results.json into self.results.

        Returns a list of dicts with model1, model2, behavior, completed count,
        and the specific missing iteration numbers.
        """
        from collections import defaultdict

        # Collect completed iteration numbers per (model1, model2, behavior)
        completed_iterations = defaultdict(set)
        for result in self.results:
            combo = (result["model1"], result["model2"], result["behavior"])
            completed_iterations[combo].add(result["iteration"])

        # Expected iteration numbers: 1 through ITERATIONS_PER_TEST
        expected_iterations = set(range(1, ITERATIONS_PER_TEST + 1))

        incomplete = []

        # Check all expected combinations (excluding same model vs same model)
        for model1_name in MODELS.keys():
            for model2_name in MODELS.keys():
                if model1_name == model2_name:
                    continue
                for behavior in SOCIAL_BEHAVIORS:
                    combo = (model1_name, model2_name, behavior["name"])
                    done = completed_iterations.get(combo, set())
                    missing = sorted(expected_iterations - done)

                    if missing:
                        incomplete.append(
                            {
                                "model1": model1_name,
                                "model2": model2_name,
                                "behavior": behavior["name"],
                                "completed": len(done),
                                "missing_iterations": missing,
                                "missing": len(missing),
                            }
                        )

        if not incomplete:
            print("All combinations are complete!")
        else:
            print(
                f"Found {len(incomplete)} incomplete combinations from {len(self.results)} loaded results"
            )

        return incomplete

    def run_all_tests(self):
        """Run all combinations of models and behaviors (excluding same model vs same model)"""

        # If in complete mode, only run missing iterations
        if self.complete_mode:
            incomplete_combinations = self.get_incomplete_combinations()

            if not incomplete_combinations:
                print("All combinations are complete with 10 iterations each!")
                return self.results

            print(f"\nFound {len(incomplete_combinations)} incomplete combinations")
            print("=" * 60)

            total_missing = sum(c["missing"] for c in incomplete_combinations)
            current_test = 0

            # Filter to only combinations where at least one model is Llama

            if not incomplete_combinations:
                print("No incomplete Llama combinations remaining!")
                return self.results

            total_missing = sum(c["missing"] for c in incomplete_combinations)
            current_test = 0

            for combo in incomplete_combinations:
                print(
                    f"\nCompleting {combo['model1']} vs {combo['model2']} - {combo['behavior']}"
                )
                print(f"  Currently has: {combo['completed']} iterations")
                print(f"  Need to run: {combo['missing']} more iterations")

                # Find the matching behavior dict
                behavior_dict = None
                for b in SOCIAL_BEHAVIORS:
                    if b["name"] == combo["behavior"]:
                        behavior_dict = b
                        break

                if not behavior_dict:
                    print(
                        f"  ERROR: Behavior '{combo['behavior']}' not found in SOCIAL_BEHAVIORS"
                    )
                    continue

                # Run the specific missing iterations
                for iteration in combo["missing_iterations"]:
                    current_test += 1
                    print(
                        f"  Running iteration {iteration} ({current_test}/{total_missing})"
                    )

                    result = self.run_single_game(
                        combo["model1"], combo["model2"], behavior_dict, iteration
                    )
                    self.results.append(result)

                    # Add to completed tests set
                    test_key = (
                        combo["model1"],
                        combo["model2"],
                        combo["behavior"],
                        iteration,
                    )
                    self.completed_tests.add(test_key)

                    # Save intermediate results
                    self.save_results()

            print(f"\nCompleted {total_missing} missing iterations!")
            return self.results

        # Normal mode - run all tests
        # Calculate total tests excluding same model comparisons
        model_combinations = []
        for model1_name in MODELS.keys():
            for model2_name in MODELS.keys():
                if model1_name != model2_name:  # Exclude same model vs same model
                    # Only run combinations where at least one model is Llama
                    if model1_name != model2_name:
                        model_combinations.append((model1_name, model2_name))
        total_tests = (
            len(model_combinations) * len(SOCIAL_BEHAVIORS) * ITERATIONS_PER_TEST
        )
        current_test = 0

        print(f"Starting {total_tests} tests (excluding same model vs same model)...")
        print(f"Model combinations: {len(model_combinations)}")
        print(f"Behaviors: {len(SOCIAL_BEHAVIORS)}")
        print(f"Iterations per combination: {ITERATIONS_PER_TEST}")

        for model1_name, model2_name in model_combinations:
            for behavior in SOCIAL_BEHAVIORS:
                print(
                    f"\nTesting {model1_name} vs {model2_name} with {behavior['name']} behavior..."
                )

                # Run all model combinations for the new behaviors only
                # This skips all previously tested combinations with old behaviors

                for iteration in range(ITERATIONS_PER_TEST):
                    current_test += 1

                    # Check if this test has already been completed
                    if self.is_test_completed(
                        model1_name, model2_name, behavior["name"], iteration + 1
                    ):
                        print(
                            f"  Iteration {iteration + 1}/{ITERATIONS_PER_TEST} ({current_test}/{total_tests}) - SKIPPED (already completed)"
                        )
                        continue

                    print(
                        f"  Iteration {iteration + 1}/{ITERATIONS_PER_TEST} ({current_test}/{total_tests})"
                    )

                    result = self.run_single_game(
                        model1_name, model2_name, behavior, iteration + 1
                    )
                    self.results.append(result)

                    # Add to completed tests set
                    test_key = (
                        model1_name,
                        model2_name,
                        behavior["name"],
                        iteration + 1,
                    )
                    self.completed_tests.add(test_key)

                    # Save intermediate results
                    self.save_results()

        print("\nAll tests completed!")
        return self.results

    def save_results(self):
        """Save results to JSON file"""
        results_file = os.path.join(self.log_base_dir, "all_results.json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

    def analyze_results(self):
        """Analyze results and create visualizations"""
        if not self.results:
            print("No results to analyze!")
            return

        # Convert to DataFrame
        df = pd.DataFrame(self.results)

        # Calculate win rates and average payoffs
        analysis_results = {}

        for behavior in SOCIAL_BEHAVIORS:
            behavior_name = behavior["name"]
            behavior_df = df[df["behavior"] == behavior_name]

            # Initialize matrices for this behavior
            models = list(MODELS.keys())
            win_rates = pd.DataFrame(index=models, columns=models, dtype=float)
            avg_payoffs = pd.DataFrame(index=models, columns=models, dtype=float)

            for model1 in models:
                for model2 in models:
                    subset = behavior_df[
                        (behavior_df["model1"] == model1)
                        & (behavior_df["model2"] == model2)
                        & (behavior_df["game_completed"] == True)
                    ]

                    if len(subset) > 0:
                        # Calculate win rate for model1 (Player 1)
                        # In Ultimatum game, success could be measured by final resources
                        player1_wins = sum(
                            subset["player1_final_resources"]
                            > subset["player2_final_resources"]
                        )
                        win_rate = player1_wins / len(subset)
                        win_rates.loc[model1, model2] = win_rate

                        # Calculate average payoff for model1
                        avg_payoff = subset["player1_final_resources"].mean()
                        avg_payoffs.loc[model1, model2] = avg_payoff
                    else:
                        win_rates.loc[model1, model2] = 0.0
                        avg_payoffs.loc[model1, model2] = 0.0

            analysis_results[behavior_name] = {
                "win_rates": win_rates,
                "avg_payoffs": avg_payoffs,
            }

        # Create visualizations
        self.create_heatmaps(analysis_results)

        return analysis_results

    def create_heatmaps(self, analysis_results):
        """Create heatmap visualizations similar to the provided image"""

        # Create a figure with subplots for each behavior
        n_behaviors = len(SOCIAL_BEHAVIORS)
        fig, axes = plt.subplots(n_behaviors, 2, figsize=(15, 4 * n_behaviors))

        if n_behaviors == 1:
            axes = axes.reshape(1, -1)

        for idx, behavior in enumerate(SOCIAL_BEHAVIORS):
            behavior_name = behavior["name"]

            # Win Rate heatmap
            win_rates = analysis_results[behavior_name]["win_rates"]
            sns.heatmap(
                win_rates,
                annot=True,
                fmt=".2f",
                cmap="Blues",
                vmin=0,
                vmax=1,
                ax=axes[idx, 0],
                cbar_kws={"label": "Win Rate"},
            )
            axes[idx, 0].set_title(f"{behavior_name} - Win Rate")
            axes[idx, 0].set_xlabel("Player 2")
            axes[idx, 0].set_ylabel("Player 1")

            # Average Payoff heatmap
            avg_payoffs = analysis_results[behavior_name]["avg_payoffs"]
            sns.heatmap(
                avg_payoffs,
                annot=True,
                fmt=".1f",
                cmap="Blues",
                ax=axes[idx, 1],
                cbar_kws={"label": "Average Payoff"},
            )
            axes[idx, 1].set_title(f"{behavior_name} - Average Payoff")
            axes[idx, 1].set_xlabel("Player 2")
            axes[idx, 1].set_ylabel("Player 1")

        plt.tight_layout()

        # Save the plot
        plot_file = os.path.join(self.log_base_dir, "social_behavior_analysis.png")
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Analysis complete! Results saved in: {self.log_base_dir}")
        print(f"Visualization saved as: {plot_file}")

    def create_summary_report(self, analysis_results):
        """Create a summary report of the results"""
        report_file = os.path.join(self.log_base_dir, "summary_report.txt")

        with open(report_file, "w") as f:
            f.write("ULTIMATUM GAME SOCIAL BEHAVIOR TEST SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Test Date: {self.timestamp}\n")
            f.write(f"Models Tested: {', '.join(MODELS.keys())}\n")
            f.write(
                f"Behaviors Tested: {', '.join([b['name'] for b in SOCIAL_BEHAVIORS])}\n"
            )
            f.write(f"Iterations per combination: {ITERATIONS_PER_TEST}\n")
            f.write(f"Total games played: {len(self.results)}\n\n")

            # Success rate
            successful_games = sum(
                1 for r in self.results if r.get("game_completed", False)
            )
            success_rate = successful_games / len(self.results) * 100
            f.write(f"Game completion rate: {success_rate:.1f}%\n\n")

            # Behavior-specific analysis
            for behavior in SOCIAL_BEHAVIORS:
                behavior_name = behavior["name"]
                f.write(f"{behavior_name.upper()} BEHAVIOR RESULTS:\n")
                f.write("-" * 30 + "\n")

                win_rates = analysis_results[behavior_name]["win_rates"]
                avg_payoffs = analysis_results[behavior_name]["avg_payoffs"]

                f.write("Win Rates (Player 1 perspective):\n")
                f.write(win_rates.to_string())
                f.write("\n\nAverage Payoffs:\n")
                f.write(avg_payoffs.to_string())
                f.write("\n\n")

        print(f"Summary report saved: {report_file}")


def find_latest_log_dir():
    """Find the most recent log directory"""
    logs_dir = Path("./.logs")
    if not logs_dir.exists():
        return None

    log_dirs = [
        d
        for d in logs_dir.iterdir()
        if d.is_dir() and d.name.startswith("ultimatum_social_behavior_")
    ]

    if not log_dirs:
        return None

    # Sort by modification time and get the most recent
    latest_dir = max(log_dirs, key=lambda d: d.stat().st_mtime)
    return str(latest_dir)


def main():
    """Main function to run the test suite"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run ultimatum game social behavior tests"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the most recent log directory",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Resume from a specific log directory path",
    )
    parser.add_argument(
        "--complete",
        action="store_true",
        help="Complete missing iterations up to 10 for each combination (requires --resume or --resume-from)",
    )

    args = parser.parse_args()

    # Check if OpenRouter API key is available
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY environment variable not found!")
        print("Please set your OpenRouter API key in the .env.local file")
        return

    # Complete mode automatically enables resume
    if args.complete:
        args.resume = True

    # Determine if we should resume
    resume_dir = None
    if args.resume_from:
        resume_dir = args.resume_from
        if not os.path.exists(resume_dir):
            print(f"ERROR: Specified directory does not exist: {resume_dir}")
            return
    elif args.resume or args.complete:
        resume_dir = find_latest_log_dir()
        if resume_dir:
            print(f"Found latest log directory: {resume_dir}")
        else:
            print("No previous log directory found. Starting fresh.")

    # Create test suite
    test_suite = UltimatumTestSuite(resume_from=resume_dir, complete_mode=args.complete)

    # Run all tests
    results = test_suite.run_all_tests()

    # Analyze results
    analysis_results = test_suite.analyze_results()

    # Create summary report
    test_suite.create_summary_report(analysis_results)

    print("\nTest suite completed successfully!")


if __name__ == "__main__":
    main()
