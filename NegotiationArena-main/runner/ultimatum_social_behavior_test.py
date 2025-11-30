import json
import os
import sys
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
}

SOCIAL_BEHAVIORS = [
    {
        "name": "Gujarati",
        "player1": "You are a Gujarati businessman and speak and negotiate only in Gujarati.",
        "player2": "The other player speaks and negotiates only in Gujarati. Negotiate accordingly.",
    },
    {
        "name": "Hindi",
        "player1": "You speak and bargain only in Hindi in Devanagari script. Negotiate accordingly.",
        "player2": "The other player speaks and bargains only in Hindi. Negotiate accordingly.",
    },
    {
        "name": "Marwadi",
        "player1": "You are a Marwadi businessman. Negotiate accordingly.",
        "player2": "The other person is a Marwadi businessman. Negotiate accordingly.",
    },
    {
        "name": "Punjabi",
        "player1": "You are a Punjabi businessman and speak and negotiate only in Punjabi.",
        "player2": "The other player speaks and negotiates only in Punjabi. Negotiate accordingly.",
    },
]

ITERATIONS_PER_TEST = 5


class UltimatumTestSuite:
    def __init__(self):
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_base_dir = f"./.logs/ultimatum_social_behavior_{self.timestamp}"
        os.makedirs(self.log_base_dir, exist_ok=True)

    def create_agent(self, model_name, agent_id):
        """Create an OpenRouter agent with the specified model"""
        return OpenRouterAgent(
            agent_name=agent_id,
            model=MODELS[model_name],
            temperature=0.7,
            max_tokens=400,
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

    def run_all_tests(self):
        """Run all combinations of models and behaviors (excluding same model vs same model)"""
        # Calculate total tests excluding same model comparisons
        model_combinations = []
        for model1_name in MODELS.keys():
            for model2_name in MODELS.keys():
                if model1_name != model2_name:  # Exclude same model vs same model
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

                for iteration in range(ITERATIONS_PER_TEST):
                    current_test += 1
                    print(
                        f"  Iteration {iteration + 1}/{ITERATIONS_PER_TEST} ({current_test}/{total_tests})"
                    )

                    result = self.run_single_game(
                        model1_name, model2_name, behavior, iteration + 1
                    )
                    self.results.append(result)

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


def main():
    """Main function to run the test suite"""

    # Check if OpenRouter API key is available
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY environment variable not found!")
        print("Please set your OpenRouter API key in the .env.local file")
        return

    # Create test suite
    test_suite = UltimatumTestSuite()

    # Run all tests
    results = test_suite.run_all_tests()

    # Analyze results
    analysis_results = test_suite.analyze_results()

    # Create summary report
    test_suite.create_summary_report(analysis_results)

    print("\nTest suite completed successfully!")


if __name__ == "__main__":
    main()
