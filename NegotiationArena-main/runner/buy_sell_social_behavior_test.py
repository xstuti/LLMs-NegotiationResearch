import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent directory to Python path for module imports
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
from dotenv import load_dotenv

from games.buy_sell_game.game import BuySellGame
from negotiationarena.agents.gemini import GeminiAgent
from negotiationarena.agents.openrouter_agent import OpenRouterAgent
from negotiationarena.constants import *
from negotiationarena.game_objects.goal import BuyerGoal, SellerGoal
from negotiationarena.game_objects.resource import Resources
from negotiationarena.game_objects.valuation import Valuation

load_dotenv(".env.local")

# Test configurations
MODELS = {
    "GPT-4o": "openai/gpt-4o",
    "Claude-3.5-Haiku": "anthropic/claude-3.5-haiku",
    #"GPT-3.5": "openai/gpt-3.5-turbo",
    # "Google-2.0-Flash": "google/gemini-2.0-flash-001",
    "Claude-3-Haiku": "anthropic/claude-3-haiku",
    # "GPT-oss": "openai/gpt-oss-20b:free",
    "Llama-3.3-70B-Instruct": "meta-llama/llama-3.3-70b-instruct",
}

SOCIAL_BEHAVIORS = [
    {
        "name": "Hindi",
        "seller": "You speak and bargain only in Hindi. Negotiate accordingly.",
        "buyer": "The other player speaks and bargains only in Hindi. Negotiate accordingly.",
    },
    {
        "name": "Gujarati",
        "seller": "You speak and negotiate only in Gujarati. Negotiate accordingly.",
        "buyer": "The other player speaks and negotiates only in Gujarati. Negotiate accordingly.",
    },
    {
        "name": "Punjabi",
        "seller": "You speak and negotiate only in Punjabi. Negotiate accordingly.",
        "buyer": "The other player speaks and negotiates only in Punjabi. Negotiate accordingly.",
    },
    # {
    #     "name": "Marwadi",
    #     "seller": "You speak and negotiate only in Marwadi language. Negotiate accordingly.",
    #     "buyer": "The other player speaks and negotiates only in Marwadi language. Negotiate accordingly.",
    # },
    {
        "name": "English",
        "seller": "",
        "buyer": "",
    },
]

ITERATIONS_PER_TEST = 30

# Buy-Sell specific configurations - using same setup as buysell_main.py
GAME_CONFIG = {
    "resource_name": "X",
    "seller_cost": 40,
    "buyer_max": 60,
    "buyer_money": 1000,
}


class BuySellTestSuite:
    def __init__(self):
        self.results = []
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_base_dir = f"./.logs/final_buysell"
        os.makedirs(self.log_base_dir, exist_ok=True)

    def create_agent(self, model_name, agent_id):
        """Create an OpenRouter agent with the specified model"""
        if "Gemini" in model_name:
            return GeminiAgent(
                agent_name=agent_id,
                model=MODELS[model_name],
                temperature=0.7,
                max_tokens=None,
            )
        return OpenRouterAgent(
            agent_name=agent_id,
            model=MODELS[model_name],
            temperature=0.7,
            max_tokens=None,
        )

    def run_single_game(self, model1_name, model2_name, behavior, iteration):
        """Run a single buy-sell game with specified models and behavior"""

        # Create agents - model1 is seller (AGENT_ONE), model2 is buyer (AGENT_TWO)
        seller_agent = self.create_agent(model1_name, AGENT_ONE)
        buyer_agent = self.create_agent(model2_name, AGENT_TWO)

        # Setup log directory for this specific test
        log_dir = os.path.join(
            self.log_base_dir,
            f"{model1_name}_{model2_name}_{behavior['name']}_iter{iteration}",
        )

        # Use exact same configuration as buysell_main.py
        game = BuySellGame(
            players=[seller_agent, buyer_agent],
            iterations=10,
            player_goals=[
                SellerGoal(
                    cost_of_production=Valuation(
                        {GAME_CONFIG["resource_name"]: GAME_CONFIG["seller_cost"]}
                    )
                ),
                BuyerGoal(
                    willingness_to_pay=Valuation(
                        {GAME_CONFIG["resource_name"]: GAME_CONFIG["buyer_max"]}
                    )
                ),
            ],
            player_starting_resources=[
                Resources({GAME_CONFIG["resource_name"]: 1}),
                Resources({MONEY_TOKEN: GAME_CONFIG["buyer_money"]}),
            ],
            player_conversation_roles=[
                f"You are {AGENT_ONE}.",
                f"You are {AGENT_TWO}.",
            ],
            player_social_behaviour=[behavior["seller"], behavior["buyer"]],
            log_dir=log_dir,
        )

        # Run the game
        try:
            result = game.run()

            # Extract results from the game state
            final_state = game.game_state[-1] if game.game_state else None

            if final_state and "summary" in final_state:
                summary = final_state["summary"]
                seller_final_resources = summary["final_resources"][0]
                buyer_final_resources = summary["final_resources"][1]
                final_response = summary.get("final_response", "NONE")
                player_outcomes = summary.get("player_outcome", [0, 0])

                # Calculate if trade happened
                trade_occurred = final_response == ACCEPTING_TAG

                # Calculate seller profit and buyer savings
                if trade_occurred:
                    # Find the money exchanged (seller's final money)
                    money_received = seller_final_resources.resource_dict.get(
                        MONEY_TOKEN, 0
                    )
                    seller_profit = money_received - GAME_CONFIG["seller_cost"]
                    buyer_savings = GAME_CONFIG["buyer_max"] - money_received
                else:
                    seller_profit = 0
                    buyer_savings = 0

                game_result = {
                    "seller_model": model1_name,
                    "buyer_model": model2_name,
                    "behavior": behavior["name"],
                    "iteration": iteration,
                    "seller_cost": GAME_CONFIG["seller_cost"],
                    "buyer_max": GAME_CONFIG["buyer_max"],
                    "trade_occurred": trade_occurred,
                    "final_response": final_response,
                    "money_exchanged": seller_final_resources.resource_dict.get(
                        MONEY_TOKEN, 0
                    )
                    if trade_occurred
                    else 0,
                    "seller_profit": seller_profit,
                    "buyer_savings": buyer_savings,
                    "seller_outcome": player_outcomes[0],
                    "buyer_outcome": player_outcomes[1],
                    "efficiency": (seller_profit + buyer_savings)
                    / (GAME_CONFIG["buyer_max"] - GAME_CONFIG["seller_cost"])
                    if trade_occurred
                    else 0,
                    "seller_final_resources": seller_final_resources.resource_dict,
                    "buyer_final_resources": buyer_final_resources.resource_dict,
                    "game_completed": True,
                    "log_dir": log_dir,
                }
            else:
                # Game didn't complete properly
                game_result = {
                    "seller_model": model1_name,
                    "buyer_model": model2_name,
                    "behavior": behavior["name"],
                    "iteration": iteration,
                    "seller_cost": GAME_CONFIG["seller_cost"],
                    "buyer_max": GAME_CONFIG["buyer_max"],
                    "trade_occurred": False,
                    "final_response": "NONE",
                    "money_exchanged": 0,
                    "seller_profit": 0,
                    "buyer_savings": 0,
                    "seller_outcome": 0,
                    "buyer_outcome": 0,
                    "efficiency": 0,
                    "seller_final_resources": {},
                    "buyer_final_resources": {},
                    "game_completed": False,
                    "log_dir": log_dir,
                }

            return game_result

        except Exception as e:
            print(
                f"Error in game {model1_name} vs {model2_name}, {behavior['name']}, iter {iteration}: {str(e)}"
            )
            return {
                "seller_model": model1_name,
                "buyer_model": model2_name,
                "behavior": behavior["name"],
                "iteration": iteration,
                "seller_cost": GAME_CONFIG["seller_cost"],
                "buyer_max": GAME_CONFIG["buyer_max"],
                "trade_occurred": False,
                "final_response": "ERROR",
                "money_exchanged": 0,
                "seller_profit": 0,
                "buyer_savings": 0,
                "seller_outcome": 0,
                "buyer_outcome": 0,
                "efficiency": 0,
                "seller_final_resources": {},
                "buyer_final_resources": {},
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

        new_behaviors = SOCIAL_BEHAVIORS

        total_tests = len(model_combinations) * len(new_behaviors) * ITERATIONS_PER_TEST
        current_test = 0

        print(
            f"Starting {total_tests} tests for ALL behaviors..."
        )
        print(f"Model combinations: {len(model_combinations)}")
        print(f"Behaviors: {len(new_behaviors)}")
        print(f"Iterations per combination: {ITERATIONS_PER_TEST}")

        for model1_name, model2_name in model_combinations:
            for behavior in new_behaviors:
                print(
                    f"\nTesting {model1_name} (seller) vs {model2_name} (buyer) with {behavior['name']} behavior..."
                )

                for iteration in range(0, 0+ ITERATIONS_PER_TEST):
                        # --- ADD THIS ---
                    log_dir = os.path.join(
                        self.log_base_dir,
                        f"{model1_name}_{model2_name}_{behavior['name']}_iter{iteration + 1}",
                    )
                    if os.path.exists(log_dir):
                        print(f"  Skipping — already exists: {log_dir}")
                        continue
                # --- END ADD ---
                    current_test += 1
                    print(
                        f"  Iteration {iteration + 1}/{ITERATIONS_PER_TEST} ({current_test}/{total_tests})"
                    )

                    result = self.run_single_game(
                        model1_name,
                        model2_name,
                        behavior,
                        iteration + 1,
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

        # Calculate metrics for each behavior
        analysis_results = {}

        for behavior in SOCIAL_BEHAVIORS:
            behavior_name = behavior["name"]
            behavior_df = df[df["behavior"] == behavior_name]

            # Initialize matrices for this behavior
            models = list(MODELS.keys())
            trade_success_rates = pd.DataFrame(
                index=models, columns=models, dtype=float
            )
            avg_seller_profits = pd.DataFrame(index=models, columns=models, dtype=float)
            avg_buyer_savings = pd.DataFrame(index=models, columns=models, dtype=float)
            avg_efficiency = pd.DataFrame(index=models, columns=models, dtype=float)

            for seller_model in models:
                for buyer_model in models:
                    subset = behavior_df[
                        (behavior_df["seller_model"] == seller_model)
                        & (behavior_df["buyer_model"] == buyer_model)
                        & (behavior_df["game_completed"] == True)
                    ]

                    if len(subset) > 0:
                        # Calculate trade success rate
                        success_rate = subset["trade_occurred"].mean()
                        trade_success_rates.loc[seller_model, buyer_model] = (
                            success_rate
                        )

                        # Calculate average seller profit
                        avg_seller_profit = subset["seller_profit"].mean()
                        avg_seller_profits.loc[seller_model, buyer_model] = (
                            avg_seller_profit
                        )

                        # Calculate average buyer savings
                        avg_buyer_saving = subset["buyer_savings"].mean()
                        avg_buyer_savings.loc[seller_model, buyer_model] = (
                            avg_buyer_saving
                        )

                        # Calculate average efficiency
                        avg_eff = subset["efficiency"].mean()
                        avg_efficiency.loc[seller_model, buyer_model] = avg_eff
                    else:
                        trade_success_rates.loc[seller_model, buyer_model] = 0.0
                        avg_seller_profits.loc[seller_model, buyer_model] = 0.0
                        avg_buyer_savings.loc[seller_model, buyer_model] = 0.0
                        avg_efficiency.loc[seller_model, buyer_model] = 0.0

            analysis_results[behavior_name] = {
                "trade_success_rates": trade_success_rates,
                "avg_seller_profits": avg_seller_profits,
                "avg_buyer_savings": avg_buyer_savings,
                "avg_efficiency": avg_efficiency,
            }

        # Create visualizations
        self.create_heatmaps(analysis_results)

        return analysis_results

    def create_heatmaps(self, analysis_results):
        """Create heatmap visualizations for buy-sell game metrics"""

        # Create a figure with subplots for each behavior
        n_behaviors = len(SOCIAL_BEHAVIORS)
        fig, axes = plt.subplots(n_behaviors, 4, figsize=(20, 4 * n_behaviors))

        if n_behaviors == 1:
            axes = axes.reshape(1, -1)

        for idx, behavior in enumerate(SOCIAL_BEHAVIORS):
            behavior_name = behavior["name"]

            # Trade Success Rate heatmap
            success_rates = analysis_results[behavior_name]["trade_success_rates"]
            sns.heatmap(
                success_rates,
                annot=True,
                fmt=".2f",
                cmap="Greens",
                vmin=0,
                vmax=1,
                ax=axes[idx, 0],
                cbar_kws={"label": "Trade Success Rate"},
            )
            axes[idx, 0].set_title(f"{behavior_name} - Trade Success Rate")
            axes[idx, 0].set_xlabel("Buyer Model")
            axes[idx, 0].set_ylabel("Seller Model")

            # Average Seller Profit heatmap
            seller_profits = analysis_results[behavior_name]["avg_seller_profits"]
            sns.heatmap(
                seller_profits,
                annot=True,
                fmt=".1f",
                cmap="Blues",
                ax=axes[idx, 1],
                cbar_kws={"label": "Average Seller Profit"},
            )
            axes[idx, 1].set_title(f"{behavior_name} - Average Seller Profit")
            axes[idx, 1].set_xlabel("Buyer Model")
            axes[idx, 1].set_ylabel("Seller Model")

            # Average Buyer Savings heatmap
            buyer_savings = analysis_results[behavior_name]["avg_buyer_savings"]
            sns.heatmap(
                buyer_savings,
                annot=True,
                fmt=".1f",
                cmap="Oranges",
                ax=axes[idx, 2],
                cbar_kws={"label": "Average Buyer Savings"},
            )
            axes[idx, 2].set_title(f"{behavior_name} - Average Buyer Savings")
            axes[idx, 2].set_xlabel("Buyer Model")
            axes[idx, 2].set_ylabel("Seller Model")

            # Average Efficiency heatmap
            efficiency = analysis_results[behavior_name]["avg_efficiency"]
            sns.heatmap(
                efficiency,
                annot=True,
                fmt=".2f",
                cmap="Purples",
                vmin=0,
                vmax=1,
                ax=axes[idx, 3],
                cbar_kws={"label": "Average Efficiency"},
            )
            axes[idx, 3].set_title(f"{behavior_name} - Average Efficiency")
            axes[idx, 3].set_xlabel("Buyer Model")
            axes[idx, 3].set_ylabel("Seller Model")

        plt.tight_layout()

        # Save the plot
        plot_file = os.path.join(
            self.log_base_dir, "buysell_social_behavior_analysis.png"
        )
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.show()

        print(f"Analysis complete! Results saved in: {self.log_base_dir}")
        print(f"Visualization saved as: {plot_file}")

    def create_summary_report(self, analysis_results):
        """Create a summary report of the results"""
        report_file = os.path.join(self.log_base_dir, "summary_report.txt")

        with open(report_file, "w") as f:
            f.write("BUY-SELL GAME SOCIAL BEHAVIOR TEST SUMMARY\n")
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
            f.write(f"Game completion rate: {success_rate:.1f}%\n")

            # Trade occurrence rate
            trades_occurred = sum(
                1 for r in self.results if r.get("trade_occurred", False)
            )
            trade_rate = trades_occurred / len(self.results) * 100
            f.write(f"Trade occurrence rate: {trade_rate:.1f}%\n\n")

            # Behavior-specific analysis
            for behavior in SOCIAL_BEHAVIORS:
                behavior_name = behavior["name"]
                f.write(f"{behavior_name.upper()} BEHAVIOR RESULTS:\n")
                f.write("-" * 30 + "\n")

                success_rates = analysis_results[behavior_name]["trade_success_rates"]
                seller_profits = analysis_results[behavior_name]["avg_seller_profits"]
                buyer_savings = analysis_results[behavior_name]["avg_buyer_savings"]
                efficiency = analysis_results[behavior_name]["avg_efficiency"]

                f.write("Trade Success Rates:\n")
                f.write(success_rates.to_string())
                f.write("\n\nAverage Seller Profits:\n")
                f.write(seller_profits.to_string())
                f.write("\n\nAverage Buyer Savings:\n")
                f.write(buyer_savings.to_string())
                f.write("\n\nAverage Efficiency:\n")
                f.write(efficiency.to_string())
                f.write("\n\n")

        print(f"Summary report saved: {report_file}")


def main():
    """Main function to run the test suite"""

    # Check if OpenRouter API key is available
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY environment variable not found!")
        print("Please set your OpenRouter API key in the .env.local file")
        return
    
    key = os.environ.get("OPENROUTER_API_KEY")
    if key:
        print(f"API Key loaded (first 8 chars): {key[:13]}...")
    else:
        print("API Key still not found in environment!")

    # Create test suite
    test_suite = BuySellTestSuite()

    # Run all tests
    results = test_suite.run_all_tests()

    # Analyze results
    analysis_results = test_suite.analyze_results()

    # Create summary report
    test_suite.create_summary_report(analysis_results)

    print("\nTest suite completed successfully!")


if __name__ == "__main__":
    main()
