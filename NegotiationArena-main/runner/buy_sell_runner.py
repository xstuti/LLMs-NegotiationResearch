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
from dotenv import load_dotenv

from games.buy_sell_game.game import BuySellGame
from negotiationarena.agents.gemini import GeminiAgent
from negotiationarena.agents.openrouter_agent import OpenRouterAgent
from negotiationarena.constants import *
from negotiationarena.game_objects.goal import BuyerGoal, SellerGoal
from negotiationarena.game_objects.resource import Resources
from negotiationarena.game_objects.valuation import Valuation

load_dotenv(".env.local", override=True)

# Test configurations
MODELS = {
    "GPT-4o": "openai/gpt-4o",
    "Claude-3.5-Haiku": "anthropic/claude-3.5-haiku",
    "GPT-3.5": "openai/gpt-3.5-turbo",
    "Claude-3-Haiku": "anthropic/claude-3-haiku",
}

SOCIAL_BEHAVIORS = [
    {
        "name": "Hindi",
        "seller": "You speak and bargain only in Hindi in Devanagari script. Negotiate accordingly.",
        "buyer": "The other player speaks and bargains only in Hindi. Negotiate accordingly.",
    },
    {
        "name": "Gujarati",
        "seller": "You are a Gujarati businessman and speak and negotiate only in Gujarati.",
        "buyer": "The other player speaks and negotiates only in Gujarati. Negotiate accordingly.",
    },
    {
        "name": "Punjabi",
        "seller": "You are a Punjabi businessman and speak and negotiate only in Punjabi.",
        "buyer": "The other player speaks and negotiates only in Punjabi. Negotiate accordingly.",
    },
    {
        "name": "Marwadi",
        "seller": "You are a Marwadi businessman and speak and negotiate only in Marwadi language. Negotiate accordingly.",
        "buyer": "The other player is a Marwadi businessman and speaks and negotiates only in Marwadi language. Negotiate accordingly.",
    },
    {
        "name": "English",
        "seller": "",
        "buyer": "",
    },
]

ITERATIONS_PER_TEST = 10

# Buy-Sell specific configurations - using same setup as buysell_main.py
GAME_CONFIG = {
    "resource_name": "X",
    "seller_cost": 40,
    "buyer_max": 60,
    "buyer_money": 1000,
}


class BuySellTestRunner:
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
            f"{model1_name}_{model2_name}_{behavior['name']}_iter_{iteration}",
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

        behaviors = SOCIAL_BEHAVIORS

        total_tests = len(model_combinations) * len(behaviors) * ITERATIONS_PER_TEST
        current_test = 0

        print(
            f"Starting {total_tests} tests for ALL behaviors..."
        )
        print(f"Model combinations: {len(model_combinations)}")
        print(f"Behaviors: {len(behaviors)}")
        print(f"Iterations per combination: {ITERATIONS_PER_TEST}")

        for model1_name, model2_name in model_combinations:
            for behavior in behaviors:
                print(
                    f"\nTesting {model1_name} (seller) vs {model2_name} (buyer) with {behavior['name']} behavior..."
                )

                for iteration in range(ITERATIONS_PER_TEST):
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


def main():
    """Main function to run the test suite"""

    # Check if OpenRouter API key is available
    if not os.environ.get("OPENROUTER_API_KEY"):
        print("ERROR: OPENROUTER_API_KEY environment variable not found!")
        print("Please set your OpenRouter API key in the .env.local file")
        return

    # Create test runner
    test_runner = BuySellTestRunner()

    # Run all tests
    results = test_runner.run_all_tests()

    print(f"\nTest suite completed successfully! Results saved to {test_runner.log_base_dir}/all_results.json")


if __name__ == "__main__":
    main()