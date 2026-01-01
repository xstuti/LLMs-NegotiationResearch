"""
Trading Game – Restartable Version (No Gemini, No Rate Limits)
"""

import sys
import os
from itertools import product
from dotenv import load_dotenv

sys.path.append(".")

from negotiationarena.agents.openrouter_agent import OpenRouterAgent
from negotiationarena.game_objects.resource import Resources
from negotiationarena.game_objects.goal import ResourceGoal
from games.trading_game.game import TradingGame
from negotiationarena.constants import AGENT_ONE, AGENT_TWO

load_dotenv(".env.local")

# --------------------------------------------------
# FINAL MODEL LIST (5 models)
# --------------------------------------------------

MODELS = {
    "GPT-3.5": (OpenRouterAgent, "openai/gpt-3.5-turbo"),
    "GPT-4o": (OpenRouterAgent, "openai/chatgpt-4o-latest"),
    "Claude-3-Haiku": (OpenRouterAgent, "anthropic/claude-3-haiku"),
    "Claude-3.5-Haiku": (OpenRouterAgent, "anthropic/claude-3.5-haiku"),
    #"GPT-OSS-20B": (OpenRouterAgent, "openai/gpt-oss-20b:free"),
}

# --------------------------------------------------
# Social behaviors (unchanged)
# --------------------------------------------------

SOCIAL_BEHAVIORS = {
    "Gujarati": [
        "You speak and negotiate only in Gujarati. Negotiate accordingly.",
        "The other player speaks and negotiates only in Gujarati. Negotiate accordingly.",
    ],
    "Hindi": [
        "You speak and negotiate only in Hindi. Negotiate accordingly.",
        "The other player speaks and bargains only in Hindi. Negotiate accordingly.",
    ],
    "Marwadi": [
        "You speak and negotiate only in Marwadi language. Negotiate accordingly.",
        "The other player speaks and negotiates only in Marwadi language. Negotiate accordingly.",
    ],
    "Punjabi": [
        "You speak and negotiate only in Punjabi. Negotiate accordingly.",
        "The other player speaks and negotiates only in Punjabi. Negotiate accordingly.",
    ],
    "English": ["",""],
}

ITERATIONS_PER_COMBO = 10
LOG_BASE_DIR = "./.logs/final_final_trading/"
os.makedirs(LOG_BASE_DIR, exist_ok=True)

# --------------------------------------------------
# Helper functions
# --------------------------------------------------

def create_agent(agent_class, agent_name, model_string):
    return agent_class(agent_name=agent_name, model=model_string)


def build_log_dir(model1, model2, behavior, iteration):
    safe_behavior = behavior.replace(" ", "_")
    return f"{LOG_BASE_DIR}{model1}_{model2}_{safe_behavior}_iter{iteration}/"


# --------------------------------------------------
# Experiment runner
# --------------------------------------------------

def run_experiment(model1_name, model2_name, behavior_name, iteration_num):
    log_dir = build_log_dir(model1_name, model2_name, behavior_name, iteration_num + 1)
    os.makedirs(log_dir, exist_ok=True)

    agent_class1, model_string1 = MODELS[model1_name]
    agent_class2, model_string2 = MODELS[model2_name]
    social_behavior = SOCIAL_BEHAVIORS[behavior_name]

    a1 = create_agent(agent_class1, AGENT_ONE, model_string1)
    a2 = create_agent(agent_class2, AGENT_TWO, model_string2)

    print(
        f"\n➡️ Running {model1_name} vs {model2_name} | "
        f"{behavior_name} | iteration {iteration_num + 1}"
    )

    game = TradingGame(
        players=[a1, a2],
        iterations=6,
        resources_support_set=Resources({"X": 0, "Y": 0}),
        player_goals=[
            ResourceGoal({"X": 15, "Y": 15}),
            ResourceGoal({"X": 15, "Y": 15}),
        ],
        player_initial_resources=[
            Resources({"X": 25, "Y": 5}),
            Resources({"X": 5, "Y": 25}),
        ],
        player_social_behaviour=social_behavior,
        player_roles=[
            f"You are {AGENT_ONE}, start by making a proposal.",
            f"You are {AGENT_TWO}, start by responding to a trade.",
        ],
        log_dir=log_dir,
    )

    try:
        game.run()
        print(f"✓ Completed {model1_name} vs {model2_name}, {behavior_name}, iter {iteration_num + 1}")
        return True
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


# --------------------------------------------------
# MAIN — skip completed runs, no self-comparison
# --------------------------------------------------

def main():
    model_names = list(MODELS.keys())
    behaviors = list(SOCIAL_BEHAVIORS.keys())

    print(f"\n🎯 Running missing experiments for {len(model_names)} models.\n")

    for m1, m2, behavior in product(model_names, model_names, behaviors):

        # Skip comparing a model with itself
        if m1 == m2:
            continue

        for iteration in range(10, 10+ ITERATIONS_PER_COMBO):

            log_dir = build_log_dir(m1, m2, behavior, iteration + 1)

            # Skip completed runs
            if os.path.exists(log_dir) and len(os.listdir(log_dir)) > 0:
                print(f"⏭️ Skipping completed: {log_dir}")
                continue

            print(f"🔄 Running: {log_dir}")
            run_experiment(m1, m2, behavior, iteration)


if __name__ == "__main__":
    main()
