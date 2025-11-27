import sys

sys.path.append(".")
from dotenv import load_dotenv
import inspect
from negotiationarena.agents.gemini import GeminiAgent
#from negotiationarena.agents.chatgpt import ChatGPTAgent
from negotiationarena.agents.openrouter_agent import OpenRouterAgent
from negotiationarena.agents.agent_behaviours import (
    SelfCheckingAgent,
    ReasoningAgent,
)
from negotiationarena.game_objects.resource import Resources
from negotiationarena.game_objects.goal import ResourceGoal
from games.trading_game.game import TradingGame
from games.trading_game.interface import TradingGameDefaultParser
from negotiationarena.constants import *

load_dotenv(".env.local")


if __name__ == "__main__":
    for i in range(1):
        a1 = OpenRouterAgent(
            agent_name=AGENT_ONE,
            model="openai/gpt-3.5-turbo",
        )
        a2 = OpenRouterAgent(
            agent_name=AGENT_TWO,
            model="openai/chatgpt-4o-latest",
        )

        c = TradingGame(
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
            player_social_behaviour=["You speak and bargain only in Hindi in Devanagari script. Negotiate accordingly.", "The other player speaks and bargains only in Hindi. Negotiate accordingly."],
            player_roles=[
                f"You are {AGENT_ONE}, start by making a proposal.",
                f"You are {AGENT_TWO}, start by responding to a trade.",
            ],
            log_dir="./.logs/trading/",
        )

        c.run()
