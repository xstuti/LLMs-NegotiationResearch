from dotenv import load_dotenv
from negotiationarena.agents.chatgpt import ChatGPTAgent
from negotiationarena.agents import GeminiAgent
from negotiationarena.game_objects.resource import Resources
from negotiationarena.game_objects.goal import UltimatumGoal
from games.ultimatum.game import MultiTurnUltimatumGame
from negotiationarena.constants import *

load_dotenv(".env.local")

if __name__ == "__main__":
    a1 = GeminiAgent(agent_name=AGENT_ONE, model="gemini-2.5-flash")
    a2 = GeminiAgent(agent_name=AGENT_TWO, model="gemini-2.5-flash")

    c = MultiTurnUltimatumGame(
        players=[a1, a2],
        iterations=6,
        resources_support_set=Resources({"Dollars": 0}),
        player_goals=[
            UltimatumGoal(),
            UltimatumGoal(),
        ],
        player_initial_resources=[
            Resources({"Dollars": 100}),
            Resources({"Dollars": 0}),
        ],
        player_social_behaviour=["", ""],
        player_roles=[
            f"You are {AGENT_ONE}.",
            f"You are {AGENT_TWO}.",
        ],
        log_dir="./.logs/ultimatum_multi_period",
    )

    c.run()
