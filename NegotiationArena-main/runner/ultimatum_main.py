import sys
from pathlib import Path

# Add current directory to Python path for module imports
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir))

from dotenv import load_dotenv

from games.ultimatum.game import MultiTurnUltimatumGame
from negotiationarena.agents.openrouter_agent import OpenRouterAgent
from negotiationarena.constants import *
from negotiationarena.game_objects.goal import UltimatumGoal
from negotiationarena.game_objects.resource import Resources

load_dotenv(".env.local")

if __name__ == "__main__":
    a1 = OpenRouterAgent(agent_name=AGENT_ONE, model="openai/gpt-3.5-turbo")
    a2 = OpenRouterAgent(agent_name=AGENT_TWO, model="openai/gpt-3.5-turbo")

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
        player_social_behaviour=[
            "You will not offer more than 40 dollars.",
            "You will reject unless the offer is at least 50 dollars.",
        ],
        player_roles=[
            f"You are {AGENT_ONE}.",
            f"You are {AGENT_TWO}.",
        ],
        log_dir="./.logs/ultimatum_multi_period",
    )

    c.run()
