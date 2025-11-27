import sys
from dotenv import load_dotenv
import inspect
from negotiationarena.agents import *
from negotiationarena.game_objects.resource import Resources
from games.simple_game.game import SimpleGame
from negotiationarena.constants import *


if __name__ == "__main__":
    a1 = GeminiAgent(
        model="gemini-2.5-pro",
        agent_name=AGENT_ONE,
    )
    a2 = GeminiAgent(
        model="gemini-2.5-pro",
        agent_name=AGENT_TWO,
    )

    c = SimpleGame(
        players=[a1, a2],
        iterations=6,
        resources_support_set=Resources({"X": 0, "Y": 0}),
        player_initial_resources=[
            Resources({"X": 25, "Y": 5}),
            Resources({"X": 0, "Y": 0}),
        ],
        player_social_behaviour=[
            "You understand English and Hindi as well as talk in Hindi.",
            "You understand Hindi and English as well as respond exclusively in Hindi.",
        ],
        player_roles=[
            f"You are {AGENT_ONE}, start by making a proposal.",
            f"You are {AGENT_TWO}, start by accepting a trade.",
        ],
        log_dir="./.logs/simple_game/",
    )

    c.run()
