import os
from copy import deepcopy
from negotiationarena.agents.agents import Agent
from negotiationarena.constants import AGENT_ONE, AGENT_TWO
from openai import OpenAI

class OpenRouterAgent(Agent):
    def __init__(
        self,
        agent_name: str,
        model="openai/gpt-3.5-turbo",      # You can use any OpenRouter model string!
        temperature=0.7,
        max_tokens=400,
        seed=None,
    ):
        super().__init__(agent_name)
        self.run_epoch_time_ms = str(round(__import__('time').time() * 1000))
        self.model = model
        self.conversation = []
        self.prompt_entity_initializer = "system"
        self.seed = (
            int(self.run_epoch_time_ms) + __import__('random').randint(0, 2**16)
            if seed is None
            else seed
        )
        # Use OpenRouter API endpoint and your OpenRouter key
        self.client = OpenAI(
            api_key=os.environ.get("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1"
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    def init_agent(self, system_prompt, role):
        if AGENT_ONE in self.agent_name:
            self.update_conversation_tracking(
                self.prompt_entity_initializer, system_prompt
            )
            self.update_conversation_tracking("user", role)
        elif AGENT_TWO in self.agent_name:
            system_prompt = system_prompt + role
            self.update_conversation_tracking(
                self.prompt_entity_initializer, system_prompt
            )
        else:
            raise ValueError("No Player 1 or Player 2 in role")

    def chat(self):
        chat = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )
        return chat.choices[0].message.content

    def update_conversation_tracking(self, role, message):
        self.conversation.append({"role": role, "content": message})

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "client" and not isinstance(v, str):
                v = v.__class__.__name__
            setattr(result, k, deepcopy(v, memo))
        return result