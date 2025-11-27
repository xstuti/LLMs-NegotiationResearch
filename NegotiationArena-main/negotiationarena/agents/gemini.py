import os
import random
import time
from negotiationarena.agents.agents import Agent
from negotiationarena.constants import AGENT_TWO, AGENT_ONE
from copy import deepcopy
import google.generativeai as genai

class GeminiAgent(Agent):
    def __init__(
        self,
        agent_name: str,
        model="gemini-2.5-pro",
        temperature=0.7,
        max_tokens=2000,
        seed=None,
    ):
        super().__init__(agent_name)
        self.run_epoch_time_ms = str(round(time.time() * 1000))
        self.model = model
        self.conversation = []
        self.prompt_entity_initializer = "system"
        self.seed = (
            int(self.run_epoch_time_ms) + random.randint(0, 2**16)
            if seed is None
            else seed
        )
        
        # Configure Gemini
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        self.client = genai.GenerativeModel(self.model)
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
            raise "No Player 1 or Player 2 in role"

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == "client" and not isinstance(v, str):
                v = v.__class__.__name__
            setattr(result, k, deepcopy(v, memo))
        return result

    def chat(self):
        # Convert conversation to Gemini format
        prompt = ""
        for msg in self.conversation:
            if msg["role"] == "system":
                prompt += f"System: {msg['content']}\n\n"
            elif msg["role"] == "user":
                prompt += f"User: {msg['content']}\n\n"
            elif msg["role"] == "assistant":
                prompt += f"Assistant: {msg['content']}\n\n"
        
        response = self.client.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
        )
        
        return response.text

    def update_conversation_tracking(self, role, message):
        self.conversation.append({"role": role, "content": message})