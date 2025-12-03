import os
import random
import time
from copy import deepcopy

import google.generativeai as genai

from negotiationarena.agents.agents import Agent
from negotiationarena.constants import AGENT_ONE, AGENT_TWO


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
            raise ValueError("No Player 1 or Player 2 in role")

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

        response = None
        try:
            response = self.client.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                ),
            )

            # Try to get the text response
            return response.text

        except ValueError as e:
            # Check if this is a safety-related error by examining the error message
            error_str = str(e)

            # Check if we have a response with candidates to examine finish_reason
            if (
                response is not None
                and hasattr(response, "candidates")
                and response.candidates
                and "finish_reason" in error_str
            ):
                candidate = response.candidates[0]
                finish_reason = candidate.finish_reason

                # Handle different finish reasons based on the numeric values
                # SAFETY = 3, RECITATION = 4, LANGUAGE = 6, MAX_TOKENS = 2
                if finish_reason == 3:  # SAFETY
                    print(
                        f"Warning: Gemini safety filters triggered for agent {self.agent_name}"
                    )
                    return "I apologize, but I cannot provide a response to that request due to safety guidelines. Let me try to continue the conversation in a different way."
                elif finish_reason == 4:  # RECITATION
                    print(
                        f"Warning: Gemini recitation filter triggered for agent {self.agent_name}"
                    )
                    return "I need to rephrase my response to avoid potential copyright issues. Let me continue our discussion."
                elif finish_reason == 6:  # LANGUAGE
                    print(
                        f"Warning: Gemini language filter triggered for agent {self.agent_name}"
                    )
                    return "I'm having difficulty with the language format. Let me respond in a way that works better."
                elif finish_reason == 2:  # MAX_TOKENS
                    print(
                        f"Warning: Gemini max tokens reached for agent {self.agent_name}"
                    )
                    return "My response was cut short due to length limits. Let me provide a more concise response."
                else:
                    print(
                        f"Warning: Gemini response issue (finish_reason: {finish_reason}) for agent {self.agent_name}"
                    )
                    return "I'm experiencing technical difficulties with my response. Let me try to continue our conversation."

            # If we can't handle the error gracefully, provide a generic fallback
            print(f"Error in Gemini agent {self.agent_name}: {error_str}")
            return "I'm sorry, I encountered an issue generating a response. Let me try to continue our conversation."

        except Exception as e:
            # Handle any other unexpected errors
            print(f"Unexpected error in Gemini agent {self.agent_name}: {str(e)}")
            return "I'm experiencing technical difficulties. Let me try to continue our conversation."

    def update_conversation_tracking(self, entity, message):
        self.conversation.append({"role": entity, "content": message})
