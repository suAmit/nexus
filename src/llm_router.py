import os

from dotenv import load_dotenv
from litellm import completion

load_dotenv()


class LLMRouter:
    def __init__(self):
        # Model mapping with fallbacks
        self.models = {
            "LITE": "gemini/gemini-2.5-flash-lite",
            "MID": "gemini/gemini-2.5-flash",
            "PRO": "gemini/gemini-2.5-pro",
        }
        # Secondary backup models if the primary is down
        self.fallbacks = {
            "PRO": "gemini/gemini-2.5-flash",
            "MID": "gemini/gemini-2.5-flash-lite",
            "LITE": "gemini/gemini-2.5-flash-lite",
        }

    def get_response_stream(self, tier, prompt, context=""):
        """
        Routes the prompt and returns a generator for streaming.
        Includes automated fallback logic.
        """
        model_name = self.models.get(tier, self.models["MID"])

        # Construct messages with the context from Memory
        messages = []
        if context:
            messages.append(
                {"role": "system", "content": f"Relevant background info: {context}"}
            )
        messages.append({"role": "user", "content": prompt})

        try:
            # First Attempt
            return self._execute_completion(model_name, messages)
        except Exception as e:
            # Fallback Attempt
            fallback_model = self.fallbacks.get(tier)
            print(
                f"Primary {tier} failed ({str(e)}). Switching to fallback: {fallback_model}"
            )
            return self._execute_completion(fallback_model, messages)

    def _execute_completion(self, model, messages):
        """Helper to handle the LiteLLM call."""
        response = completion(
            model=model, messages=messages, stream=True  # Enables the generator
        )
        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                yield content

    def get_model_name(self, tier):
        return self.models.get(tier, "unknown")
