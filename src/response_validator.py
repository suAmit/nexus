import re

from litellm import completion


class ResponseValidator:
    def __init__(self):
        # Legacy patterns kept as a fast first-pass filter
        self.refusal_patterns = [
            r"i am sorry",
            r"as an ai",
            r"i cannot fulfill",
        ]

    def validate(self, prompt, response, tier):
        """
        Performs a multi-stage validation: Regex -> Technical -> LLM Evaluator.
        Returns (is_valid, reason)
        """
        # 1. Fast Pass: Regex Refusal Check
        for pattern in self.refusal_patterns:
            if re.search(pattern, response.lower()):
                return False, "Refusal Keyword Detected"

        # 2. Technical Validation: Check for Code if requested
        code_keywords = ["code", "python", "script", "function", "javascript"]
        if any(word in prompt.lower() for word in code_keywords):
            if "```" not in response:
                return False, "Missing Code Blocks in Technical Query"

            # Basic Syntax Check: Ensure code blocks aren't empty
            code_blocks = re.findall(r"```(?:\w+)?\n(.*?)\n```", response, re.DOTALL)
            if any(len(block.strip()) < 5 for block in code_blocks):
                return False, "Empty or Malformed Code Block"

        # 3. Intelligent Pass: LLM-based Evaluation (Using LITE Tier)
        # We only do this for MID/PRO to save costs on simple LITE queries
        if tier in ["MID", "PRO"]:
            is_helpful = self._llm_verify_helpfulness(prompt, response)
            if not is_helpful:
                return False, "Semantic Validation Failed (Unhelpful/Off-topic)"

        return True, "Valid"

    def _llm_verify_helpfulness(self, prompt, response):
        """Uses a cheap model to verify if the response actually addresses the prompt."""
        check_prompt = f"""
        User Prompt: {prompt}
        Assistant Response: {response}
        
        Does the assistant response directly answer or fulfill the user prompt? 
        Answer ONLY 'YES' or 'NO'.
        """
        try:
            # Using the Flash-Lite model for speed and low cost
            res = completion(
                model="gemini/gemini-2.5-flash-lite",
                messages=[{"role": "user", "content": check_prompt}],
                max_tokens=5,
            )
            decision = res.choices[0].message.content.strip().upper()
            return "YES" in decision
        except:
            return True  # Default to valid if the guard model fails

    def get_upcycle_tier(self, current_tier):
        """Determines the next step up if a model fails."""
        mapping = {"LITE": "MID", "MID": "PRO", "PRO": "PRO"}
        return mapping.get(current_tier, "MID")
