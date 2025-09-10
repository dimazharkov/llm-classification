import time
from typing import Optional

from google import genai
from google.genai import types

from src.app.config.config import config
from src.core.contracts.llm_client import LLMClient


class GeminiClient(LLMClient):
    def __init__(self, model_name: str = "gemini-2.0-flash", temperature: float = 0.1, max_tokens: int = 200):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.input_tokens_limit = 8192
        self._client: Optional[genai.Client] = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            api_key = config.gemini_api_key
            if not api_key:
                raise ValueError("GEMINI_API_KEY is not set in the environment.")
            self._client = genai.Client(api_key=api_key)
        return self._client

    def run(self, prompt: str) -> str:
        config_params = {
            "temperature": self.temperature,
        }

        client_config = types.GenerateContentConfig(**config_params)
        response = self.client.models.generate_content(model=self.model_name, contents=prompt, config=client_config)

        if response.text is None:
            raise RuntimeError("Gemini response did not contain text.")

        return response.text

    def safe_run(self, prompt: str, max_retries: int = 5, base_delay: int = 10) -> str:
        for attempt in range(max_retries):
            try:
                return self.run(prompt)
            except Exception as e:
                wait_time = base_delay * (attempt + 1)
                print(  # noqa: T201
                    f"Error: {e}. Waiting {wait_time} seconds before retrying... ({attempt + 1}/{max_retries})",
                )
                time.sleep(wait_time)
        print("Max retries reached. Skipping this request.")  # noqa: T201
        return ""

    def generate(self, prompt: str, instructions: Optional[str] = None) -> str:
        return self.safe_run(prompt)
