import time
from typing import Any

import openai
import tiktoken
from openai import OpenAIError, RateLimitError

from src.app.config.config import config
from src.core.contracts.llm_client import LLMClient


class OpenAIClient(LLMClient):
    def __init__(
        self,
        model_name: str = "gpt-5-mini", # "gpt-4.1-mini",
        temperature: float = 0.1,
        max_tokens: int = 200,
    ) -> None:
        self.model_name = model_name
        self._model: Any = None
        self.input_tokens_limit = 8_192
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def model(self) -> Any:
        if self._model is None:
            openai.api_key = config.openai_api_key
            self._model = openai

        return self._model

    def check_input_limit(self, prompt: str) -> bool:
        try:
            tokenizer = tiktoken.encoding_for_model(self.model_name)
        except KeyError:
            tokenizer = tiktoken.get_encoding("cl100k_base")

        if tokenizer:
            tokens = tokenizer.encode(prompt)
            tokens_len = len(tokens)
            print(f"Number of tokens in prompt: {tokens_len}, maximum: {self.input_tokens_limit}")  # noqa: T201
            return tokens_len <= self.input_tokens_limit
        return False

    def run(self, prompt: str, instructions: str | None = None) -> str:
        messages = []
        if instructions:
            messages.append({"role": "system", "content": instructions})

        messages.append({"role": "user", "content": prompt})

        chat = self.model.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        answer = chat.choices[0].message.content

        return str(answer)

    def safe_run(
        self,
        prompt: str,
        instructions: str | None = None,
        max_retries: int = 5,
        base_delay: int = 10,
    ) -> str:
        for attempt in range(max_retries):
            try:  # noqa: PERF203
                return self.run(prompt, instructions)
            except RateLimitError:
                wait_time = base_delay * (attempt + 1)
                print(  # noqa: T201
                    f"Rate limit hit. Waiting {wait_time} s before retrying... ({attempt + 1}/{max_retries})"
                )
                time.sleep(wait_time)
            except OpenAIError as e:
                print(f"OpenAI error: {e}")  # noqa: T201
                break
            except Exception as e:
                print(f"Unexpected error: {e}")  # noqa: T201
                break
        print("Max retries reached. Skipping this request.")  # noqa: T201
        return ""

    def generate(self, prompt: str, instructions: str | None = None) -> str:
        return self.safe_run(prompt, instructions)
