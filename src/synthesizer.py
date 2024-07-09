import json
import os
import random
from typing import Any, List, Optional

import g4f
from g4f.client import Client
from g4f.Provider import DeepInfra, FreeGpt, HuggingChat

from src.schemas import SynthesizeParameters


class Synthesizer:
    def __init__(
        self,
        google_cookies_dir: Optional[str] = None,
        huggingface_tokens_dir: Optional[str] = None,
        deepinfra_tokens_dir: Optional[str] = None,
    ):
        self.provider_list = ["free-gpt", "hugging-chat", "deep-infra"]
        self.google_cookies_dir = google_cookies_dir
        self.huggingface_tokens_dir = huggingface_tokens_dir
        self.deepinfra_tokens_dir = deepinfra_tokens_dir
        self.client = Client()
        self.google_cookies = []
        self.huggingface_tokens = []
        self.deepinfra_tokens = []
        self._init_huggingface_tokens()
        self._init_google_cookies()
        self._init_deepinfra_tokens()

    def _init_tokens(self, directory_path: Optional[str], target_list: List[Any]) -> None:
        try:
            for file_path in os.listdir(directory_path):
                absolute_file_path = os.path.join(directory_path, file_path)
                with open(absolute_file_path, "r") as f:
                    json_content = json.load(f)
                target_list.append(json_content)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Error initializing tokens: {e}")

    def _init_google_cookies(self):
        self._init_tokens(self.google_cookies_dir, self.google_cookies)

    def _init_huggingface_tokens(self):
        self._init_tokens(self.huggingface_tokens_dir, self.huggingface_tokens)

    def _init_deepinfra_tokens(self):
        self._init_tokens(self.deepinfra_tokens_dir, self.deepinfra_tokens)

    def _random_values(self, target_list: List[Any]) -> Any:
        random.seed(random.randint(1, 10000))
        random_index = random.randint(0, len(target_list) - 1)
        return target_list[random_index]

    def completion(self, parameters: SynthesizeParameters, provider: Optional[str]):

        if provider == "free-gpt":
            response = self.client.chat.completions.create(
                provider=FreeGpt,
                model=g4f.models.default,
                messages=[
                    {"role": "system", "content": parameters.system_prompt},
                    {"role": "user", "content": parameters.user_prompt},
                ],
                api_key=parameters.api_key,
            )

        elif provider == "hugging-chat":
            response = self.client.chat.completions.create(
                provider=HuggingChat,
                model="CohereForAI/c4ai-command-r-plus",
                messages=[
                    {"role": "system", "content": parameters.system_prompt},
                    {"role": "user", "content": parameters.user_prompt},
                ],
                api_key=parameters.api_key,
            )

        elif provider == "deep-infra":
            response = self.client.chat.completions.create(
                provider=DeepInfra,
                model=g4f.models.default,
                messages=[
                    {"role": "system", "content": parameters.system_prompt},
                    {"role": "user", "content": parameters.user_prompt},
                ],
                api_key=parameters.api_key,
            )

        else:
            raise ValueError(f"Unknown provider {provider}")

        return response.choices[0].message.content

    def synthesize(self, parameters: SynthesizeParameters) -> str:
        random_provider = self._random_values(self.provider_list)

        if random_provider == "deep-infra":
            random_api_key = self._random_values(self.deepinfra_tokens)["api_key"]
            parameters.api_key = random_api_key
            return self.completion(parameters, random_provider)

        elif random_provider == "hugging-chat":
            random_api_key = self._random_values(self.huggingface_tokens)
            parameters.api_key = random_api_key["api_key"]
            return self.completion(parameters, random_provider)

        elif random_provider == "free-gpt":
            return self.completion(parameters, random_provider)
