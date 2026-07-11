from typing import Any, List, TypedDict
from skyrl_agent.integrations.base import (
    AsyncInferBackend,
    GeneratorOutput,
    GeneratorInput,
    register_backend,
    BackendSpec,
)
import omegaconf
from loguru import logger
import os
import aiohttp

from skyrl_agent.agents.rollout_diagnostic_utils import extract_exact_output_tokens


class OpenAIBackendConfig(TypedDict):
    model_name: str
    api_url: str
    require_token_ids: bool


class OpenAIBackend(AsyncInferBackend):
    def __init__(self, infer_engine: Any, cfg: OpenAIBackendConfig, tokenizer: Any = None):
        assert os.environ.get("OPENAI_API_KEY") is not None, "OPENAI_API_KEY is not set"
        self.model_name = cfg["model_name"]
        self.api_url = cfg["api_url"]
        self.tokenizer = tokenizer
        self.require_token_ids = cfg.get("require_token_ids", False)
        try:
            self.model_max_len = cfg["model_max_len"]
        except omegaconf.errors.ConfigKeyError:
            logger.info("`model_max_len` not provided, using `max_position_embeddings` from the model config")
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(self.model_name)
            self.model_max_len = config.max_position_embeddings
            logger.info(f"Using `model_max_len` {self.model_max_len}")

    async def async_generate_prompts(self, prompts: str, sampling_params: dict, **kwargs) -> str:
        # NOTE: In some agents like OpenHands, the generate calls are from a different thread, so the session needs to be created in the same thread
        # TODO: support long lived session depending on the task
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:

            headers = {"Content-Type": "application/json"}
            if not isinstance(sampling_params, dict):
                sampling_params = dict(sampling_params)
            payload = sampling_params.copy()
            payload["model"] = self.model_name

            # TODO: user template might be applied twice here. need to double check.
            payload["messages"] = {"role": "user", "content": prompts}
            output = await session.post(f"{self.api_url}/v1/chat/completions", json=payload, headers=headers)
            output = await output.json()

            try:
                return output["choices"][0]["message"]["content"]
            except Exception as e:
                logger.info(f"Errored out while extracting first response from output {output} with exception {str(e)}")

    async def async_generate_ids(self, input_ids: List[int], sampling_params: dict, **kwargs) -> str:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None)) as session:
            headers = {"Content-Type": "application/json"}
            if not isinstance(sampling_params, dict):
                sampling_params = dict(sampling_params)
            payload = sampling_params.copy()
            payload["model"] = self.model_name
            if self.require_token_ids:
                payload["return_token_ids"] = True
            # payload["max_tokens"] = self.model_max_len - len(input_ids) - 1
            print(f"max tokens: {payload['max_tokens']}")

            payload["prompt"] = input_ids
            output = await session.post(f"{self.api_url}/v1/completions", json=payload, headers=headers)
            output = await output.json()

        choice = output["choices"][0]
        response_text = choice["text"]
        if self.require_token_ids:
            output_tokens = extract_exact_output_tokens(choice)
        else:
            output_tokens = choice.get("token_ids")
            if output_tokens is None:
                output_tokens = self.tokenizer.encode(response_text, add_special_tokens=False) if self.tokenizer else []
        meta_info = {
            "output_tokens": output_tokens,
            "finish_reason": choice["finish_reason"],
            "logprobs": None,
        }

        return response_text, meta_info


class OpenAIGeneratorOutput(GeneratorOutput):
    def __init__(self, result: Any):
        self.result = result


class OpenAIGeneratorInput(GeneratorInput):
    def __init__(self, input_batch: Any):
        self.input_batch = input_batch


register_backend(
    "openai_server",
    BackendSpec(
        infer_backend_cls=OpenAIBackend,
        generator_output_cls=OpenAIGeneratorOutput,
        generator_input_cls=OpenAIGeneratorInput,
    ),
)
