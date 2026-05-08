from __future__ import annotations

from typing import Any

from core.chat_template import messages_to_prompt
from core.engine import MiniEngine
from core.model_runner import ModelRunner
from core.request import RequestOutput
from core.sampling_params import SamplingParams


class MiniLLM:
    """User-facing local inference wrapper."""

    def __init__(
        self,
        model: str = ModelRunner.FALLBACK_MODEL_NAME,
        block_size: int = 16,
        num_blocks: int = 512,
        max_batch_size: int = 4,
        enable_prefix_cache: bool = True,
        metrics_logger: Any | None = None,
    ) -> None:
        self.model_runner = ModelRunner(model_name=model)
        self.engine = MiniEngine(
            model_runner=self.model_runner,
            block_size=block_size,
            num_blocks=num_blocks,
            max_batch_size=max_batch_size,
            enable_prefix_cache=enable_prefix_cache,
            metrics_logger=metrics_logger,
        )

    def generate(
        self,
        prompts: list[str],
        sampling_params: SamplingParams | None = None,
    ) -> list[RequestOutput]:
        params = sampling_params or SamplingParams()
        request_ids = [self.engine.add_request(prompt, params) for prompt in prompts]
        return self.engine.run_to_completion(request_ids)

    def chat(
        self,
        messages: list[dict[str, Any]],
        sampling_params: SamplingParams | None = None,
    ) -> RequestOutput:
        prompt = messages_to_prompt(self.model_runner.tokenizer, messages)
        return self.generate([prompt], sampling_params=sampling_params)[0]

    def stats(self) -> dict[str, int | float]:
        return self.engine.stats()
