from __future__ import annotations

from core.llm import MiniLLM
from core.model_runner import ModelRunner
from core.sampling_params import SamplingParams


def test_fallback_engine_generates_text() -> None:
    llm = MiniLLM(
        model=ModelRunner.FALLBACK_MODEL_NAME,
        block_size=4,
        num_blocks=64,
        max_batch_size=2,
    )
    outputs = llm.generate(["hello"], SamplingParams(max_tokens=4))

    assert len(outputs) == 1
    assert outputs[0].text
    assert outputs[0].finish_reason == "length"
    assert llm.stats()["num_total_blocks"] == 64
