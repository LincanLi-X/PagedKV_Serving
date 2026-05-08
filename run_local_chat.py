from __future__ import annotations

import argparse
import json

from core.llm import MiniLLM
from core.model_runner import ModelRunner
from core.sampling_params import SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local PagedKV chat inference.")
    parser.add_argument("--model", default=ModelRunner.FALLBACK_MODEL_NAME)
    parser.add_argument("--prompt", default="Explain paged KV cache in one paragraph.")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=256)
    args = parser.parse_args()

    llm = MiniLLM(
        model=args.model,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
    )
    output = llm.chat(
        [{"role": "user", "content": args.prompt}],
        SamplingParams(max_tokens=args.max_tokens),
    )
    print(output.text)
    print(json.dumps(llm.stats(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
