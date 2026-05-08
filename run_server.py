from __future__ import annotations

import argparse
import json

from api.openai_server import complete_chat, create_app
from api.schemas import ChatCompletionRequest, ChatMessage
from core.model_runner import ModelRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the PagedKV OpenAI-compatible API.")
    parser.add_argument("--model", default=ModelRunner.FALLBACK_MODEL_NAME)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        response = complete_chat(
            ChatCompletionRequest(
                model=args.model,
                messages=[ChatMessage(role="user", content="Say one sentence about paging.")],
                max_tokens=args.max_tokens,
            ),
            block_size=args.block_size,
            num_blocks=args.num_blocks,
        )
        print(json.dumps(response.model_dump(), indent=2, sort_keys=True))
        return

    import uvicorn

    app = create_app(
        model=args.model,
        block_size=args.block_size,
        num_blocks=args.num_blocks,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
