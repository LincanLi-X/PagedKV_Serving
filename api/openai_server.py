from __future__ import annotations

from functools import lru_cache

from api.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    ChatMessage,
)
from core.llm import MiniLLM
from core.sampling_params import SamplingParams


@lru_cache(maxsize=8)
def get_llm(
    model: str,
    block_size: int = 16,
    num_blocks: int = 512,
    max_batch_size: int = 4,
) -> MiniLLM:
    return MiniLLM(
        model=model,
        block_size=block_size,
        num_blocks=num_blocks,
        max_batch_size=max_batch_size,
    )


def complete_chat(
    request: ChatCompletionRequest,
    block_size: int = 16,
    num_blocks: int = 512,
    max_batch_size: int = 4,
) -> ChatCompletionResponse:
    if request.stream:
        raise ValueError("streaming is out of scope for this course prototype")
    llm = get_llm(request.model, block_size, num_blocks, max_batch_size)
    output = llm.chat(
        [message.model_dump() for message in request.messages],
        sampling_params=SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=request.stop or [],
        ),
    )
    return ChatCompletionResponse(
        id=f"chatcmpl-{output.request_id}",
        model=request.model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=ChatMessage(role="assistant", content=output.text),
                finish_reason=output.finish_reason,
            )
        ],
        usage=ChatCompletionUsage(
            prompt_tokens=len(output.prompt_token_ids),
            completion_tokens=len(output.token_ids),
            total_tokens=len(output.prompt_token_ids) + len(output.token_ids),
        ),
        metrics=llm.stats(),
    )


def create_app(
    model: str = "fallback-local-model",
    block_size: int = 16,
    num_blocks: int = 512,
    max_batch_size: int = 4,
):
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="PagedKV_Serving OpenAI-compatible API")

    @app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
    def chat_completions(request: ChatCompletionRequest):
        try:
            if not request.model:
                request.model = model
            return complete_chat(request, block_size, num_blocks, max_batch_size)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/health")
    def health():
        return {"status": "ok", "model": model}

    return app
