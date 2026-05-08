from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from core.sampling_params import SamplingParams


@dataclass(slots=True)
class RequestOutput:
    request_id: str
    text: str
    token_ids: list[int]
    prompt_token_ids: list[int]
    finish_reason: str
    latency: float | None
    normalized_latency: float | None
    cached_prefix_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RequestState:
    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    arrival_time: float = field(default_factory=time.perf_counter)
    status: str = "waiting"
    generated_token_ids: list[int] = field(default_factory=list)
    generated_text: str = ""
    block_table: list[int] = field(default_factory=list)
    block_hashes: list[str] = field(default_factory=list)
    cached_prefix_block_ids: list[int] = field(default_factory=list)
    cached_prefix_tokens: int = 0
    num_prompt_tokens_committed: int = 0
    start_time: float | None = None
    finish_time: float | None = None
    finish_reason: str | None = None
    prompt_logits: Any | None = None
    runtime_past_key_values: Any | None = None
    pending_token_id: int | None = None
    pending_token_text: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def finished(self) -> bool:
        return self.status == "finished"

    @property
    def current_length(self) -> int:
        return len(self.prompt_token_ids) + len(self.generated_token_ids)

    @property
    def output_length(self) -> int:
        return len(self.generated_token_ids)

    @property
    def latency(self) -> float | None:
        if self.start_time is None or self.finish_time is None:
            return None
        return self.finish_time - self.start_time

    @property
    def normalized_latency(self) -> float | None:
        if self.latency is None:
            return None
        return self.latency / max(1, self.output_length)

    def mark_running(self, now: float | None = None) -> None:
        if self.status == "finished":
            return
        self.status = "running"
        if self.start_time is None:
            self.start_time = now if now is not None else time.perf_counter()

    def append_generated_token(self, token_id: int, token_text: str) -> None:
        self.generated_token_ids.append(token_id)
        self.generated_text += token_text

    def should_stop(self) -> tuple[bool, str | None]:
        if self.output_length >= self.sampling_params.max_tokens:
            return True, "length"
        for stop_text in self.sampling_params.stop or []:
            if stop_text and stop_text in self.generated_text:
                return True, "stop"
        return False, None

    def mark_finished(self, reason: str = "stop", now: float | None = None) -> None:
        self.status = "finished"
        self.finish_reason = reason
        self.finish_time = now if now is not None else time.perf_counter()

    def to_output(self) -> RequestOutput:
        return RequestOutput(
            request_id=self.request_id,
            text=self.generated_text,
            token_ids=list(self.generated_token_ids),
            prompt_token_ids=list(self.prompt_token_ids),
            finish_reason=self.finish_reason or "unknown",
            latency=self.latency,
            normalized_latency=self.normalized_latency,
            cached_prefix_tokens=self.cached_prefix_tokens,
            metadata=dict(self.metadata),
        )
