from __future__ import annotations

from collections import OrderedDict
from typing import Any
from uuid import uuid4

from core.model_runner import ModelRunner
from core.paged_kv_cache import PagedKVCache
from core.prefix_cache import PrefixCache
from core.request import RequestOutput, RequestState
from core.sampling_params import SamplingParams
from core.scheduler import Scheduler


class MiniEngine:
    """Single-process engine tying scheduler, prefix cache, KV cache, and model."""

    def __init__(
        self,
        model_runner: ModelRunner,
        block_size: int = 16,
        num_blocks: int = 512,
        max_batch_size: int = 4,
        enable_prefix_cache: bool = True,
        metrics_logger: Any | None = None,
    ) -> None:
        self.model_runner = model_runner
        self.kv_cache = PagedKVCache(block_size=block_size, num_blocks=num_blocks)
        self.scheduler = Scheduler(max_batch_size=max_batch_size)
        self.prefix_cache = PrefixCache(block_size=block_size) if enable_prefix_cache else None
        if self.prefix_cache is not None:
            self.kv_cache.register_eviction_callback(self.prefix_cache.evict_block)
        self.requests: OrderedDict[str, RequestState] = OrderedDict()
        self.completed_outputs: dict[str, RequestOutput] = {}
        self.metrics_logger = metrics_logger
        self.step_index = 0

    def add_request(self, prompt: str, sampling_params: SamplingParams) -> str:
        request_id = f"req-{uuid4().hex[:12]}"
        prompt_token_ids = self.model_runner.tokenize(prompt)
        request = RequestState(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )
        self.kv_cache.attach_request(request_id, request.block_table)
        self.requests[request_id] = request
        self.scheduler.add_request(request)
        return request_id

    def _register_new_full_blocks(self, request: RequestState) -> None:
        if self.prefix_cache is None:
            return
        for index, block_id in enumerate(request.block_table):
            if not self.kv_cache.block_is_full(block_id):
                break
            if index < len(request.block_hashes):
                continue
            parent_hash = request.block_hashes[index - 1] if index > 0 else None
            block_hash = self.prefix_cache.register_full_block(
                block_id=block_id,
                block_tokens=self.kv_cache.block_tokens(block_id),
                parent_hash=parent_hash,
            )
            request.block_hashes.append(block_hash)
            self.kv_cache.set_cache_key(block_id, block_hash)

    def _materialize_plans_with_past(self, plans, past_key_values) -> None:
        for plan in plans:
            kv_slice = self.model_runner.slice_past(
                past_key_values,
                plan.request_position_start,
                plan.request_position_end,
            )
            self.kv_cache.append_kv_to_block(plan.block_id, kv_slice)

    def _prefill_request(self, request: RequestState) -> None:
        cached_block_ids: list[int] = []
        cached_block_hashes: list[str] = []
        cached_tokens = 0
        past = None
        if self.prefix_cache is not None:
            match = self.prefix_cache.lookup_longest_prefix(request.prompt_token_ids)
            cached_block_ids = match.block_ids
            cached_block_hashes = match.block_hashes
            cached_tokens = match.matched_tokens
        if cached_block_ids:
            self.kv_cache.acquire_cached_blocks(request.request_id, cached_block_ids)
            request.cached_prefix_block_ids = list(cached_block_ids)
            request.block_hashes = list(cached_block_hashes)
            request.cached_prefix_tokens = cached_tokens
            past = self.kv_cache.gather_past_key_values(cached_block_ids)

        remaining_prompt_tokens = request.prompt_token_ids[cached_tokens:]
        if not remaining_prompt_tokens:
            remaining_prompt_tokens = request.prompt_token_ids[-1:]
        logits, past = self.model_runner.prefill(remaining_prompt_tokens, past_key_values=past)
        plans = self.kv_cache.append_tokens(
            request.request_id,
            request.block_table,
            remaining_prompt_tokens,
        )
        self._materialize_plans_with_past(plans, past)
        self._register_new_full_blocks(request)
        request.num_prompt_tokens_committed = len(request.prompt_token_ids)
        request.runtime_past_key_values = past
        request.prompt_logits = logits

    def _sample_from_logits(self, request: RequestState, logits) -> None:
        token_id = self.model_runner.sample_next_token(logits, request.sampling_params)
        if self.model_runner.eos_token_id is not None and token_id == self.model_runner.eos_token_id:
            request.mark_finished("stop")
            return
        token_text = self.model_runner.decode_token(token_id)
        request.append_generated_token(token_id, token_text)
        should_stop, reason = request.should_stop()
        if should_stop:
            request.mark_finished(reason or "stop")
            return
        request.pending_token_id = token_id
        request.pending_token_text = token_text

    def _decode_request(self, request: RequestState) -> None:
        if request.pending_token_id is None:
            return
        logits, past = self.model_runner.decode(
            request.pending_token_id,
            request.runtime_past_key_values,
        )
        plans = self.kv_cache.append_tokens(
            request.request_id,
            request.block_table,
            [request.pending_token_id],
        )
        self._materialize_plans_with_past(plans, past)
        self._register_new_full_blocks(request)
        request.runtime_past_key_values = past
        request.pending_token_id = None
        request.pending_token_text = ""
        self._sample_from_logits(request, logits)

    def _advance_request(self, request: RequestState) -> None:
        if request.prompt_logits is None:
            self._prefill_request(request)
            self._sample_from_logits(request, request.prompt_logits)
        elif not request.finished:
            self._decode_request(request)

    def _log_step(self, event: str, request: RequestState | None = None) -> None:
        if self.metrics_logger is None:
            return
        record = {
            "event": event,
            "step_index": self.step_index,
            "running_requests": len(self.scheduler.running),
            "waiting_requests": len(self.scheduler.waiting),
            **self.kv_cache.memory_stats(),
        }
        if self.prefix_cache is not None:
            record.update({f"prefix_{k}": v for k, v in self.prefix_cache.stats().items()})
        if request is not None:
            record.update(
                {
                    "request_id": request.request_id,
                    "prompt_len": len(request.prompt_token_ids),
                    "output_len": request.output_length,
                    "cached_prefix_tokens": request.cached_prefix_tokens,
                }
            )
        self.metrics_logger.log(record)

    def step(self) -> list[RequestOutput]:
        self.step_index += 1
        self.scheduler.admit_requests()
        self._log_step("before_step")
        completed: list[RequestOutput] = []
        for request in self.scheduler.active_requests():
            if not request.finished:
                self._advance_request(request)
        for request in list(self.scheduler.active_requests()):
            if not request.finished:
                continue
            self._log_step("request_finished", request)
            self.kv_cache.free_request(request.request_id)
            self.scheduler.finish_request(request.request_id)
            output = request.to_output()
            self.completed_outputs[request.request_id] = output
            completed.append(output)
        self._log_step("after_step")
        return completed

    def run_to_completion(self, request_ids: list[str] | None = None) -> list[RequestOutput]:
        pending = set(request_ids) if request_ids is not None else set(self.requests)
        while pending:
            completed = self.step()
            for output in completed:
                pending.discard(output.request_id)
        ordered_ids = request_ids if request_ids is not None else list(self.requests)
        return [self.completed_outputs[request_id] for request_id in ordered_ids]

    def stats(self) -> dict[str, int | float]:
        stats = {
            "kv_usage": self.kv_cache.usage(),
            "free_blocks": self.kv_cache.num_free_blocks,
            "running_requests": len(self.scheduler.running),
            "waiting_requests": len(self.scheduler.waiting),
            **self.kv_cache.memory_stats(),
        }
        if self.prefix_cache is not None:
            stats.update({f"prefix_{key}": value for key, value in self.prefix_cache.stats().items()})
        return stats
