from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable


def _clone_tensor_like(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().clone()
    if isinstance(value, list):
        return list(value)
    return value


def _concat_tensor_along_seq(left: Any, right: Any) -> Any:
    if left is None:
        return _clone_tensor_like(right)
    if right is None:
        return _clone_tensor_like(left)
    if hasattr(left, "ndim"):
        import torch

        dim = -2 if left.ndim >= 3 else 0
        return torch.cat([left, right], dim=dim)
    if isinstance(left, list):
        return list(left) + list(right)
    return left + right


def concat_past_segments(segments: list[Any]) -> Any | None:
    if not segments:
        return None
    merged = segments[0]
    for segment in segments[1:]:
        merged = tuple(
            (
                _concat_tensor_along_seq(layer_left[0], layer_right[0]),
                _concat_tensor_along_seq(layer_left[1], layer_right[1]),
            )
            for layer_left, layer_right in zip(merged, segment)
        )
    return merged


@dataclass(slots=True)
class BlockAppendPlan:
    block_id: int
    input_token_start: int
    input_token_end: int
    request_position_start: int
    request_position_end: int


@dataclass(slots=True)
class KVBlock:
    block_id: int
    capacity: int
    token_ids: list[int] = field(default_factory=list)
    kv_segment: Any | None = None
    ref_count: int = 0
    cache_key: str | None = None

    @property
    def is_full(self) -> bool:
        return len(self.token_ids) >= self.capacity

    @property
    def free_slots(self) -> int:
        return self.capacity - len(self.token_ids)


class PagedKVCache:
    """Fixed-size physical block pool with per-request logical block tables."""

    def __init__(self, block_size: int, num_blocks: int) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.block_size = block_size
        self.blocks: dict[int, KVBlock] = {
            idx: KVBlock(idx, capacity=block_size) for idx in range(num_blocks)
        }
        self.free_block_order: OrderedDict[int, None] = OrderedDict(
            (idx, None) for idx in range(num_blocks)
        )
        self.request_blocks: dict[str, list[int]] = {}
        self.cow_count = 0
        self.copied_block_count = 0
        self._eviction_callback: Callable[[int, str], None] | None = None

    @property
    def num_total_blocks(self) -> int:
        return len(self.blocks)

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_order)

    @property
    def num_used_blocks(self) -> int:
        return self.num_total_blocks - self.num_free_blocks

    def register_eviction_callback(self, callback: Callable[[int, str], None]) -> None:
        self._eviction_callback = callback

    def attach_request(self, request_id: str, block_table: list[int] | None = None) -> list[int]:
        table = block_table if block_table is not None else []
        self.request_blocks[request_id] = table
        return table

    def usage(self) -> float:
        return self.num_used_blocks / self.num_total_blocks

    def _remove_from_free_order(self, block_id: int) -> None:
        self.free_block_order.pop(block_id, None)

    def _push_free_tail(self, block_id: int) -> None:
        self.free_block_order.pop(block_id, None)
        self.free_block_order[block_id] = None

    def _evict_if_cached(self, block: KVBlock) -> None:
        if block.cache_key is not None and self._eviction_callback is not None:
            self._eviction_callback(block.block_id, block.cache_key)
        block.cache_key = None

    def _allocate_empty_block(self) -> KVBlock:
        if not self.free_block_order:
            raise RuntimeError("No free KV cache blocks available")
        block_id, _ = self.free_block_order.popitem(last=False)
        block = self.blocks[block_id]
        self._evict_if_cached(block)
        block.token_ids.clear()
        block.kv_segment = None
        block.ref_count = 0
        return block

    def _copy_private_block(self, shared_block: KVBlock) -> KVBlock:
        private = self._allocate_empty_block()
        private.token_ids = list(shared_block.token_ids)
        private.kv_segment = _clone_past(shared_block.kv_segment)
        private.ref_count = 1
        shared_block.ref_count -= 1
        self.cow_count += 1
        self.copied_block_count += 1
        return private

    def acquire_cached_blocks(self, request_id: str, block_ids: list[int]) -> None:
        block_table = self.request_blocks.setdefault(request_id, [])
        for block_id in block_ids:
            block = self.blocks[block_id]
            if len(block.token_ids) != self.block_size:
                raise ValueError("Only full blocks can be shared from prefix cache")
            if block.ref_count == 0:
                self._remove_from_free_order(block_id)
            block.ref_count += 1
            block_table.append(block_id)

    def append_tokens(
        self,
        request_id: str,
        block_table: list[int],
        token_ids: list[int],
    ) -> list[BlockAppendPlan]:
        if request_id not in self.request_blocks:
            self.request_blocks[request_id] = block_table
        if not token_ids:
            return []

        plans: list[BlockAppendPlan] = []
        request_length = self.request_token_count(request_id)
        input_start = 0
        while input_start < len(token_ids):
            tail_block = self.blocks[block_table[-1]] if block_table else None
            if tail_block is not None and tail_block.ref_count > 1 and tail_block.free_slots > 0:
                private = self._copy_private_block(tail_block)
                block_table[-1] = private.block_id
                tail_block = private
            if tail_block is None or tail_block.is_full:
                tail_block = self._allocate_empty_block()
                tail_block.ref_count = 1
                block_table.append(tail_block.block_id)

            free_slots = tail_block.free_slots
            chunk = token_ids[input_start : input_start + free_slots]
            tail_block.token_ids.extend(chunk)
            input_end = input_start + len(chunk)
            plans.append(
                BlockAppendPlan(
                    block_id=tail_block.block_id,
                    input_token_start=input_start,
                    input_token_end=input_end,
                    request_position_start=request_length,
                    request_position_end=request_length + len(chunk),
                )
            )
            request_length += len(chunk)
            input_start = input_end
        return plans

    def append_kv_to_block(self, block_id: int, kv_slice: Any) -> None:
        block = self.blocks[block_id]
        if block.kv_segment is None:
            block.kv_segment = _clone_past(kv_slice)
            return
        block.kv_segment = tuple(
            (
                _concat_tensor_along_seq(layer_existing[0], layer_new[0]),
                _concat_tensor_along_seq(layer_existing[1], layer_new[1]),
            )
            for layer_existing, layer_new in zip(block.kv_segment, kv_slice)
        )

    def set_cache_key(self, block_id: int, cache_key: str | None) -> None:
        self.blocks[block_id].cache_key = cache_key

    def request_token_count(self, request_id: str) -> int:
        block_table = self.request_blocks.get(request_id, [])
        return sum(len(self.blocks[block_id].token_ids) for block_id in block_table)

    def flatten_block_tokens(self, block_ids: list[int]) -> list[int]:
        flattened: list[int] = []
        for block_id in block_ids:
            flattened.extend(self.blocks[block_id].token_ids)
        return flattened

    def gather_past_key_values(self, block_ids: list[int]) -> Any | None:
        segments = []
        for block_id in block_ids:
            segment = self.blocks[block_id].kv_segment
            if segment is None:
                raise ValueError(f"Block {block_id} does not have stored KV tensors")
            segments.append(segment)
        return concat_past_segments(segments)

    def block_is_full(self, block_id: int) -> bool:
        return len(self.blocks[block_id].token_ids) == self.block_size

    def block_tokens(self, block_id: int) -> list[int]:
        return list(self.blocks[block_id].token_ids)

    def free_request(self, request_id: str) -> None:
        block_table = self.request_blocks.pop(request_id, [])
        for block_id in reversed(block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count < 0:
                raise RuntimeError(f"Negative ref_count for block {block_id}")
            if block.ref_count == 0:
                self._push_free_tail(block_id)

    def memory_stats(self) -> dict[str, int | float]:
        useful_tokens = sum(
            len(block.token_ids) for block in self.blocks.values() if block.ref_count > 0
        )
        reserved_token_slots = self.num_used_blocks * self.block_size
        internal_waste_tokens = sum(
            block.capacity - len(block.token_ids)
            for block in self.blocks.values()
            if block.ref_count > 0 and len(block.token_ids) < block.capacity
        )
        external_waste_tokens = 0
        useful_capacity_ratio = (
            useful_tokens / reserved_token_slots if reserved_token_slots else 1.0
        )
        return {
            "num_total_blocks": self.num_total_blocks,
            "num_free_blocks": self.num_free_blocks,
            "num_used_blocks": self.num_used_blocks,
            "useful_tokens": useful_tokens,
            "reserved_token_slots": reserved_token_slots,
            "internal_waste_tokens": internal_waste_tokens,
            "external_waste_tokens": external_waste_tokens,
            "useful_capacity_ratio": useful_capacity_ratio,
            "internal_fragmentation_ratio": (
                internal_waste_tokens / reserved_token_slots if reserved_token_slots else 0.0
            ),
            "external_fragmentation_ratio": 0.0,
            "cow_count": self.cow_count,
            "copied_block_count": self.copied_block_count,
        }


def _clone_past(past: Any) -> Any:
    if past is None:
        return None
    return tuple(
        (_clone_tensor_like(layer[0]), _clone_tensor_like(layer[1]))
        for layer in past
    )
