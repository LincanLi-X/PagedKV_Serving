from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class ReservationPolicy(StrEnum):
    ORACLE = "orca_oracle"
    POW2 = "orca_pow2"
    MAX = "orca_max"
    FASTERTRANSFORMER_LIKE = "fastertransformer_like"


@dataclass(slots=True)
class ContiguousAllocation:
    request_id: str
    start: int
    reserved_slots: int
    prompt_len: int
    max_output_len: int
    actual_output_len: int
    realized_output_len: int = 0

    @property
    def useful_tokens(self) -> int:
        return self.prompt_len + self.realized_output_len

    @property
    def final_useful_tokens(self) -> int:
        return self.prompt_len + self.actual_output_len

    @property
    def reservation_waste_tokens(self) -> int:
        return max(0, self.reserved_slots - self.final_useful_tokens)

    @property
    def internal_waste_tokens(self) -> int:
        return max(0, self.reserved_slots - self.useful_tokens)


class ContiguousKVCache:
    def __init__(
        self,
        total_token_slots: int,
        model_max_sequence_length: int,
        policy: ReservationPolicy | str = ReservationPolicy.ORACLE,
    ) -> None:
        if total_token_slots <= 0:
            raise ValueError("total_token_slots must be positive")
        if model_max_sequence_length <= 0:
            raise ValueError("model_max_sequence_length must be positive")
        self.total_token_slots = total_token_slots
        self.model_max_sequence_length = model_max_sequence_length
        self.policy = ReservationPolicy(policy)
        self.allocations: dict[str, ContiguousAllocation] = {}
        self.rejected_allocations = 0
        self.admitted_allocations = 0

    def _reservation_size(
        self,
        prompt_len: int,
        max_output_len: int,
        actual_output_len: int | None,
    ) -> int:
        if prompt_len < 0 or max_output_len < 0:
            raise ValueError("lengths must be non-negative")
        actual = max_output_len if actual_output_len is None else actual_output_len
        if self.policy == ReservationPolicy.ORACLE:
            return prompt_len + actual
        if self.policy == ReservationPolicy.POW2:
            rounded = 1 if max_output_len <= 1 else 2 ** math.ceil(math.log2(max_output_len))
            return prompt_len + rounded
        if self.policy == ReservationPolicy.MAX:
            return self.model_max_sequence_length
        if self.policy == ReservationPolicy.FASTERTRANSFORMER_LIKE:
            return min(
                self.model_max_sequence_length,
                max(self.model_max_sequence_length, prompt_len + max_output_len),
            )
        raise ValueError(f"unsupported policy {self.policy}")

    def _sorted_allocations(self) -> list[ContiguousAllocation]:
        return sorted(self.allocations.values(), key=lambda allocation: allocation.start)

    def free_segments(self) -> list[tuple[int, int]]:
        segments: list[tuple[int, int]] = []
        cursor = 0
        for allocation in self._sorted_allocations():
            if allocation.start > cursor:
                segments.append((cursor, allocation.start - cursor))
            cursor = max(cursor, allocation.start + allocation.reserved_slots)
        if cursor < self.total_token_slots:
            segments.append((cursor, self.total_token_slots - cursor))
        return segments

    def allocate(
        self,
        request_id: str,
        prompt_len: int,
        max_output_len: int,
        actual_output_len: int | None = None,
    ) -> bool:
        if request_id in self.allocations:
            raise ValueError(f"request {request_id} already allocated")
        actual = max_output_len if actual_output_len is None else actual_output_len
        reserved = self._reservation_size(prompt_len, max_output_len, actual)
        for start, size in self.free_segments():
            if size >= reserved:
                self.allocations[request_id] = ContiguousAllocation(
                    request_id=request_id,
                    start=start,
                    reserved_slots=reserved,
                    prompt_len=prompt_len,
                    max_output_len=max_output_len,
                    actual_output_len=actual,
                )
                self.admitted_allocations += 1
                return True
        self.rejected_allocations += 1
        return False

    def append_token(self, request_id: str, count: int = 1) -> None:
        allocation = self.allocations[request_id]
        if allocation.useful_tokens + count > allocation.reserved_slots:
            raise RuntimeError(f"request {request_id} exceeded its contiguous reservation")
        allocation.realized_output_len += count

    def free_request(self, request_id: str) -> None:
        self.allocations.pop(request_id, None)

    def stats(self) -> dict[str, int | float]:
        reserved = sum(allocation.reserved_slots for allocation in self.allocations.values())
        useful = sum(allocation.useful_tokens for allocation in self.allocations.values())
        final_useful = sum(allocation.final_useful_tokens for allocation in self.allocations.values())
        reservation_waste = sum(
            allocation.reservation_waste_tokens for allocation in self.allocations.values()
        )
        internal_waste = sum(
            allocation.internal_waste_tokens for allocation in self.allocations.values()
        )
        free_segments = self.free_segments()
        total_free = sum(size for _, size in free_segments)
        largest_free = max((size for _, size in free_segments), default=0)
        external_waste = max(0, total_free - largest_free)
        return {
            "policy": self.policy.value,
            "total_token_slots": self.total_token_slots,
            "active_requests": len(self.allocations),
            "admitted_allocations": self.admitted_allocations,
            "rejected_allocations": self.rejected_allocations,
            "useful_tokens": useful,
            "final_useful_tokens": final_useful,
            "reserved_token_slots": reserved,
            "reservation_waste_tokens": reservation_waste,
            "internal_waste_tokens": internal_waste,
            "external_waste_tokens": external_waste,
            "num_free_segments": len(free_segments),
            "largest_free_segment": largest_free,
            "useful_capacity_ratio": useful / reserved if reserved else 1.0,
            "internal_fragmentation_ratio": internal_waste / reserved if reserved else 0.0,
            "external_fragmentation_ratio": external_waste / self.total_token_slots,
        }


def make_contiguous_cache(
    policy: ReservationPolicy | str,
    total_token_slots: int = 4096,
    model_max_sequence_length: int = 1024,
) -> ContiguousKVCache:
    return ContiguousKVCache(
        total_token_slots=total_token_slots,
        model_max_sequence_length=model_max_sequence_length,
        policy=policy,
    )
