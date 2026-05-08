from __future__ import annotations

from core.contiguous_kv_cache import ContiguousKVCache, ReservationPolicy
from core.paged_kv_cache import PagedKVCache


def test_oracle_reserves_exact_final_length() -> None:
    cache = ContiguousKVCache(
        total_token_slots=100,
        model_max_sequence_length=32,
        policy=ReservationPolicy.ORACLE,
    )
    assert cache.allocate("r0", prompt_len=5, max_output_len=20, actual_output_len=7)
    stats = cache.stats()
    assert stats["reserved_token_slots"] == 12
    assert stats["reservation_waste_tokens"] == 0


def test_pow2_policy_rounds_output_capacity() -> None:
    cache = ContiguousKVCache(
        total_token_slots=100,
        model_max_sequence_length=32,
        policy=ReservationPolicy.POW2,
    )
    assert cache.allocate("r0", prompt_len=5, max_output_len=9, actual_output_len=9)
    stats = cache.stats()
    assert stats["reserved_token_slots"] == 21
    assert stats["reservation_waste_tokens"] == 7


def test_max_policy_wastes_most_on_short_request() -> None:
    oracle = ContiguousKVCache(100, 32, ReservationPolicy.ORACLE)
    max_policy = ContiguousKVCache(100, 32, ReservationPolicy.MAX)
    oracle.allocate("r0", prompt_len=4, max_output_len=4, actual_output_len=4)
    max_policy.allocate("r0", prompt_len=4, max_output_len=4, actual_output_len=4)

    assert max_policy.stats()["reservation_waste_tokens"] > oracle.stats()["reservation_waste_tokens"]


def test_external_fragmentation_appears_after_middle_free() -> None:
    cache = ContiguousKVCache(30, 10, ReservationPolicy.ORACLE)
    cache.allocate("a", 5, 0, 0)
    cache.allocate("b", 5, 0, 0)
    cache.allocate("c", 5, 0, 0)
    cache.free_request("b")

    stats = cache.stats()

    assert stats["num_free_segments"] == 2
    assert stats["external_waste_tokens"] == 5


def test_paged_admits_more_variable_requests_under_same_budget() -> None:
    contiguous = ContiguousKVCache(16, 8, ReservationPolicy.MAX)
    assert contiguous.allocate("a", prompt_len=1, max_output_len=1, actual_output_len=1)
    assert contiguous.allocate("b", prompt_len=1, max_output_len=1, actual_output_len=1)
    assert not contiguous.allocate("c", prompt_len=1, max_output_len=1, actual_output_len=1)

    paged = PagedKVCache(block_size=2, num_blocks=8)
    for request_id in ["a", "b", "c"]:
        table = paged.attach_request(request_id)
        paged.append_tokens(request_id, table, [1, 2])

    assert paged.memory_stats()["num_used_blocks"] == 3
