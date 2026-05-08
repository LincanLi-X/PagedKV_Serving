from __future__ import annotations

from core.contiguous_kv_cache import ReservationPolicy, make_contiguous_cache


def build_cache(total_token_slots: int = 4096, model_max_sequence_length: int = 1024):
    return make_contiguous_cache(
        ReservationPolicy.MAX,
        total_token_slots=total_token_slots,
        model_max_sequence_length=model_max_sequence_length,
    )
