from __future__ import annotations

from core.request import RequestState
from core.sampling_params import SamplingParams
from core.scheduler import Scheduler


def _request(index: int) -> RequestState:
    return RequestState(
        request_id=f"req-{index}",
        prompt=f"prompt {index}",
        prompt_token_ids=[index],
        sampling_params=SamplingParams(max_tokens=2),
    )


def test_waiting_requests_are_admitted_in_order() -> None:
    scheduler = Scheduler(max_batch_size=2)
    requests = [_request(i) for i in range(3)]
    for request in requests:
        scheduler.add_request(request)

    admitted = scheduler.admit_requests()

    assert [request.request_id for request in admitted] == ["req-0", "req-1"]
    assert [request.request_id for request in scheduler.active_requests()] == ["req-0", "req-1"]
    assert [request.request_id for request in scheduler.waiting] == ["req-2"]


def test_running_count_never_exceeds_max_batch_size() -> None:
    scheduler = Scheduler(max_batch_size=1)
    scheduler.add_request(_request(0))
    scheduler.add_request(_request(1))

    scheduler.admit_requests()
    scheduler.admit_requests()

    assert len(scheduler.running) == 1
    assert len(scheduler.waiting) == 1


def test_finish_request_releases_running_slot() -> None:
    scheduler = Scheduler(max_batch_size=1)
    scheduler.add_request(_request(0))
    scheduler.add_request(_request(1))
    scheduler.admit_requests()

    finished = scheduler.finish_request("req-0")
    admitted = scheduler.admit_requests()

    assert finished is not None
    assert [request.request_id for request in admitted] == ["req-1"]
    assert not scheduler.waiting
