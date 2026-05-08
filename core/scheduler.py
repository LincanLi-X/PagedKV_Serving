from __future__ import annotations

from collections import OrderedDict, deque

from core.request import RequestState


class Scheduler:
    """continuous batching scheduler with waiting and running queues."""

    def __init__(self, max_batch_size: int = 4) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")
        self.max_batch_size = max_batch_size
        self.waiting: deque[RequestState] = deque()
        self.running: OrderedDict[str, RequestState] = OrderedDict()

    def add_request(self, request: RequestState) -> None:
        if request.request_id in self.running:
            raise ValueError(f"request {request.request_id} is already running")
        if any(existing.request_id == request.request_id for existing in self.waiting):
            raise ValueError(f"request {request.request_id} is already waiting")
        request.status = "waiting"
        self.waiting.append(request)

    def admit_requests(self) -> list[RequestState]:
        admitted: list[RequestState] = []
        while self.waiting and len(self.running) < self.max_batch_size:
            request = self.waiting.popleft()
            request.mark_running()
            self.running[request.request_id] = request
            admitted.append(request)
        return admitted

    def active_requests(self) -> list[RequestState]:
        return list(self.running.values())

    def finish_request(self, request_id: str) -> RequestState | None:
        return self.running.pop(request_id, None)

    def is_idle(self) -> bool:
        return not self.waiting and not self.running

    def __len__(self) -> int:
        return len(self.waiting) + len(self.running)
