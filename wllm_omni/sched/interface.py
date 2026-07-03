from __future__ import annotations

import enum
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wllm_omni.request import OmniRequest


class RequestStatus(enum.IntEnum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    FINISHED_COMPLETED = enum.auto()
    FINISHED_ABORTED = enum.auto()
    FINISHED_ERROR = enum.auto()

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status >= RequestStatus.FINISHED_COMPLETED


@dataclass(slots=True)
class SchedulerRequestState:
    sched_req_id: str
    req: OmniRequest
    status: RequestStatus = RequestStatus.WAITING
    error: str | None = None

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)


@dataclass(slots=True)
class ScheduledRequest:
    sched_req_id: str
    req: OmniRequest | None = None
    is_new: bool = False

    @classmethod
    def from_state(cls, state: SchedulerRequestState, is_new: bool) -> "ScheduledRequest":
        return cls(
            sched_req_id=state.sched_req_id,
            req=state.req if is_new else None,
            is_new=is_new,
        )


@dataclass(slots=True)
class NewRequestData:
    sched_req_id: str
    req: OmniRequest

    @classmethod
    def from_state(cls, state: SchedulerRequestState) -> "NewRequestData":
        return cls(sched_req_id=state.sched_req_id, req=state.req)


@dataclass(slots=True)
class CachedRequestData:
    sched_req_ids: list[str]

    @classmethod
    def make_empty(cls) -> "CachedRequestData":
        return cls(sched_req_ids=[])


@dataclass(slots=True)
class SchedulerOutput:
    step_id: int
    scheduled_reqs: list[ScheduledRequest]
    finished_req_ids: set[str]
    num_running_reqs: int
    num_waiting_reqs: int

    @property
    def scheduled_req_ids(self) -> list[str]:
        return [req.sched_req_id for req in self.scheduled_reqs]

    @property
    def scheduled_entries(self) -> list[ScheduledRequest]:
        return self.scheduled_reqs

    @property
    def scheduled_new_reqs(self) -> list[NewRequestData]:
        return [
            NewRequestData(sched_req_id=req.sched_req_id, req=req.req)
            for req in self.scheduled_reqs
            if req.is_new and req.req is not None
        ]

    @property
    def scheduled_cached_reqs(self) -> CachedRequestData:
        return CachedRequestData(
            sched_req_ids=[req.sched_req_id for req in self.scheduled_reqs if not req.is_new]
        )

    @property
    def is_empty(self) -> bool:
        return len(self.scheduled_reqs) == 0

    @property
    def num_scheduled_reqs(self) -> int:
        return len(self.scheduled_reqs)



class SchedulerInterface(ABC):

    def _make_sched_req_id(self, request: OmniRequest) -> str:
        base = request.request_id or f"req_{uuid.uuid4().hex[:8]}"
        sched_req_id = base
        suffix = 1
        while self.get_request_state(sched_req_id) is not None:
            sched_req_id = f"{base}#{suffix}"
            suffix += 1
        return sched_req_id

    @abstractmethod
    def add_request(self, request: OmniRequest) -> str:
        pass

    @abstractmethod
    def schedule(self) -> SchedulerOutput:
        pass

    @abstractmethod
    def update_from_output(self, sched_output: SchedulerOutput, output) -> set[str]:
        pass

    @abstractmethod
    def get_request_state(self, sched_req_id: str) -> SchedulerRequestState | None:
        pass

    @abstractmethod
    def has_requests(self) -> bool:
        pass

    @abstractmethod
    def get_sched_req_id(self, request_id: str) -> str | None:
        pass

    @abstractmethod
    def pop_request_state(self, sched_req_id: str) -> SchedulerRequestState | None:
        pass

    @abstractmethod
    def preempt_request(self, sched_req_id: str) -> bool:
        pass

    @abstractmethod
    def finish_requests(self, sched_req_ids: str | list[str], status: RequestStatus) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
