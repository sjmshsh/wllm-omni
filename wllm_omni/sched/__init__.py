from __future__ import annotations

from wllm_omni.sched.interface import (
    CachedRequestData,
    NewRequestData,
    RequestStatus,
    ScheduledRequest,
    SchedulerOutput,
    SchedulerRequestState,
)

__all__ = [
    "CachedRequestData",
    "NewRequestData",
    "RequestScheduler",
    "RequestStatus",
    "ScheduledRequest",
    "SchedulerOutput",
    "SchedulerRequestState",
    "StepScheduler",
]


def __getattr__(name: str):
    if name == "RequestScheduler":
        from wllm_omni.sched.request_scheduler import RequestScheduler

        return RequestScheduler
    if name == "StepScheduler":
        from wllm_omni.sched.step_scheduler import StepScheduler

        return StepScheduler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
