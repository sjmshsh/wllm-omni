from __future__ import annotations

from dataclasses import dataclass

from wllm_omni.request import OmniRequest
from wllm_omni.sched.base_scheduler import BaseScheduler
from wllm_omni.sched.interface import RequestStatus, SchedulerOutput, SchedulerRequestState
from wllm_omni.worker.utils import RunnerBatchOutput, RunnerOutput


@dataclass(slots=True)
class _StepProgress:
    current_step: int
    total_steps: int


class StepScheduler(BaseScheduler):

    def __init__(self, max_num_running_reqs: int = 1):
        super().__init__(max_num_running_reqs=max_num_running_reqs)
        self._request_progress: dict[str, _StepProgress] = {}

    def add_request(self, request: OmniRequest) -> str:
        sched_req_id = self._make_sched_req_id(request)
        total_steps = int(request.sampling_params.num_inference_steps)
        if total_steps <= 0:
            raise ValueError(f"Request {sched_req_id} must have positive num_inference_steps, got {total_steps}")
        self._request_states[sched_req_id] = SchedulerRequestState(
            sched_req_id=sched_req_id,
            req=request,
        )
        self._request_id_to_sched_req_id[request.request_id] = sched_req_id
        self._waiting.append(sched_req_id)
        self._request_progress[sched_req_id] = _StepProgress(current_step=0, total_steps=total_steps)
        return sched_req_id

    def update_from_output(self, sched_output: SchedulerOutput, output: RunnerBatchOutput | RunnerOutput) -> set[str]:
        scheduled_req_ids = sched_output.scheduled_req_ids
        if not scheduled_req_ids:
            return set()

        output_by_req_id = self._outputs_by_req_id(output)
        terminal_statuses: dict[str, RequestStatus] = {}
        terminal_errors: dict[str, str | None] = {}
        for sched_req_id in scheduled_req_ids:
            state = self._request_states.get(sched_req_id)
            progress = self._request_progress.get(sched_req_id)
            if state is None or progress is None or state.is_finished():
                continue

            item = output_by_req_id.get(sched_req_id)
            if item is None:
                terminal_statuses[sched_req_id] = RequestStatus.FINISHED_ERROR
                terminal_errors[sched_req_id] = "Missing RunnerOutput for scheduled request"
                continue

            if item.error is not None:
                terminal_statuses[sched_req_id] = RequestStatus.FINISHED_ERROR
                terminal_errors[sched_req_id] = item.error
                continue

            if item.step_index is None:
                terminal_statuses[sched_req_id] = RequestStatus.FINISHED_ERROR
                terminal_errors[sched_req_id] = "Missing step_index in RunnerOutput"
                continue

            progress.current_step = item.step_index
            if item.finished:
                terminal_statuses[sched_req_id] = RequestStatus.FINISHED_COMPLETED
                terminal_errors[sched_req_id] = None
            else:
                state.error = None

        return self._finalize_update_from_output(sched_output, terminal_statuses, terminal_errors)

    def pop_request_state(self, sched_req_id: str):
        self._request_progress.pop(sched_req_id, None)
        return super().pop_request_state(sched_req_id)

    def close(self) -> None:
        self._request_progress.clear()
        super().close()

    @staticmethod
    def _outputs_by_req_id(output: RunnerBatchOutput | RunnerOutput) -> dict[str, RunnerOutput]:
        if isinstance(output, RunnerBatchOutput):
            outputs = output.outputs
        else:
            outputs = [output]
        return {item.req_id: item for item in outputs}
