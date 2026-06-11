from __future__ import annotations

from wllm_omni.sched.base_scheduler import BaseScheduler
from wllm_omni.sched.interface import RequestStatus, SchedulerOutput
from wllm_omni.worker.utils import RunnerBatchOutput, RunnerOutput


class RequestScheduler(BaseScheduler):

    def update_from_output(self, sched_output: SchedulerOutput, output: RunnerBatchOutput | RunnerOutput) -> set[str]:
        scheduled_req_ids = sched_output.scheduled_req_ids
        if not scheduled_req_ids:
            return set()

        output_by_req_id = self._outputs_by_req_id(output)
        terminal_statuses: dict[str, RequestStatus] = {}
        terminal_errors: dict[str, str | None] = {}
        for sched_req_id in scheduled_req_ids:
            state = self._request_states.get(sched_req_id)
            if state is None or state.is_finished():
                continue

            item = output_by_req_id.get(sched_req_id)
            if item is None:
                terminal_statuses[sched_req_id] = RequestStatus.FINISHED_ERROR
                terminal_errors[sched_req_id] = "Missing RunnerOutput for scheduled request"
                continue

            if item.error:
                terminal_statuses[sched_req_id] = RequestStatus.FINISHED_ERROR
                terminal_errors[sched_req_id] = item.error
            elif item.finished and item.result is None:
                terminal_statuses[sched_req_id] = RequestStatus.FINISHED_ERROR
                terminal_errors[sched_req_id] = "No output result"
            elif item.finished:
                terminal_statuses[sched_req_id] = RequestStatus.FINISHED_COMPLETED
                terminal_errors[sched_req_id] = None
            else:
                state.error = None

        return self._finalize_update_from_output(sched_output, terminal_statuses, terminal_errors)

    @staticmethod
    def _outputs_by_req_id(output: RunnerBatchOutput | RunnerOutput) -> dict[str, RunnerOutput]:
        if isinstance(output, RunnerBatchOutput):
            outputs = output.outputs
        else:
            outputs = [output]
        return {item.req_id: item for item in outputs}
