from __future__ import annotations

from collections import defaultdict

import torch

from wllm_omni.config import EngineConfig
from wllm_omni.model_types import ModelParadigm
from wllm_omni.models import ModelExecutor
from wllm_omni.models.diffusion_executor import DiffusionExecutor
from wllm_omni.models.wan22 import Wan22I2VPipeline
from wllm_omni.sched.interface import SchedulerOutput
from wllm_omni.worker.utils import RequestState, RunnerBatchOutput, RunnerOutput


class ModelRunner:
    """Generic runner that orchestrates executor-driven model forward.

    The runner owns request lifecycle and batching orchestration. Concrete
    executors own model-family details such as diffusion latents, AR KV cache,
    multimodal feature caches, or world-model rollout state.
    """

    def __init__(self, config: EngineConfig, executors: list[ModelExecutor] | None = None):
        self.config = config
        if executors is None:
            executors = [DiffusionExecutor(Wan22I2VPipeline(config))]
        if not executors:
            raise ValueError("ModelRunner requires at least one executor.")

        self.executors = {executor.paradigm: executor for executor in executors}
        self.default_executor = executors[0]
        self.state_cache: dict[str, RequestState] = {}

    def execute(self, scheduler_output: SchedulerOutput) -> RunnerBatchOutput:
        """Execute the scheduled requests and return per-request outputs.

        This is the V1 equivalent of SGLang's ScheduleBatch -> ForwardBatch
        boundary. The scheduler output stays CPU/request-level; each executor
        builds the family-specific ForwardBatch consumed by model forward.
        """

        outputs: list[RunnerOutput] = []
        with torch.no_grad():
            self._release_finished_scheduler_states(scheduler_output.finished_req_ids)
            try:
                states = self._prepare_scheduled_states(scheduler_output)
            except Exception as exc:
                return RunnerBatchOutput(outputs=self._scheduler_error_outputs(scheduler_output, str(exc)))
            if not states:
                return RunnerBatchOutput(outputs=[])

            for group_states in self._group_states(states):
                executor = self._executor_for_state(group_states[0])
                try:
                    forward_batch = executor.build_forward_batch(group_states)
                    model_output = executor.forward(forward_batch)
                    executor.update_states(group_states, model_output)
                    group_outputs = executor.collect_outputs(group_states, model_output)
                    outputs.extend(group_outputs)
                except Exception as exc:
                    outputs.extend(self._mark_group_error(group_states, str(exc)))

        self._release_finished_outputs(outputs)
        return RunnerBatchOutput(outputs=outputs)

    def execute_stepwise(self, scheduler_output: SchedulerOutput) -> RunnerOutput:
        """Compatibility shim for the current single-request scheduler."""

        batch_output = self.execute(scheduler_output)
        if scheduler_output.num_scheduled_reqs != 1:
            if batch_output.outputs:
                return batch_output.outputs[0]
            return RunnerOutput(
                req_id="unknown",
                finished=True,
                error=(
                    "wllm-omni step execution currently supports exactly one scheduled request, "
                    f"got {scheduler_output.num_scheduled_reqs}."
                ),
            )
        return batch_output.to_single()

    def _prepare_scheduled_states(self, scheduler_output: SchedulerOutput) -> list[RequestState]:
        for entry in scheduler_output.scheduled_entries:
            if not entry.is_new:
                continue
            if entry.req is None:
                raise ValueError(f"Scheduled new request {entry.sched_req_id} is missing request payload")
            executor = self._executor_for_request(entry.req)
            state = executor.init_state(entry.sched_req_id, entry.req)
            self.state_cache[entry.sched_req_id] = state

        states: list[RequestState] = []
        for sched_req_id in scheduler_output.scheduled_req_ids:
            state = self.state_cache.get(sched_req_id)
            if state is None:
                raise ValueError(f"Missing cached state for sched_req_id={sched_req_id}")
            states.append(state)
        return states

    def _group_states(self, states: list[RequestState]) -> list[list[RequestState]]:
        grouped: dict[tuple, list[RequestState]] = defaultdict(list)
        for state in states:
            executor = self._executor_for_state(state)
            grouped[(state.paradigm, executor.batch_key(state))].append(state)
        return list(grouped.values())

    def _executor_for_request(self, request) -> ModelExecutor:
        paradigm = getattr(request, "model_paradigm", None)
        if paradigm is None:
            return self.default_executor
        if isinstance(paradigm, str):
            paradigm = ModelParadigm(paradigm)
        executor = self.executors.get(paradigm)
        if executor is None:
            raise ValueError(f"No executor registered for request paradigm={paradigm}.")
        return executor

    def _executor_for_state(self, state: RequestState) -> ModelExecutor:
        executor = self.executors.get(state.paradigm)
        if executor is None:
            raise ValueError(f"No executor registered for paradigm={state.paradigm}.")
        return executor

    def _release_finished_scheduler_states(self, finished_req_ids: set[str]) -> None:
        for sched_req_id in finished_req_ids:
            state = self.state_cache.pop(sched_req_id, None)
            if state is not None:
                self._executor_for_state(state).release(state)

    def _release_finished_outputs(self, outputs: list[RunnerOutput]) -> None:
        for output in outputs:
            if not output.finished and output.error is None:
                continue
            state = self.state_cache.pop(output.req_id, None)
            if state is not None:
                self._executor_for_state(state).release(state)

    def _mark_group_error(self, states: list[RequestState], error: str) -> list[RunnerOutput]:
        outputs: list[RunnerOutput] = []
        for state in states:
            state.error = error
            state.finished = True
            outputs.append(RunnerOutput(req_id=state.sched_req_id, finished=True, error=error))
        return outputs

    def _scheduler_error_outputs(
        self,
        scheduler_output: SchedulerOutput,
        error: str,
    ) -> list[RunnerOutput]:
        req_ids = scheduler_output.scheduled_req_ids or ["unknown"]
        outputs: list[RunnerOutput] = []
        for req_id in req_ids:
            state = self.state_cache.pop(req_id, None)
            if state is not None:
                state.error = error
                state.finished = True
                self._executor_for_state(state).release(state)
            outputs.append(RunnerOutput(req_id=req_id, finished=True, error=error))
        return outputs
