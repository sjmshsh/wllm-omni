import torch

from wllm_omni.config import EngineConfig
from wllm_omni.models import supports_step_execution
from wllm_omni.models.wan22 import Wan22I2VPipeline
from wllm_omni.sched.interface import DiffusionSchedulerOutput
from wllm_omni.worker.utils import RunnerOutput, RunnerState


class ModelRunner:

    def __init__(self, config: EngineConfig):
        self.config = config
        self.pipeline = Wan22I2VPipeline(config)
        self.state_cache: dict[str, RunnerState] = {}
        if not supports_step_execution(self.pipeline):
            raise TypeError(f"{self.pipeline.__class__.__name__} does not implement the step execution contract.")

    def _get_or_create_state(self, scheduler_output: DiffusionSchedulerOutput) -> tuple[str, RunnerState, bool]:
        if scheduler_output.num_scheduled_reqs != 1:
            raise ValueError(
                "wllm-omni step execution currently supports exactly one scheduled request, "
                f"got {scheduler_output.num_scheduled_reqs}."
            )

        if scheduler_output.scheduled_new_reqs:
            new_req = scheduler_output.scheduled_new_reqs[0]
            request = new_req.req
            state = RunnerState(
                req_id=request.request_id,
                sampling=request.sampling_params,
                prompt=request.prompt,
                image=request.image,
                negative_prompt=request.sampling_params.negative_prompt,
            )
            self.state_cache[new_req.sched_req_id] = state
            return new_req.sched_req_id, state, True

        sched_req_id = scheduler_output.scheduled_cached_reqs.sched_req_ids[0]
        state = self.state_cache.get(sched_req_id)
        if state is None:
            raise ValueError(f"Missing cached state for sched_req_id={sched_req_id}")
        return sched_req_id, state, False

    def execute_stepwise(self, scheduler_output: DiffusionSchedulerOutput) -> RunnerOutput:
        sched_req_id = "unknown"
        try:
            with torch.inference_mode():
                for finished_req_id in scheduler_output.finished_req_ids:
                    self.state_cache.pop(finished_req_id, None)

                sched_req_id, state, is_new_request = self._get_or_create_state(scheduler_output)
                if is_new_request:
                    state = self.pipeline.prepare_encode(state)
                    self.state_cache[sched_req_id] = state

                if not state.denoise_completed:
                    noise_pred = self.pipeline.denoise_step(state)
                    self.pipeline.step_scheduler(state, noise_pred)

                if state.denoise_completed:
                    result = self.pipeline.post_decode(state)
                    self.state_cache.pop(sched_req_id, None)
                    return RunnerOutput(
                        req_id=sched_req_id,
                        step_index=state.step_index,
                        finished=True,
                        result=result,
                    )

                return RunnerOutput(req_id=sched_req_id, step_index=state.step_index, finished=False, result=None)
        except Exception as exc:
            self.state_cache.pop(sched_req_id, None)
            return RunnerOutput(req_id=sched_req_id, finished=True, result=None, error=str(exc))
