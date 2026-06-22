from __future__ import annotations

from contextlib import nullcontext

import torch

from wllm_omni.model_types import ModelParadigm
from wllm_omni.models import ModelExecutor, supports_step_execution
from wllm_omni.models.wan22 import Wan22I2VPipeline
from wllm_omni.profiler import RequestProfiler
from wllm_omni.request import OmniRequest
from wllm_omni.worker.utils import (
    ExecutionPhase,
    ExecutorCapability,
    ForwardBatch,
    ModelForwardOutput,
    RequestState,
    RunnerOutput,
    RunnerState,
)


class DiffusionExecutor(ModelExecutor):
    """Step-wise diffusion executor used by the generic ModelRunner V1.

    The executor owns diffusion-specific state and model calls. The generic
    runner only sees RequestState and ForwardBatch.
    """

    paradigm = ModelParadigm.DIFFUSION
    capabilities = frozenset({
        ExecutorCapability.STEPWISE,
        ExecutorCapability.CACHEABLE_PREPARE,
        ExecutorCapability.MULTIMODAL_INPUT,
    })

    def __init__(self, pipeline: Wan22I2VPipeline):
        self.pipeline = pipeline
        if not supports_step_execution(self.pipeline):
            raise TypeError(f"{self.pipeline.__class__.__name__} does not implement the step execution contract.")

    def init_state(self, sched_req_id: str, request: OmniRequest) -> RequestState:
        payload = RunnerState(
            req_id=request.request_id,
            sampling=request.sampling_params,
            prompt=request.prompt,
            image=request.image,
            negative_prompt=request.sampling_params.negative_prompt,
        )
        if self.pipeline.config.enable_profiling:
            payload.extra["profiler"] = RequestProfiler(request.request_id)
        return RequestState(
            req_id=request.request_id,
            sched_req_id=sched_req_id,
            paradigm=self.paradigm,
            payload=payload,
        )

    def batch_key(self, state: RequestState) -> tuple:
        payload = self._payload(state)
        sampling = payload.sampling
        return (
            self.paradigm.value,
            sampling.height,
            sampling.width,
            sampling.num_frames,
            sampling.num_inference_steps,
            sampling.guidance_scale,
            sampling.flow_shift,
            sampling.negative_prompt,
            sampling.fps,
            payload.step_index,
            state.sched_req_id,
        )

    def build_forward_batch(self, states: list[RequestState]) -> ForwardBatch:
        if len(states) != 1:
            raise ValueError(f"DiffusionExecutor V1 supports exactly one request per forward batch, got {len(states)}.")
        state = states[0]
        payload = self._payload(state)
        phase = ExecutionPhase.PREPARE if not state.initialized else ExecutionPhase.STEP
        if payload.denoise_completed:
            phase = ExecutionPhase.FINALIZE
        return ForwardBatch(paradigm=self.paradigm, req_ids=[state.sched_req_id], phase=phase, payload=payload)

    def forward(self, batch: ForwardBatch) -> ModelForwardOutput:
        if batch.paradigm != self.paradigm:
            raise ValueError(f"DiffusionExecutor cannot run batch for paradigm={batch.paradigm}.")
        if len(batch.req_ids) != 1:
            raise ValueError(f"DiffusionExecutor V1 supports exactly one request per forward batch, got {len(batch.req_ids)}.")

        payload = self._batch_payload(batch)
        profile = self._profiler(payload)
        output: ModelForwardOutput
        with self._profile_stage(payload, "forward.total"):
            if batch.phase == ExecutionPhase.PREPARE:
                with self._profile_stage(payload, "forward.prepare_encode"):
                    payload = self.pipeline.prepare_encode(payload)

            if not payload.denoise_completed:
                with self._profile_stage(payload, "forward.denoise_step"):
                    noise_pred = self.pipeline.denoise_step(payload)
                with self._profile_stage(payload, "forward.step_scheduler"):
                    self.pipeline.step_scheduler(payload, noise_pred)

            req_id = batch.req_ids[0]
            if payload.denoise_completed:
                with self._profile_stage(payload, "forward.post_decode"):
                    result = self.pipeline.post_decode(payload)
                output = ModelForwardOutput(
                    outputs=[
                        RunnerOutput(
                            req_id=req_id,
                            step_index=payload.step_index,
                            finished=True,
                            result=result,
                        )
                    ],
                    payload=payload,
                )
            else:
                output = ModelForwardOutput(
                    outputs=[RunnerOutput(req_id=req_id, step_index=payload.step_index, finished=False)],
                    payload=payload,
                )

        if profile is not None and payload.denoise_completed:
            self._emit_profile(payload)
        return output

    def update_states(self, states: list[RequestState], output: ModelForwardOutput) -> None:
        output_by_req_id = {item.req_id: item for item in output.outputs}
        for state in states:
            item = output_by_req_id.get(state.sched_req_id)
            if item is None:
                continue
            if output.payload is not None:
                state.payload = output.payload
                state.initialized = True
            if item.error is not None:
                state.error = item.error
                state.finished = True
            if item.step_index is not None:
                state.step_index = item.step_index
            if item.finished:
                state.finished = True

    def collect_outputs(
        self,
        states: list[RequestState],
        output: ModelForwardOutput,
    ) -> list[RunnerOutput]:
        return output.outputs

    def release(self, state: RequestState) -> None:
        state.payload = None

    @staticmethod
    def _payload(state: RequestState) -> RunnerState:
        if not isinstance(state.payload, RunnerState):
            raise TypeError(f"Expected RunnerState payload, got {type(state.payload).__name__}.")
        return state.payload

    @staticmethod
    def _batch_payload(batch: ForwardBatch) -> RunnerState:
        if not isinstance(batch.payload, RunnerState):
            raise TypeError(f"Expected RunnerState batch payload, got {type(batch.payload).__name__}.")
        return batch.payload

    def _profile_stage(self, state: RunnerState, name: str):
        profile = self._profiler(state)
        if profile is None:
            return nullcontext()
        return profile.stage(name, self._cuda_sync)

    def _profiler(self, state: RunnerState) -> RequestProfiler | None:
        profile = state.extra.get("profiler")
        if profile is None:
            return None
        if not isinstance(profile, RequestProfiler):
            raise TypeError(f"Expected RequestProfiler payload, got {type(profile).__name__}.")
        return profile

    def _emit_profile(self, state: RunnerState) -> None:
        profile = self._profiler(state)
        if profile is None:
            return
        profile.set_metadata(
            steps=state.step_index,
            total_steps=state.total_steps,
            height=state.extra.get("height"),
            width=state.extra.get("width"),
            num_frames=state.extra.get("num_frames"),
            guidance_scale=state.extra.get("guidance_scale"),
            prompt_cache_hit=state.extra.get("prompt_cache_hit"),
            image_cache_hit=state.extra.get("image_cache_hit"),
            condition_cache_hit=state.extra.get("condition_cache_hit"),
            condition_cache_mode=state.extra.get("condition_cache_mode"),
            latents_shape=state.extra.get("latents_shape"),
            condition_shape=state.extra.get("condition_shape"),
            first_frame_mask_shape=state.extra.get("first_frame_mask_shape"),
            denoise_latent_model_input_shape=state.extra.get("denoise_latent_model_input_shape"),
            denoise_timestep_shape=state.extra.get("denoise_timestep_shape"),
            condition_probe_enabled=state.extra.get("condition_probe_enabled"),
            condition_same_across_seed=state.extra.get("condition_same_across_seed"),
            first_frame_mask_same_across_seed=state.extra.get("first_frame_mask_same_across_seed"),
            latents_same_across_seed=state.extra.get("latents_same_across_seed"),
            condition_cache_candidate=state.extra.get("condition_cache_candidate"),
            **self.pipeline.runtime_info(),
        )
        if self.pipeline.config.use_cpu_offload:
            print(
                "[wllm-omni][profile] note cpu_offload=True; timings include CPU/GPU transfer overhead. "
                "Use --disable-cpu-offload for a GPU-resident baseline.",
                flush=True,
            )
        for line in profile.summary_lines():
            print(line, flush=True)

    def _cuda_sync(self) -> None:
        if not torch.cuda.is_available():
            return
        device = getattr(self.pipeline.pipe, "_execution_device", None)
        if device is None:
            return
        device = torch.device(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
