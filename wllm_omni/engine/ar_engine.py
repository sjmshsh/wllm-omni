from __future__ import annotations

from wllm_omni.config import EngineConfig
from wllm_omni.engine.model_runner import ModelRunner
from wllm_omni.model_types import ModelParadigm
from wllm_omni.models.ar_executor import ARExecutor
from wllm_omni.models.ar_pipeline import ARPipeline, ARTextOutput, TransformersARPipeline
from wllm_omni.request import OmniRequest
from wllm_omni.sched.request_scheduler import RequestScheduler


class AREngine:
    """Request-level AR engine for mini-Omni V0.

    This intentionally starts with RequestScheduler. Future vLLM-style AR
    optimization should replace it with a token/KV-aware scheduler without
    changing the stage interface.
    """

    def __init__(self, config: EngineConfig, pipeline: ARPipeline | None = None):
        self.config = config
        self.scheduler = RequestScheduler(max_num_running_reqs=config.max_num_seqs)
        self.runner = ModelRunner(config, executors=[ARExecutor(pipeline or self._make_pipeline(config))])

    def generate(self, request: OmniRequest) -> ARTextOutput:
        request.model_paradigm = ModelParadigm.AUTOREGRESSIVE
        self.scheduler.add_request(request)
        outputs: list[ARTextOutput] = []
        while self.scheduler.has_requests():
            sched_output = self.scheduler.schedule()
            if sched_output.is_empty:
                break

            runner_output = self.runner.execute(sched_output)
            finished_req_ids = self.scheduler.update_from_output(sched_output, runner_output)
            for finished_req_id in finished_req_ids:
                self.scheduler.pop_request_state(finished_req_id)

            for item in runner_output.outputs:
                if item.error is not None:
                    raise RuntimeError(item.error)
                if item.finished and isinstance(item.result, ARTextOutput):
                    outputs.append(item.result)

        if not outputs:
            raise RuntimeError("AR generation finished without output.")
        return outputs[0]

    @staticmethod
    def _make_pipeline(config: EngineConfig) -> ARPipeline | None:
        if config.ar_model is None:
            return None
        return TransformersARPipeline(
            config.ar_model,
            device=config.device,
            dtype=config.dtype,
            local_files_only=config.local_files_only,
            max_new_tokens=config.ar_max_new_tokens,
        )
