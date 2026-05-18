from nanovllm_omni.config import EngineConfig
from nanovllm_omni.engine.model_runner import ModelRunner
from nanovllm_omni.outputs import OmniOutput
from nanovllm_omni.request import OmniRequest
from nanovllm_omni.sched.step_scheduler import StepScheduler


class OmniEngine:

    def __init__(self, config: EngineConfig):
        self.config = config
        # The current teaching engine executes one step for one request at a time.
        self.scheduler = StepScheduler(max_num_running_reqs=1)
        self.runner = ModelRunner(config)

    def generate(self, requests: OmniRequest | list[OmniRequest]) -> list[OmniOutput]:
        if isinstance(requests, OmniRequest):
            requests = [requests]

        for request in requests:
            self.scheduler.add_request(request)

        outputs: list[OmniOutput] = []
        while self.scheduler.has_requests():
            sched_output = self.scheduler.schedule()
            if sched_output.is_empty:
                break

            runner_output = self.runner.execute_stepwise(sched_output)
            finished_req_ids = self.scheduler.update_from_output(sched_output, runner_output)
            for finished_req_id in finished_req_ids:
                self.scheduler.pop_request_state(finished_req_id)

            if runner_output.result is not None:
                outputs.append(runner_output.result)

        return outputs
