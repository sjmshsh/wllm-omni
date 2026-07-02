from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

from wllm_omni.model_types import ModelParadigm
from wllm_omni.models.ar_pipeline import ARTextOutput
from wllm_omni.request import OmniRequest
from wllm_omni.sampling_params import clone_sampling_params

from wllm_omni.engine.stage import StageOutput


@dataclass(slots=True)
class ConnectorContext:
    root_request: OmniRequest
    source_node: str
    target_node: str
    source_output: StageOutput


class StageConnector(ABC):
    """Transforms an upstream stage output into a downstream stage request."""

    @abstractmethod
    def connect(self, context: ConnectorContext) -> OmniRequest:
        pass


class ARToDiffusionConnector(StageConnector):
    """Bridge AR text output into a Wan image-to-video diffusion request."""

    def connect(self, context: ConnectorContext) -> OmniRequest:
        if not isinstance(context.source_output.data, ARTextOutput):
            raise TypeError(
                "ARToDiffusionConnector expects ARTextOutput, "
                f"got {type(context.source_output.data).__name__}."
            )
        sampling = clone_sampling_params(context.root_request.sampling_params)
        return OmniRequest(
            prompt=context.source_output.data.text,
            image=context.root_request.image,
            sampling_params=sampling,
            model_paradigm=ModelParadigm.DIFFUSION,
            request_id=context.root_request.request_id,
        )


class CallableARToDiffusionConnector(StageConnector):
    """Compatibility wrapper for older AR-output connector callables."""

    def __init__(self, connector: Callable[[OmniRequest, ARTextOutput], OmniRequest]):
        self.connector = connector

    def connect(self, context: ConnectorContext) -> OmniRequest:
        if not isinstance(context.source_output.data, ARTextOutput):
            raise TypeError(
                "CallableARToDiffusionConnector expects ARTextOutput, "
                f"got {type(context.source_output.data).__name__}."
            )
        return self.connector(context.root_request, context.source_output.data)
