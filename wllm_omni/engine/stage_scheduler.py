from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

from wllm_omni.engine.connectors import ConnectorContext
from wllm_omni.engine.stage import Stage, StageOutput
from wllm_omni.engine.stage_graph import StageGraph, StageNode, StageResultStore
from wllm_omni.request import OmniRequest


@dataclass(slots=True)
class StageExecutionRecord:
    node_id: str
    stage_name: str
    request_id: str
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class StageSchedulerResult:
    root_request_id: str
    outputs: StageResultStore
    records: list[StageExecutionRecord]
    final_outputs: list[StageOutput]


class StageScheduler:
    """Executes a StageGraph in dependency order.

    This scheduler is the top-level omni scheduler. It decides which stage node
    is ready, runs that stage, then uses edge connectors to create downstream
    requests. It deliberately does not understand AR tokens, KV cache, or
    diffusion denoise steps.
    """

    def __init__(self, graph: StageGraph):
        self.graph = graph
        self.graph.validate()

    def run(self, root_request: OmniRequest) -> StageSchedulerResult:
        outputs = StageResultStore()
        records: list[StageExecutionRecord] = []
        completed: set[str] = set()
        scheduled: set[str] = set()
        requests: dict[str, OmniRequest] = {}

        for root in self.graph.roots():
            requests[root.node_id] = root_request

        while len(completed) < len(self.graph.nodes):
            ready_nodes = self.graph.ready_nodes(completed=completed, scheduled=scheduled)
            if not ready_nodes:
                remaining = sorted(set(self.graph.nodes) - completed)
                raise RuntimeError(f"StageGraph stalled; remaining nodes: {remaining}")

            for node in ready_nodes:
                scheduled.add(node.node_id)
                request = self._request_for_node(node, root_request, requests, outputs)
                output, elapsed_s = self._run_stage(node.stage, request)
                outputs.put(node.node_id, output)
                records.append(self._make_record(node, output, elapsed_s, root_request))
                completed.add(node.node_id)
                self._materialize_downstream_requests(node, root_request, requests, outputs)

        final_outputs = [outputs.get(node.node_id) for node in self.graph.leaves()]
        return StageSchedulerResult(
            root_request_id=root_request.request_id,
            outputs=outputs,
            records=records,
            final_outputs=final_outputs,
        )

    def _request_for_node(
        self,
        node: StageNode,
        root_request: OmniRequest,
        requests: dict[str, OmniRequest],
        outputs: StageResultStore,
    ) -> OmniRequest:
        if node.node_id in requests:
            return requests[node.node_id]

        in_edges = self.graph.in_edges(node.node_id)
        if len(in_edges) != 1:
            raise RuntimeError(
                f"Stage node {node.node_id!r} requires exactly one input request in V1, got {len(in_edges)}."
            )
        edge = in_edges[0]
        context = ConnectorContext(
            root_request=root_request,
            source_node=edge.source,
            target_node=edge.target,
            source_output=outputs.get(edge.source),
        )
        request = edge.connector.connect(context)
        requests[node.node_id] = request
        return request

    def _materialize_downstream_requests(
        self,
        node: StageNode,
        root_request: OmniRequest,
        requests: dict[str, OmniRequest],
        outputs: StageResultStore,
    ) -> None:
        for edge in self.graph.out_edges(node.node_id):
            if edge.target in requests:
                continue
            context = ConnectorContext(
                root_request=root_request,
                source_node=edge.source,
                target_node=edge.target,
                source_output=outputs.get(edge.source),
            )
            requests[edge.target] = edge.connector.connect(context)

    @staticmethod
    def _run_stage(stage: Stage, request: OmniRequest) -> tuple[StageOutput, float]:
        prepare_metadata = stage.prepare()
        start = perf_counter()
        output = stage.run(request)
        elapsed_s = perf_counter() - start
        if prepare_metadata:
            output.metadata.update(prepare_metadata)
        return output, elapsed_s

    def _make_record(
        self,
        node: StageNode,
        output: StageOutput,
        elapsed_s: float,
        root_request: OmniRequest,
    ) -> StageExecutionRecord:
        metadata = dict(output.metadata)
        metadata["elapsed_s"] = elapsed_s
        metadata["paradigm"] = node.stage.paradigm.value
        in_edges = self.graph.in_edges(node.node_id)
        if in_edges:
            edge = in_edges[0]
            metadata.setdefault("bridge", getattr(edge.connector, "name", type(edge.connector).__name__))
            metadata.setdefault("source_node", edge.source)
            metadata.setdefault("source_request_id", root_request.request_id)
        else:
            metadata.setdefault("bridge", "direct_request")
        return StageExecutionRecord(
            node_id=node.node_id,
            stage_name=node.stage.name,
            request_id=output.request_id,
            metadata=metadata,
        )
