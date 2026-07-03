from __future__ import annotations

from dataclasses import dataclass, field

from wllm_omni.engine.connectors import StageConnector
from wllm_omni.engine.stage import Stage, StageOutput


@dataclass(slots=True)
class StageNode:
    node_id: str
    stage: Stage


@dataclass(slots=True)
class StageEdge:
    source: str
    target: str
    connector: StageConnector


@dataclass(slots=True)
class StageResultStore:
    outputs: dict[str, StageOutput] = field(default_factory=dict)

    def put(self, node_id: str, output: StageOutput) -> None:
        self.outputs[node_id] = output

    def get(self, node_id: str) -> StageOutput:
        try:
            return self.outputs[node_id]
        except KeyError as exc:
            raise KeyError(f"Missing stage output for node_id={node_id!r}.") from exc

    def has(self, node_id: str) -> bool:
        return node_id in self.outputs


class StageGraph:
    """A small explicit stage DAG for mini-Omni V1.

    This is the top-level omni graph. It does not schedule tokens or diffusion
    steps; it only represents stage dependencies and inter-stage connectors.
    """

    def __init__(self):
        self._nodes: dict[str, StageNode] = {}
        self._out_edges: dict[str, list[StageEdge]] = {}
        self._in_edges: dict[str, list[StageEdge]] = {}

    @property
    def nodes(self) -> dict[str, StageNode]:
        return dict(self._nodes)

    def add_node(self, node_id: str, stage: Stage) -> None:
        if node_id in self._nodes:
            raise ValueError(f"Duplicate stage node_id={node_id!r}.")
        self._nodes[node_id] = StageNode(node_id=node_id, stage=stage)
        self._out_edges.setdefault(node_id, [])
        self._in_edges.setdefault(node_id, [])

    def add_edge(self, source: str, target: str, connector: StageConnector) -> None:
        if source not in self._nodes:
            raise ValueError(f"Unknown source node_id={source!r}.")
        if target not in self._nodes:
            raise ValueError(f"Unknown target node_id={target!r}.")
        edge = StageEdge(source=source, target=target, connector=connector)
        self._out_edges.setdefault(source, []).append(edge)
        self._in_edges.setdefault(target, []).append(edge)

    def roots(self) -> list[StageNode]:
        return [node for node_id, node in self._nodes.items() if not self._in_edges.get(node_id)]

    def leaves(self) -> list[StageNode]:
        return [node for node_id, node in self._nodes.items() if not self._out_edges.get(node_id)]

    def out_edges(self, node_id: str) -> list[StageEdge]:
        self._require_node(node_id)
        return list(self._out_edges.get(node_id, []))

    def in_edges(self, node_id: str) -> list[StageEdge]:
        self._require_node(node_id)
        return list(self._in_edges.get(node_id, []))

    def ready_nodes(self, completed: set[str], scheduled: set[str]) -> list[StageNode]:
        ready: list[StageNode] = []
        for node_id, node in self._nodes.items():
            if node_id in completed or node_id in scheduled:
                continue
            dependencies = self._in_edges.get(node_id, [])
            if all(edge.source in completed for edge in dependencies):
                ready.append(node)
        return ready

    def validate(self) -> None:
        if not self._nodes:
            raise ValueError("StageGraph requires at least one node.")
        roots = self.roots()
        if not roots:
            raise ValueError("StageGraph must have at least one root node.")
        leaves = self.leaves()
        if not leaves:
            raise ValueError("StageGraph must have at least one leaf node.")
        multi_input_nodes = [node_id for node_id, edges in self._in_edges.items() if len(edges) > 1]
        if multi_input_nodes:
            raise ValueError(
                "StageGraph V1 supports at most one input edge per node; "
                f"got multi-input nodes: {sorted(multi_input_nodes)}"
            )
        self._validate_reachable_from_roots(roots)

    def _validate_reachable_from_roots(self, roots: list[StageNode]) -> None:
        visited: set[str] = set()
        stack = [node.node_id for node in roots]
        while stack:
            node_id = stack.pop()
            if node_id in visited:
                continue
            visited.add(node_id)
            stack.extend(edge.target for edge in self._out_edges.get(node_id, []))
        missing = set(self._nodes) - visited
        if missing:
            raise ValueError(f"StageGraph contains unreachable nodes: {sorted(missing)}")

    def _require_node(self, node_id: str) -> None:
        if node_id not in self._nodes:
            raise KeyError(f"Unknown stage node_id={node_id!r}.")
