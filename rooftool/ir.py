"""Internal IR: the single contract between frontends, analysis and rendering."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional

DTYPE_BYTES = {"fp64": 8, "fp32": 4, "tf32": 4, "fp16": 2, "bf16": 2, "fp8": 1,
               "int64": 8, "int32": 4, "int16": 2, "int8": 1, "uint8": 1,
               "int4": 0.5, "bool": 1}


@dataclass
class Tensor:
    name: str
    shape: List[int]
    dtype: str = "fp32"
    is_weight: bool = False

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= max(int(d), 1)
        return n

    @property
    def nbytes(self) -> float:
        return self.numel * DTYPE_BYTES.get(self.dtype, 4)


@dataclass
class Node:
    name: str
    op_type: str                      # canonical op name (see ops.CANONICAL)
    inputs: List[str]
    outputs: List[str]
    attrs: Dict = field(default_factory=dict)
    orig_op: str = ""                 # original op name in the source format


@dataclass
class Graph:
    name: str
    nodes: List[Node] = field(default_factory=list)
    tensors: Dict[str, Tensor] = field(default_factory=dict)
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_networkx(self):
        import networkx as nx
        g = nx.DiGraph()
        producer = {o: n.name for n in self.nodes for o in n.outputs}
        for n in self.nodes:
            g.add_node(n.name, op=n.op_type)
        for n in self.nodes:
            for t in n.inputs:
                if t in producer:
                    g.add_edge(producer[t], n.name, tensor=t)
        return g
