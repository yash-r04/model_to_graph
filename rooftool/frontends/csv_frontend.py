"""CSV schema (one row per layer, rows in execution order; see examples/model_template.csv)
columns: layer, op, in_shape, out_shape, kernel, stride, groups, dtype, weight_shape, inputs
 - shapes are written like 1x3x224x224
 - kernel/stride like 3x3 / 1x1
 - weight_shape optional (computed for conv/gemm if blank)
 - inputs optional: ';'-separated layer names (default: previous row) -> gives DAG branches
"""
from __future__ import annotations
import pandas as pd
from ..ir import Graph, Node, Tensor
from ..ops import canonical_from_name


def _sh(s):
    if pd.isna(s) or str(s).strip() == "":
        return None
    return [int(x) for x in str(s).lower().replace(",", "x").split("x") if x.strip()]


def from_csv(path: str) -> Graph:
    df = pd.read_csv(path, dtype=str).fillna("")
    G = Graph(name=path)
    prev_out = None
    out_of = {}
    for i, r in df.iterrows():
        name = r["layer"] or f"layer_{i}"
        op = canonical_from_name(r["op"])
        dt = (r.get("dtype") or "fp32").lower()
        ins, outs = _sh(r["in_shape"]), _sh(r["out_shape"])
        in_names = []
        srcs = [s for s in str(r.get("inputs", "")).split(";") if s] or ([prev_name] if i else [])
        if not srcs:
            tn = f"{name}__in"
            G.tensors[tn] = Tensor(tn, ins, dt); G.inputs.append(tn); in_names.append(tn)
        else:
            in_names = [out_of[s] for s in srcs]
        on = f"{name}__out"
        G.tensors[on] = Tensor(on, outs or ins, dt)
        attrs = {}
        k, s, g = _sh(r.get("kernel")), _sh(r.get("stride")), r.get("groups")
        if k: attrs["kernel_shape"] = k
        if s: attrs["strides"] = s
        if g: attrs["group"] = int(g)
        ws = _sh(r.get("weight_shape"))
        if ws:
            wn = f"{name}__w"; G.tensors[wn] = Tensor(wn, ws, dt, True); in_names.append(wn)
        G.nodes.append(Node(name, op, in_names, [on], attrs, r["op"]))
        out_of[name] = on
        prev_name = name
    G.outputs = [G.nodes[-1].outputs[0]] if G.nodes else []
    return G
