from __future__ import annotations
import onnx
from onnx import numpy_helper, shape_inference, TensorProto
from ..ir import Graph, Node, Tensor
from ..ops import canonical_from_onnx

_DT = {TensorProto.FLOAT: "fp32", TensorProto.FLOAT16: "fp16", TensorProto.BFLOAT16: "bf16",
       TensorProto.DOUBLE: "fp64", TensorProto.INT8: "int8", TensorProto.UINT8: "uint8",
       TensorProto.INT32: "int32", TensorProto.INT64: "int64", TensorProto.BOOL: "bool",
       TensorProto.INT16: "int16"}


def _shape(vi_type, dyn):
    dims = []
    for d in vi_type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.dim_value > 0 else dyn)   # dynamic dim -> user value
    return dims


def from_onnx(path: str, dynamic_dim: int = 1) -> Graph:
    model = onnx.load(path)
    try:
        model = shape_inference.infer_shapes(model)
    except Exception as e:  # keep going with whatever shapes exist
        g_warn = [f"shape inference failed: {e}"]
    else:
        g_warn = []
    gr = model.graph
    G = Graph(name=gr.name or path, warnings=g_warn)

    weights = {i.name: i for i in gr.initializer}
    for name, init in weights.items():
        G.tensors[name] = Tensor(name, list(init.dims), _DT.get(init.data_type, "fp32"), True)
    for vi in list(gr.input) + list(gr.output) + list(gr.value_info):
        if vi.name in G.tensors:
            continue
        tt = vi.type.tensor_type
        G.tensors[vi.name] = Tensor(vi.name, _shape(vi.type, dynamic_dim),
                                    _DT.get(tt.elem_type, "fp32"), False)
    G.inputs = [i.name for i in gr.input if i.name not in weights]
    G.outputs = [o.name for o in gr.output]

    for i, n in enumerate(gr.node):
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute
                 if a.type in (a.INT, a.INTS, a.FLOAT, a.FLOATS, a.STRING)}
        # Constant nodes become weights
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    arr = numpy_helper.to_array(a.t)
                    G.tensors[n.output[0]] = Tensor(n.output[0], list(arr.shape), "fp32", True)
            continue
        canon = canonical_from_onnx(n.op_type)
        G.nodes.append(Node(n.name or f"{n.op_type}_{i}", canon,
                            [x for x in n.input if x], list(n.output), attrs, n.op_type))
    for n in G.nodes:
        for t in n.inputs + n.outputs:
            if t not in G.tensors:
                G.warnings.append(f"tensor '{t}' has no shape (node {n.name}); treated as scalar")
                G.tensors[t] = Tensor(t, [1], "fp32", False)
    return G
