"""TFLite frontend using the `tflite` flatbuffer package (pip install tflite).
Shapes and quantisation (dtype) are read straight from the file."""
from __future__ import annotations
from ..ir import Graph, Node, Tensor
from ..ops import canonical_from_tflite

_DT = {0: "fp32", 1: "fp16", 2: "int32", 3: "uint8", 4: "int64", 6: "bool", 7: "int16",
       9: "int8", 10: "fp64"}


def from_tflite(path: str, dynamic_dim: int = 1) -> Graph:
    import tflite
    buf = open(path, "rb").read()
    m = tflite.Model.GetRootAsModel(buf, 0)
    sg = m.Subgraphs(0)
    opnames = {v: k for k, v in vars(tflite.BuiltinOperator).items() if not k.startswith("_")}
    G = Graph(name=path)
    for i in range(sg.TensorsLength()):
        t = sg.Tensors(i)
        shape = [d if d > 0 else dynamic_dim for d in (t.ShapeAsNumpy().tolist()
                 if t.ShapeLength() else [1])]
        data_len = m.Buffers(t.Buffer()).DataLength()
        G.tensors[f"t{i}"] = Tensor(f"t{i}", shape, _DT.get(t.Type(), "fp32"), is_weight=data_len > 0)
    G.inputs = [f"t{sg.Inputs(i)}" for i in range(sg.InputsLength())]
    G.outputs = [f"t{sg.Outputs(i)}" for i in range(sg.OutputsLength())]
    for j in range(sg.OperatorsLength()):
        op = sg.Operators(j)
        code = m.OperatorCodes(op.OpcodeIndex())
        bi = max(code.BuiltinCode(), code.DeprecatedBuiltinCode())
        oname = opnames.get(bi, str(bi))
        ins = [f"t{op.Inputs(k)}" for k in range(op.InputsLength()) if op.Inputs(k) >= 0]
        outs = [f"t{op.Outputs(k)}" for k in range(op.OutputsLength())]
        G.nodes.append(Node(f"{oname}_{j}", canonical_from_tflite(oname), ins, outs, {}, oname))
    return G
