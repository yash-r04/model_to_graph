"""Helpers to build tiny ONNX models with known, hand-computable FLOPs."""
import numpy as np
import onnx
from onnx import helper as h, TensorProto as TP, numpy_helper as nh


def W(name, *shape, dtype="float32"):
    return nh.from_array(np.random.randn(*shape).astype(dtype), name)


def build_onnx(path, nodes, inits, inputs, outputs, opset=13):
    """inputs/outputs: list of (name, shape). shape may contain a string for a dynamic dim."""
    g = h.make_graph(
        nodes, "t",
        [h.make_tensor_value_info(n, TP.FLOAT, s) for n, s in inputs],
        [h.make_tensor_value_info(n, TP.FLOAT, s) for n, s in outputs], inits)
    m = h.make_model(g, opset_imports=[h.make_opsetid("", opset)])
    m.ir_version = 8
    onnx.save(m, str(path))
    return str(path)
