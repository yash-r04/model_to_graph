"""Builds a small ResNet-like block + attention-ish MatMul/Softmax ONNX with batch as a dynamic dim."""
import numpy as np, onnx
from onnx import helper as h, TensorProto as TP, numpy_helper as nh

def W(name, *shape): return nh.from_array(np.random.randn(*shape).astype("float32"), name)
inits = [W("w1", 64, 3, 7, 7), W("w2", 64, 64, 3, 3), W("wfc", 64, 1000), W("bfc", 1000)]
nodes = [
  h.make_node("Conv", ["x", "w1"], ["c1"], name="conv1", kernel_shape=[7,7], strides=[2,2], pads=[3,3,3,3]),
  h.make_node("Relu", ["c1"], ["r1"], name="relu1"),
  h.make_node("MaxPool", ["r1"], ["p1"], name="pool1", kernel_shape=[3,3], strides=[2,2]),
  h.make_node("Conv", ["p1", "w2"], ["c2"], name="conv2", kernel_shape=[3,3], pads=[1,1,1,1]),
  h.make_node("Add", ["c2", "p1"], ["a1"], name="skip_add"),
  h.make_node("Relu", ["a1"], ["r2"], name="relu2"),
  h.make_node("GlobalAveragePool", ["r2"], ["g"], name="gap"),
  h.make_node("Flatten", ["g"], ["f"], name="flatten", axis=1),
  h.make_node("Gemm", ["f", "wfc", "bfc"], ["logits"], name="fc"),
  h.make_node("Softmax", ["logits"], ["y"], name="softmax"),
]
g = h.make_graph(nodes, "tiny_resnet",
    [h.make_tensor_value_info("x", TP.FLOAT, ["batch", 3, 224, 224])],
    [h.make_tensor_value_info("y", TP.FLOAT, ["batch", 1000])], inits)
m = h.make_model(g, opset_imports=[h.make_opsetid("", 13)]); m.ir_version = 8
onnx.checker.check_model(m); onnx.save(m, "examples/tiny_resnet.onnx"); print("saved")
