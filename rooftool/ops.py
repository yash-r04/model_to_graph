"""Op registry: canonical op names + per-op formulas.
Each formula returns OpStats(macs, flops, weight_bytes, act_in_bytes, out_bytes, nocache_bytes).
Hardware-independent."""
from __future__ import annotations
from dataclasses import dataclass
from .ir import Graph, Node, DTYPE_BYTES

# ---------- name mapping ----------
_ONNX = {
    "Conv": "Conv", "ConvInteger": "Conv", "QLinearConv": "Conv", "ConvTranspose": "Conv",
    "Gemm": "Gemm", "MatMul": "Gemm", "MatMulInteger": "Gemm", "QLinearMatMul": "Gemm",
    "MaxPool": "Pool", "AveragePool": "Pool", "GlobalAveragePool": "GlobalPool", "GlobalMaxPool": "GlobalPool",
    "Relu": "Elementwise", "Add": "Elementwise", "Sub": "Elementwise", "Mul": "Elementwise",
    "Div": "Elementwise", "Clip": "Elementwise", "LeakyRelu": "Elementwise", "Neg": "Elementwise",
    "Sigmoid": "Transcendental", "Tanh": "Transcendental", "Gelu": "Transcendental", "Erf": "Transcendental",
    "HardSwish": "Elementwise", "HardSigmoid": "Elementwise", "Sqrt": "Transcendental", "Pow": "Transcendental",
    "Softmax": "Softmax", "LogSoftmax": "Softmax",
    "BatchNormalization": "Norm", "LayerNormalization": "Norm", "InstanceNormalization": "Norm",
    "GroupNormalization": "Norm", "LRN": "Norm", "SkipLayerNormalization": "Norm",
    "Reshape": "Shape", "Flatten": "Shape", "Squeeze": "Shape", "Unsqueeze": "Shape", "Identity": "Shape",
    "Shape": "Shape", "Gather": "Shape", "Slice": "Shape", "Cast": "Shape", "Dropout": "Shape",
    "Transpose": "DataMove", "Concat": "DataMove", "Split": "DataMove", "Pad": "DataMove", "Expand": "DataMove",
    "Resize": "Resize", "Upsample": "Resize",
    "ReduceMean": "Reduce", "ReduceSum": "Reduce", "ReduceMax": "Reduce",
    "QuantizeLinear": "Elementwise", "DequantizeLinear": "Elementwise",
}
_TFL = {
    "CONV_2D": "Conv", "DEPTHWISE_CONV_2D": "Conv", "TRANSPOSE_CONV": "Conv",
    "FULLY_CONNECTED": "Gemm", "BATCH_MATMUL": "Gemm",
    "MAX_POOL_2D": "Pool", "AVERAGE_POOL_2D": "Pool", "MEAN": "Reduce",
    "ADD": "Elementwise", "SUB": "Elementwise", "MUL": "Elementwise", "RELU": "Elementwise",
    "RELU6": "Elementwise", "LEAKY_RELU": "Elementwise", "HARD_SWISH": "Elementwise",
    "QUANTIZE": "Elementwise", "DEQUANTIZE": "Elementwise",
    "LOGISTIC": "Transcendental", "TANH": "Transcendental", "GELU": "Transcendental",
    "SOFTMAX": "Softmax", "RESHAPE": "Shape", "SQUEEZE": "Shape", "EXPAND_DIMS": "Shape",
    "TRANSPOSE": "DataMove", "CONCATENATION": "DataMove", "PAD": "DataMove",
    "RESIZE_BILINEAR": "Resize", "RESIZE_NEAREST_NEIGHBOR": "Resize",
}
_CSV = {k.lower(): v for k, v in {**_ONNX, **{
    "conv2d": "Conv", "depthwiseconv": "Conv", "dense": "Gemm", "fc": "Gemm", "linear": "Gemm",
    "matmul": "Gemm", "pool": "Pool", "relu": "Elementwise", "add": "Elementwise",
    "softmax": "Softmax", "layernorm": "Norm", "batchnorm": "Norm", "reshape": "Shape",
    "globalpool": "GlobalPool", "gelu": "Transcendental"}}.items()}
_CSV.update({v.lower(): v for v in set(_ONNX.values())})


def canonical_from_onnx(op): return _ONNX.get(op, "Unknown")
def canonical_from_tflite(op): return _TFL.get(op, "Unknown")
def canonical_from_name(op): return _CSV.get(str(op).strip().lower(), "Unknown")


# ---------- formulas ----------
@dataclass
class OpStats:
    macs: float = 0.0
    flops: float = 0.0
    weight_bytes: float = 0.0
    act_in_bytes: float = 0.0
    out_bytes: float = 0.0
    nocache_bytes: float = 0.0
    note: str = ""


def _prod(xs):
    p = 1
    for x in xs: p *= max(int(x), 1)
    return p


def analyze_node(G: Graph, n: Node) -> OpStats:
    T = G.tensors
    ins = [T[t] for t in n.inputs if t in T]
    outs = [T[t] for t in n.outputs if t in T]
    w = [t for t in ins if t.is_weight]
    a = [t for t in ins if not t.is_weight]
    out_numel = sum(t.numel for t in outs)
    s = OpStats(weight_bytes=sum(t.nbytes for t in w),
                act_in_bytes=sum(t.nbytes for t in a),
                out_bytes=sum(t.nbytes for t in outs))
    bpe = DTYPE_BYTES.get(outs[0].dtype if outs else "fp32", 4)
    op = n.op_type

    if op == "Conv":
        # MACs = out_elements * (weight_elements / Cout)  -> works for grouped/depthwise & NCHW/NHWC
        wt = w[0] if w else None
        if wt is None and len(ins) > 1: wt = ins[1]
        if wt is not None and outs:
            cout_axis = -1 if n.orig_op == "DEPTHWISE_CONV_2D" else 0
            cout = max(wt.shape[cout_axis], 1)
            s.macs = out_numel * wt.numel / cout
            if len(n.inputs) >= 3:      # bias input: one add per output element (onnx-tool counts it)
                s.macs += out_numel
        s.nocache_bytes = s.macs * 2 * bpe + s.out_bytes
    elif op == "Gemm":
        if len(ins) >= 2 and outs:
            A = ins[0]
            k = A.shape[-2] if n.attrs.get("transA", 0) else A.shape[-1]
            s.macs = out_numel * k
            if len(ins) >= 3:           # bias add: one extra op per output element (onnx-tool counts it too)
                s.macs += out_numel
        s.nocache_bytes = s.macs * 2 * bpe + s.out_bytes
        if not w and len(a) >= 2: s.note = "activation x activation (e.g. attention)"
    elif op == "Pool":
        k = _prod(n.attrs.get("kernel_shape", [2, 2]))
        s.flops = out_numel * k
    elif op == "GlobalPool":
        s.flops = sum(t.numel for t in a)
    elif op == "Elementwise":
        s.flops = out_numel
        if n.orig_op == "Clip":     # ReLU6 = min and max: two comparisons per element (onnx-tool convention)
            s.flops = 2 * out_numel
    elif op == "Transcendental":
        s.flops = 8 * out_numel
    elif op == "Softmax":
        s.flops = 5 * out_numel
    elif op == "Norm":
        s.flops = 5 * out_numel
    elif op == "Resize":
        s.flops = 4 * out_numel
    elif op == "Reduce":
        s.flops = sum(t.numel for t in a)
    elif op == "Shape":
        s.act_in_bytes = s.out_bytes = s.weight_bytes = 0   # metadata only, no data movement
        s.note = "metadata-only (no copy assumed)"
    elif op == "DataMove":
        s.note = "pure data movement"
    else:
        G.warnings.append(f"unsupported op '{n.orig_op}' in node {n.name}: counted as 0 FLOPs")
        s.note = "UNSUPPORTED"

    if s.macs: s.flops = 2 * s.macs
    if not s.nocache_bytes:
        s.nocache_bytes = s.weight_bytes + s.act_in_bytes + s.out_bytes
    return s
