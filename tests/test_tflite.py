"""Layer 4b: TFLite frontend, against a real .tflite built by examples/make_test_tflite.py.

Model (hand-computed numbers, same as examples/tiny_cnn.csv plus the two bias terms):
  CONV_2D 4x3x3x3 + bias -> 1x8x8x4 : 256 outputs * 27 MACs + 256 bias = 7168
  RELU                               : 256 FLOPs
  DEPTHWISE_CONV_2D 1x3x3x4          : 256 outputs * 9 MACs            = 2304
  RESHAPE                            : metadata only
  FULLY_CONNECTED 10x256 + bias      : 10 * 256 + 10                   = 2570
  SOFTMAX 1x10                       : 5 * 10 = 50 FLOPs
"""
import importlib.util
import pathlib
import networkx as nx
import pytest

pytest.importorskip("tflite")
from rooftool import analyze, roofline
from rooftool.frontends import load_graph
from rooftool.frontends.tflite_frontend import from_tflite
from rooftool.hw import custom_spec
from rooftool.ops import analyze_node

EX = pathlib.Path(__file__).resolve().parent.parent / "examples"
FP32, INT8 = EX / "tiny_cnn.tflite", EX / "tiny_cnn_int8.tflite"


def _generator():
    spec = importlib.util.spec_from_file_location("make_test_tflite", EX / "make_test_tflite.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def layers(path, **kw):
    G = from_tflite(str(path), **kw)
    return G, {n.name: analyze_node(G, n) for n in G.nodes}


# ---------- structure ----------
def test_layer_count_ops_and_order():
    G = from_tflite(str(FP32))
    assert [n.orig_op for n in G.nodes] == ["CONV_2D", "RELU", "DEPTHWISE_CONV_2D", "RESHAPE",
                                            "FULLY_CONNECTED", "SOFTMAX"]
    assert [n.op_type for n in G.nodes] == ["Conv", "Elementwise", "Conv", "Shape", "Gemm", "Softmax"]
    assert G.warnings == []


def test_graph_inputs_outputs_and_shapes():
    G = from_tflite(str(FP32))
    assert G.inputs == ["t0"] and G.outputs == ["t12"]
    assert G.tensors["t0"].shape == [1, 8, 8, 3]
    assert G.tensors["t12"].shape == [1, 10]


def test_weights_marked_activations_not():
    G = from_tflite(str(FP32))
    weights = {k for k, t in G.tensors.items() if t.is_weight}
    # conv_w, conv_b, dw_w, reshape-shape constant, fc_w, fc_b
    assert weights == {"t1", "t2", "t5", "t7", "t9", "t10"}


def test_dag_is_a_linear_chain():
    g = from_tflite(str(FP32)).to_networkx()
    assert nx.is_directed_acyclic_graph(g)
    names = [n for n in g.nodes]
    assert set(g.edges()) == set(zip(names, names[1:]))


# ---------- hand-computed op numbers ----------
def test_macs_and_flops_per_layer():
    _, s = layers(FP32)
    assert s["CONV_2D_0"].macs == 256 * 27 + 256 == 7168
    assert s["DEPTHWISE_CONV_2D_2"].macs == 256 * 9 == 2304      # cout axis is LAST for depthwise
    assert s["FULLY_CONNECTED_4"].macs == 10 * 256 + 10 == 2570
    assert s["RELU_1"].flops == 256
    assert s["SOFTMAX_5"].flops == 50
    assert s["CONV_2D_0"].flops == 2 * 7168


def test_reshape_is_free():
    _, s = layers(FP32)
    r = s["RESHAPE_3"]
    assert (r.flops, r.weight_bytes, r.act_in_bytes, r.out_bytes) == (0, 0, 0, 0)


def test_fp32_weight_bytes():
    _, s = layers(FP32)
    assert s["CONV_2D_0"].weight_bytes == (108 + 4) * 4          # 4x3x3x3 weights + 4 biases
    assert s["DEPTHWISE_CONV_2D_2"].weight_bytes == 36 * 4
    assert s["FULLY_CONNECTED_4"].weight_bytes == (2560 + 10) * 4


def test_summary_totals():
    r = roofline.run(from_tflite(str(FP32)), custom_spec(1000, 50))["summary"]
    assert r["n_layers"] == 6
    assert r["total_macs"] == 7168 + 2304 + 2570
    assert r["total_flops"] == 2 * 12042 + 256 + 50


# ---------- quantised model: dtype is read from the file ----------
def test_int8_dtypes_read_from_file():
    G = from_tflite(str(INT8))
    assert G.tensors["t0"].dtype == "int8"
    assert G.tensors["t1"].dtype == "int8" and G.tensors["t1"].is_weight
    assert G.tensors["t2"].dtype == "int32"                      # TFLite keeps int8-model biases in int32
    assert G.tensors["t7"].dtype == "int32"


def test_int8_same_flops_fewer_bytes():
    _, f = layers(FP32)
    _, q = layers(INT8)
    for k in f:
        assert q[k].macs == f[k].macs and q[k].flops == f[k].flops
    assert q["CONV_2D_0"].weight_bytes == 108 + 4 * 4            # int8 weights + int32 bias
    assert q["CONV_2D_0"].out_bytes == 256                        # 1 byte/element vs 4
    hw = custom_spec(1000, 50)
    b32 = roofline.run(from_tflite(str(FP32)), hw)["summary"]["total_bytes_ideal"]
    b8 = roofline.run(from_tflite(str(INT8)), hw)["summary"]["total_bytes_ideal"]
    assert b8 < b32 / 2


# ---------- dynamic batch ----------
def test_dynamic_dim_scales_flops(tmp_path):
    p = tmp_path / "dyn.tflite"
    _generator().build(p, 0, 0, 0, batch=-1)
    hw = custom_spec(1000, 50)
    f1 = roofline.run(from_tflite(str(p), 1), hw)["summary"]["total_flops"]
    f8 = roofline.run(from_tflite(str(p), 8), hw)["summary"]["total_flops"]
    assert f1 == roofline.run(from_tflite(str(FP32)), hw)["summary"]["total_flops"]
    assert f8 == 8 * f1


# ---------- plumbing ----------
def test_load_graph_dispatches_on_extension():
    assert len(load_graph(str(FP32)).nodes) == 6


def test_committed_files_match_generator(tmp_path):
    """Guards against someone editing the generator and forgetting to regenerate the samples."""
    g = _generator()
    g.build(tmp_path / "a.tflite", 0, 0, 0)
    g.build(tmp_path / "b.tflite", 9, 9, 2)
    assert (tmp_path / "a.tflite").read_bytes() == FP32.read_bytes()
    assert (tmp_path / "b.tflite").read_bytes() == INT8.read_bytes()


def test_corrupt_file_raises(tmp_path):
    p = tmp_path / "bad.tflite"
    p.write_bytes(b"this is not a flatbuffer")
    with pytest.raises(Exception):
        from_tflite(str(p))


def test_end_to_end_outputs(tmp_path):
    out = tmp_path / "out"
    G, res = analyze(str(FP32), out=str(out), peak_gflops=1000, bw=50)
    for f in ("result.json", "layers.csv", "dag.dot", "roofline.html", "index.html"):
        assert (out / f).exists(), f
    assert len(res["layers"]) == 6
    assert all(l["bound"] for l in res["layers"])
