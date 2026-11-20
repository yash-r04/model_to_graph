"""Layer 5: end to end. Output contract that teammates depend on."""
import json
import networkx as nx
from rooftool import analyze
from helpers import build_onnx
from onnx import helper as h

REQUIRED_LAYER_KEYS = {"name", "op", "dtype", "flops", "macs", "weight_bytes", "in_bytes", "out_bytes",
                       "bytes_ideal", "bytes_nocache", "ai_ideal", "ai_nocache", "bound",
                       "attainable_gflops", "latency_us_lb", "fits_onchip"}


def test_end_to_end_outputs_and_schema(tmp_path):
    p = build_onnx(tmp_path / "m.onnx", [h.make_node("Relu", ["x"], ["a"], name="r"),
                                         h.make_node("Relu", ["a"], ["y"], name="s")], [],
                   [("x", [1, 8])], [("y", [1, 8])])
    out = tmp_path / "out"
    G, res = analyze(p, hw="HW-GPU-002", out=str(out))
    for f in ("result.json", "layers.csv", "dag.dot", "roofline.html", "index.html"):
        assert (out / f).exists(), f
    d = json.load(open(out / "result.json"))
    assert {"graph", "layers", "summary", "hw", "warnings"} <= set(d)
    assert REQUIRED_LAYER_KEYS <= set(d["layers"][0])
    assert {"nodes", "edges", "tensors"} <= set(d["graph"])
    assert nx.is_directed_acyclic_graph(G.to_networkx())
    assert len(d["layers"]) == len(d["graph"]["nodes"]) == 2


def test_two_number_mode_needs_no_catalogue(tmp_path):
    p = build_onnx(tmp_path / "m.onnx", [h.make_node("Relu", ["x"], ["y"])], [], [("x", [1, 8])], [("y", [1, 8])])
    _, res = analyze(p, out=str(tmp_path / "o"), peak_gflops=1000, bw=50)
    assert res["hw"]["bw_gbs"] == 50
