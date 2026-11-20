"""Layer 2: roofline classification and latency, with hardware numbers chosen so the answer is obvious."""
import pytest
from onnx import helper as h
from rooftool.frontends.onnx_frontend import from_onnx
from rooftool.hw import custom_spec, HardwareSpec
from rooftool import roofline
from helpers import W, build_onnx

HW = custom_spec(peak_gflops=1000, bw_gbs=50)          # ridge = 1000/50 = 20 FLOP/byte


def run(tmp_path, nodes, inits, inputs, outputs, hw=HW):
    G = from_onnx(build_onnx(tmp_path / "m.onnx", nodes, inits, inputs, outputs))
    return roofline.run(G, hw)


def test_big_weights_small_activations_is_memory_bound(tmp_path):
    # 1x1024 @ 1024x1024: flops=2*1024*1024; bytes ~ 4 MiB weights -> AI ~ 0.5 << 20
    r = run(tmp_path, [h.make_node("Gemm", ["x", "w"], ["y"])], [W("w", 1024, 1024)],
            [("x", [1, 1024])], [("y", [1, 1024])])
    L = r["layers"][0]
    assert L["bound"] == "memory"
    assert L["ai_ideal"] == pytest.approx(2 * 1024 * 1024 / (4 * (1024 * 1024 + 1024 + 1024)))
    assert L["attainable_gflops"] == pytest.approx(L["ai_ideal"] * 50)


def test_square_matmul_is_compute_bound_and_latency_is_compute_time(tmp_path):
    n = 1024
    r = run(tmp_path, [h.make_node("MatMul", ["a", "b"], ["y"])], [],
            [("a", [n, n]), ("b", [n, n])], [("y", [n, n])])
    L = r["layers"][0]
    assert L["bound"] == "compute" and L["attainable_gflops"] == 1000
    assert L["latency_us_lb"] == pytest.approx(2 * n ** 3 / 1000e9 * 1e6)      # compute time dominates


def test_ridge_point(tmp_path):
    r = run(tmp_path, [h.make_node("Relu", ["x"], ["y"])], [], [("x", [1, 8])], [("y", [1, 8])])
    assert r["layers"][0]["ridge"] == 20


def test_shape_op_is_not_classified_compute_or_memory(tmp_path):
    r = run(tmp_path, [h.make_node("Flatten", ["x"], ["y"])], [], [("x", [1, 8, 4])], [("y", [1, 32])])
    assert r["layers"][0]["bound"] in ("free", "data-movement")


def test_missing_bandwidth_warns_and_does_not_crash(tmp_path):
    hw = HardwareSpec(id="X", name="x", peak_gflops={"fp32": 1000}, bw_gbs=None)
    r = run(tmp_path, [h.make_node("Relu", ["x"], ["y"])], [], [("x", [1, 8])], [("y", [1, 8])], hw)
    assert r["layers"][0]["bound"] is None
    assert any("bandwidth unknown" in w for w in r["warnings"])


def test_dtype_without_peak_falls_back_and_warns(tmp_path):
    hw = HardwareSpec(id="X", name="x", peak_gflops={"fp16": 2000}, bw_gbs=100)
    r = run(tmp_path, [h.make_node("Relu", ["x"], ["y"])], [], [("x", [1, 8])], [("y", [1, 8])], hw)
    assert r["layers"][0]["peak_dtype_used"] == "fp16"
    assert any("optimistic" in w for w in r["warnings"])


def test_summary_totals_equal_sum_of_layers(tmp_path):
    r = run(tmp_path, [h.make_node("Gemm", ["x", "w"], ["a"]), h.make_node("Relu", ["a"], ["y"])],
            [W("w", 16, 16)], [("x", [1, 16])], [("y", [1, 16])])
    assert r["summary"]["total_flops"] == sum(l["flops"] for l in r["layers"])
    assert r["summary"]["n_memory_bound"] + r["summary"]["n_compute_bound"] <= r["summary"]["n_layers"]
