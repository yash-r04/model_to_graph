"""Layer 4: frontends produce the same IR contract."""
import pytest
from onnx import helper as h
from rooftool.frontends import load_graph
from rooftool.frontends.csv_frontend import from_csv
from rooftool.frontends.onnx_frontend import from_onnx
from rooftool.hw import custom_spec
from rooftool import roofline
from helpers import W, build_onnx


def test_dynamic_batch_scales_flops_linearly(tmp_path):
    p = build_onnx(tmp_path / "m.onnx", [h.make_node("Gemm", ["x", "w"], ["y"])], [W("w", 32, 32)],
                   [("x", ["batch", 32])], [("y", ["batch", 32])])
    hw = custom_spec(1000, 50)
    f1 = roofline.run(from_onnx(p, 1), hw)["summary"]["total_flops"]
    f8 = roofline.run(from_onnx(p, 8), hw)["summary"]["total_flops"]
    assert f8 == 8 * f1


def test_onnx_dag_edges_include_skip_connection(tmp_path):
    p = build_onnx(tmp_path / "m.onnx",
                   [h.make_node("Relu", ["x"], ["a"], name="r"), h.make_node("Relu", ["a"], ["b"], name="s"),
                    h.make_node("Add", ["a", "b"], ["y"], name="add")], [],
                   [("x", [1, 4])], [("y", [1, 4])])
    edges = set(from_onnx(p).to_networkx().edges())
    assert edges == {("r", "s"), ("r", "add"), ("s", "add")}


def test_weights_are_marked_as_weights(tmp_path):
    G = from_onnx(build_onnx(tmp_path / "m.onnx", [h.make_node("Gemm", ["x", "w"], ["y"])], [W("w", 4, 4)],
                             [("x", [1, 4])], [("y", [1, 4])]))
    assert G.tensors["w"].is_weight and not G.tensors["x"].is_weight
    assert G.inputs == ["x"]


def test_csv_matches_equivalent_onnx(tmp_path):
    csv = tmp_path / "m.csv"
    csv.write_text("layer,op,in_shape,out_shape,kernel,stride,groups,dtype,weight_shape,inputs\n"
                   "c,Conv,1x3x8x8,1x4x8x8,3x3,1x1,1,fp32,4x3x3x3,\n")
    G = from_csv(str(csv))
    r = roofline.run(G, custom_spec(1000, 50))
    assert r["layers"][0]["macs"] == 6912                # same hand number as the ONNX conv test


def test_csv_inputs_column_creates_branches(tmp_path):
    csv = tmp_path / "m.csv"
    csv.write_text("layer,op,in_shape,out_shape,kernel,stride,groups,dtype,weight_shape,inputs\n"
                   "a,Relu,1x4,1x4,,,,fp32,,\n"
                   "b,Relu,1x4,1x4,,,,fp32,,a\n"
                   "c,Add,1x4,1x4,,,,fp32,,a;b\n")
    assert set(from_csv(str(csv)).to_networkx().edges()) == {("a", "b"), ("a", "c"), ("b", "c")}


def test_unknown_extension_rejected(tmp_path):
    f = tmp_path / "m.pt"; f.write_text("x")
    with pytest.raises(ValueError):
        load_graph(str(f))

