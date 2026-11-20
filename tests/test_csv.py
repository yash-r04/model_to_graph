"""Layer 4c: CSV frontend, against examples/tiny_cnn.csv and edge cases.

tiny_cnn.csv mirrors the TFLite sample except that CSV has no bias column, so
CSV MACs = TFLite MACs minus the bias terms (256 for the conv, 10 for the FC).
"""
import pathlib
import networkx as nx
import pytest
from rooftool import analyze, roofline
from rooftool.frontends import load_graph
from rooftool.frontends.csv_frontend import from_csv
from rooftool.hw import custom_spec
from rooftool.ops import analyze_node

EX = pathlib.Path(__file__).resolve().parent.parent / "examples"
SAMPLE = EX / "tiny_cnn.csv"
HEADER = "layer,op,in_shape,out_shape,kernel,stride,groups,dtype,weight_shape,inputs\n"


def csv(tmp_path, body, header=HEADER):
    p = tmp_path / "m.csv"
    p.write_text(header + body)
    return str(p)


def stats(path):
    G = from_csv(path)
    return G, {n.name: analyze_node(G, n) for n in G.nodes}


# ---------- the sample file ----------
def test_sample_structure():
    G = from_csv(str(SAMPLE))
    assert [n.name for n in G.nodes] == ["conv", "relu", "dw", "flatten", "fc", "softmax"]
    assert [n.op_type for n in G.nodes] == ["Conv", "Elementwise", "Conv", "Shape", "Gemm", "Softmax"]
    assert G.inputs == ["conv__in"] and G.outputs == ["softmax__out"]
    assert G.warnings == []


def test_sample_default_input_is_previous_row():
    edges = set(from_csv(str(SAMPLE)).to_networkx().edges())
    assert edges == {("conv", "relu"), ("relu", "dw"), ("dw", "flatten"), ("flatten", "fc"), ("fc", "softmax")}


def test_sample_hand_numbers():
    _, s = stats(str(SAMPLE))
    assert s["conv"].macs == 256 * 27 == 6912
    assert s["dw"].macs == 256 * 9 == 2304                        # groups=4, weight 4x1x3x3
    assert s["fc"].macs == 10 * 256 == 2560
    assert s["relu"].flops == 256 and s["softmax"].flops == 50
    assert s["flatten"].flops == 0 and s["flatten"].out_bytes == 0
    assert s["conv"].weight_bytes == 432 and s["fc"].weight_bytes == 10240


def test_sample_matches_tflite_sample_minus_bias():
    pytest.importorskip("tflite")
    from rooftool.frontends.tflite_frontend import from_tflite
    hw = custom_spec(1000, 50)
    rc = roofline.run(from_csv(str(SAMPLE)), hw)["summary"]
    rt = roofline.run(from_tflite(str(EX / "tiny_cnn.tflite")), hw)["summary"]
    assert rt["total_macs"] - rc["total_macs"] == 256 + 10


def test_sample_end_to_end(tmp_path):
    out = tmp_path / "out"
    G, res = analyze(str(SAMPLE), out=str(out), peak_gflops=1000, bw=50)
    for f in ("result.json", "layers.csv", "dag.dot", "roofline.html", "index.html"):
        assert (out / f).exists(), f
    assert nx.is_directed_acyclic_graph(G.to_networkx())
    assert res["summary"]["n_layers"] == 6 and res["summary"]["total_macs"] == 6912 + 2304 + 2560


def test_template_file_loads():
    """The shipped template (with a skip connection) must keep working."""
    G = from_csv(str(EX / "model_template.csv"))
    assert ("conv2", "add") in G.to_networkx().edges() and ("pool1", "add") in G.to_networkx().edges()
    assert G.warnings == []


# ---------- column handling ----------
def test_dtype_int8_shrinks_bytes_not_flops(tmp_path):
    row = "c,Conv,1x3x8x8,1x4x8x8,3x3,1x1,1,{},4x3x3x3,\n"
    _, a = stats(csv(tmp_path, row.format("fp32")))
    _, b = stats(csv(tmp_path, row.format("int8")))
    assert a["c"].macs == b["c"].macs == 6912
    assert b["c"].weight_bytes == 108 and a["c"].weight_bytes == 432


def test_blank_dtype_defaults_to_fp32(tmp_path):
    _, s = stats(csv(tmp_path, "c,Conv,1x3x8x8,1x4x8x8,3x3,1x1,1,,4x3x3x3,\n"))
    assert s["c"].weight_bytes == 432


@pytest.mark.parametrize("alias,canon", [("Conv2D", "Conv"), ("conv", "Conv"), ("CONV", "Conv"),
                                          ("Dense", "Gemm"), ("fc", "Gemm"), ("Linear", "Gemm"),
                                          ("ReLU", "Elementwise"), ("LayerNorm", "Norm")])
def test_op_name_aliases_and_case(tmp_path, alias, canon):
    G = from_csv(csv(tmp_path, f"a,{alias},1x4,1x4,,,,fp32,,\n"))
    assert G.nodes[0].op_type == canon


def test_shape_separators_and_case(tmp_path):
    G = from_csv(csv(tmp_path, 'a,Relu,1X4,"1,4",,,,fp32,,\n'))
    assert G.tensors["a__in"].shape == [1, 4] and G.tensors["a__out"].shape == [1, 4]


def test_blank_out_shape_falls_back_to_in_shape(tmp_path):
    G = from_csv(csv(tmp_path, "a,Relu,1x4,,,,,fp32,,\n"))
    assert G.tensors["a__out"].shape == [1, 4]


def test_blank_layer_name_gets_generated(tmp_path):
    G = from_csv(csv(tmp_path, ",Relu,1x4,1x4,,,,fp32,,\n,Relu,1x4,1x4,,,,fp32,,\n"))
    assert [n.name for n in G.nodes] == ["layer_0", "layer_1"]


def test_pool_flops_use_kernel(tmp_path):
    # out 1x4x4x4 = 64 elements * 3x3 window
    _, s = stats(csv(tmp_path, "p,MaxPool,1x4x8x8,1x4x4x4,3x3,2x2,,fp32,,\n"))
    assert s["p"].flops == 64 * 9


def test_unknown_op_warns_and_counts_zero(tmp_path):
    G, s = stats(csv(tmp_path, "a,Relu,1x4,1x4,,,,fp32,,\nb,FooBar,1x4,1x4,,,,fp32,,\n"))
    assert s["b"].flops == 0
    assert any("FooBar" in w for w in G.warnings)


def test_extra_columns_are_ignored(tmp_path):
    p = csv(tmp_path, "a,Relu,1x4,1x4,,,,fp32,,,hello\n", header=HEADER.strip() + ",comment\n")
    assert len(from_csv(p).nodes) == 1


def test_header_only_gives_empty_graph(tmp_path):
    G = from_csv(csv(tmp_path, ""))
    assert G.nodes == [] and G.outputs == []


# ---------- DAG wiring ----------
def test_inputs_column_creates_branches_and_second_input_tensor(tmp_path):
    G = from_csv(csv(tmp_path, "a,Relu,1x4,1x4,,,,fp32,,\nb,Relu,1x4,1x4,,,,fp32,,a\nc,Add,1x4,1x4,,,,fp32,,a;b\n"))
    assert G.nodes[2].inputs == ["a__out", "b__out"]


def test_only_first_row_gets_a_graph_input(tmp_path):
    G = from_csv(csv(tmp_path, "a,Relu,1x4,1x4,,,,fp32,,\nb,Relu,1x4,1x4,,,,fp32,,\n"))
    assert G.inputs == ["a__in"]


def test_final_row_is_graph_output(tmp_path):
    G = from_csv(csv(tmp_path, "a,Relu,1x4,1x4,,,,fp32,,\nb,Relu,1x4,1x4,,,,fp32,,\n"))
    assert G.outputs == ["b__out"]


# ---------- bad input must fail loudly ----------
def test_reference_to_unknown_layer_raises(tmp_path):
    with pytest.raises((KeyError, ValueError)):
        from_csv(csv(tmp_path, "a,Relu,1x4,1x4,,,,fp32,,nothere\n"))


def test_forward_reference_raises(tmp_path):
    with pytest.raises((KeyError, ValueError)):
        from_csv(csv(tmp_path, "a,Relu,1x4,1x4,,,,fp32,,b\nb,Relu,1x4,1x4,,,,fp32,,\n"))


def test_missing_required_column_raises(tmp_path):
    with pytest.raises((KeyError, ValueError)):
        from_csv(csv(tmp_path, "a,Relu\n", header="layer,op\n"))


def test_dispatch_by_extension():
    assert len(load_graph(str(SAMPLE)).nodes) == 6


# ---------- KNOWN GAPS in the frontend (xfail = the test describes the intended behaviour) ----------
@pytest.mark.xfail(strict=True, reason="csv_frontend docstring says weight_shape is computed for conv/gemm when "
                                       "blank, but the code leaves the layer with 0 FLOPs and no warning")
def test_blank_weight_shape_conv_is_computed_or_warned(tmp_path):
    G, s = stats(csv(tmp_path, "c,Conv,1x3x8x8,1x4x8x8,3x3,1x1,1,fp32,,\n"))
    assert s["c"].macs == 6912 or G.warnings


@pytest.mark.xfail(strict=True, reason="same gap for Gemm: blank weight_shape silently gives 0 FLOPs")
def test_blank_weight_shape_gemm_is_computed_or_warned(tmp_path):
    G, s = stats(csv(tmp_path, "f,Gemm,1x64,1x10,,,,fp32,,\n"))
    assert s["f"].macs == 640 or G.warnings


@pytest.mark.xfail(strict=True, reason="duplicate layer names silently collide on <name>__out / <name>__w tensors")
def test_duplicate_layer_names_rejected(tmp_path):
    with pytest.raises(ValueError):
        from_csv(csv(tmp_path, "a,Relu,1x4,1x4,,,,fp32,,\na,Relu,1x4,1x4,,,,fp32,,\n"))
