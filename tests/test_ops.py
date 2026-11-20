"""Layer 1: op formulas against numbers worked out BY HAND (not copied from the code)."""
import pytest
from onnx import helper as h
from rooftool.frontends.onnx_frontend import from_onnx
from rooftool.ops import analyze_node
from helpers import W, build_onnx


def stats(tmp_path, nodes, inits, inputs, outputs):
    p = build_onnx(tmp_path / "m.onnx", nodes, inits, inputs, outputs)
    G = from_onnx(p)
    return G, [analyze_node(G, n) for n in G.nodes]


def test_conv_plain(tmp_path):
    # out 1x4x8x8 = 256 elems; each needs Cin*kh*kw = 27 MACs -> 6912
    _, s = stats(tmp_path, [h.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])],
                 [W("w", 4, 3, 3, 3)], [("x", [1, 3, 8, 8])], [("y", [1, 4, 8, 8])])
    assert s[0].macs == 6912
    assert s[0].flops == 2 * 6912
    assert s[0].weight_bytes == 4 * 3 * 3 * 3 * 4          # fp32


def test_conv_bias_adds_one_mac_per_output(tmp_path):
    _, s = stats(tmp_path, [h.make_node("Conv", ["x", "w", "b"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])],
                 [W("w", 4, 3, 3, 3), W("b", 4)], [("x", [1, 3, 8, 8])], [("y", [1, 4, 8, 8])])
    assert s[0].macs == 6912 + 256


def test_depthwise_conv(tmp_path):
    # group == channels: weight 4x1x3x3, each output needs only 9 MACs -> 256*9
    _, s = stats(tmp_path, [h.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], group=4)],
                 [W("w", 4, 1, 3, 3)], [("x", [1, 4, 8, 8])], [("y", [1, 4, 8, 8])])
    assert s[0].macs == 256 * 9


def test_grouped_conv(tmp_path):
    # Cin=4,Cout=8,group=2 -> weight 8x2x3x3; per output: 2*9=18 MACs; outputs 8*8*8=512
    _, s = stats(tmp_path, [h.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], group=2)],
                 [W("w", 8, 2, 3, 3)], [("x", [1, 4, 8, 8])], [("y", [1, 8, 8, 8])])
    assert s[0].macs == 512 * 18


def test_strided_conv_uses_output_size(tmp_path):
    # 224 -> 112 with stride 2: output 64*112*112, 3*7*7=147 MACs each
    _, s = stats(tmp_path, [h.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[7, 7], strides=[2, 2], pads=[3, 3, 3, 3])],
                 [W("w", 64, 3, 7, 7)], [("x", [1, 3, 224, 224])], [("y", [1, 64, 112, 112])])
    assert s[0].macs == 64 * 112 * 112 * 147 == 118_013_952


def test_gemm_with_and_without_bias(tmp_path):
    _, s = stats(tmp_path, [h.make_node("Gemm", ["x", "w"], ["y"])],
                 [W("w", 64, 10)], [("x", [1, 64])], [("y", [1, 10])])
    assert s[0].macs == 640
    _, s = stats(tmp_path, [h.make_node("Gemm", ["x", "w", "b"], ["y"])],
                 [W("w", 64, 10), W("b", 10)], [("x", [1, 64])], [("y", [1, 10])])
    assert s[0].macs == 650


def test_matmul_activation_x_activation_has_no_weights(tmp_path):
    # attention scores: (1,12,16,64) x (1,12,64,16) -> (1,12,16,16); K=64
    _, s = stats(tmp_path, [h.make_node("MatMul", ["a", "b"], ["y"])], [],
                 [("a", [1, 12, 16, 64]), ("b", [1, 12, 64, 16])], [("y", [1, 12, 16, 16])])
    assert s[0].macs == 12 * 16 * 16 * 64
    assert s[0].weight_bytes == 0
    assert s[0].act_in_bytes == (12 * 16 * 64 * 2) * 4


def test_shape_ops_move_no_bytes(tmp_path):
    _, s = stats(tmp_path, [h.make_node("Flatten", ["x"], ["y"], axis=1)], [],
                 [("x", [1, 8, 4, 4])], [("y", [1, 128])])
    assert (s[0].flops, s[0].act_in_bytes, s[0].out_bytes) == (0, 0, 0)


def test_transpose_is_pure_data_movement(tmp_path):
    _, s = stats(tmp_path, [h.make_node("Transpose", ["x"], ["y"], perm=[0, 2, 1])], [],
                 [("x", [1, 8, 4])], [("y", [1, 4, 8])])
    assert s[0].flops == 0 and s[0].out_bytes == 32 * 4 and s[0].act_in_bytes == 32 * 4


def test_relu6_clip_counts_two_ops_per_element(tmp_path):
    from onnx import numpy_helper as nh
    import numpy as np
    inits = [nh.from_array(np.array(0, dtype="float32"), "mn"), nh.from_array(np.array(6, dtype="float32"), "mx")]
    _, s = stats(tmp_path, [h.make_node("Clip", ["x", "mn", "mx"], ["y"])], inits,
                 [("x", [1, 4, 4, 4])], [("y", [1, 4, 4, 4])])
    assert s[0].flops == 2 * 64


def test_unsupported_op_warns_and_counts_zero(tmp_path):
    G, s = stats(tmp_path, [h.make_node("Einsum", ["x"], ["y"], equation="ij->ji")], [],
                 [("x", [4, 4])], [("y", [4, 4])])
    assert s[0].flops == 0
    assert any("unsupported" in w for w in G.warnings)
