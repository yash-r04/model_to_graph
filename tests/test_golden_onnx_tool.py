"""Layer 6: independent cross-check against onnx-tool (Conv/Gemm MACs only; elementwise conventions differ).
Set ROOFTOOL_MODELS to a folder with extra .onnx files (resnet50.onnx, mobilenet_v2.onnx, ...)."""
import glob, os
import numpy as np
import pytest
from rooftool import analyze

onnx_tool = pytest.importorskip("onnx_tool")
HERE = os.path.dirname(__file__)
MODELS = glob.glob(os.path.join(HERE, "..", "examples", "*.onnx"))
extra = os.environ.get("ROOFTOOL_MODELS")
if extra:
    MODELS += glob.glob(os.path.join(extra, "*.onnx"))


def ot_macs(path, batch=1):
    g = onnx_tool.Model(path).graph
    name = g.input[0]
    shape = [d if isinstance(d, int) and d > 0 else batch for d in g.tensormap[name].shape]
    g.shape_infer({name: np.zeros(shape, dtype=np.float32)})
    g.profile(exclude_ops=set())
    return {n: (node.op_type, node.macs[0]) for n, node in g.nodemap.items()}


@pytest.mark.parametrize("path", MODELS, ids=lambda p: os.path.basename(p))
def test_conv_gemm_macs_match_onnx_tool(path, tmp_path):
    _, res = analyze(path, hw="HW-GPU-002", out=str(tmp_path), dynamic_dim=1)
    ref = ot_macs(path)
    bad = []
    for L in res["layers"]:
        if L["op"] in ("Conv", "Gemm") and L["name"] in ref and ref[L["name"]][1] > 0:
            r = ref[L["name"]][1]
            if abs(L["macs"] - r) / r > 0.001:
                bad.append((L["name"], L["macs"], r))
    assert not bad, f"{len(bad)} layers differ from onnx-tool, first: {bad[:3]}"
