"""Compare rooftool per-layer MACs against onnx-tool (installed version checked: onnx-tool exposes
Model(path).graph.nodemap, shape_infer(), profile(), and each node has .macs = [macs, extra]).

Usage:  python compare_onnx_tool.py model.onnx [batch]
Writes: out/compare/comparison.csv
"""
import os
import sys

import numpy as np
import pandas as pd
import onnx_tool

from rooftool import analyze

model_path = sys.argv[1]
batch = int(sys.argv[2]) if len(sys.argv) > 2 else 1

# ---- 1. rooftool ----------------------------------------------------------
G, res = analyze(model_path, hw="HW-GPU-002", out="out/compare", dynamic_dim=batch)
ours = pd.DataFrame(res["layers"]).set_index("name")[["op", "macs", "flops"]]

# ---- 2. onnx-tool ---------------------------------------------------------
m = onnx_tool.Model(model_path)
g = m.graph
input_name = g.input[0]
shape = [d if isinstance(d, int) and d > 0 else batch for d in g.tensormap[input_name].shape]
g.shape_infer({input_name: np.zeros(shape, dtype=np.float32)})
g.profile(exclude_ops=set())    # empty set: do not hide any op from the profile

ot_rows = []
for name, node in g.nodemap.items():
    macs = node.macs[0] if getattr(node, "macs", None) else 0
    ot_rows.append({"name": name, "ot_op": node.op_type, "ot_macs": float(macs)})
ot = pd.DataFrame(ot_rows).set_index("name")

# ---- 3. join and compare --------------------------------------------------
cmp = ours.join(ot, how="outer")
missing_ours = cmp[cmp["op"].isna()].index.tolist()
missing_ot = cmp[cmp["ot_op"].isna()].index.tolist()
# onnx-tool zero-MAC helper nodes (Constant, Identity) are not interesting mismatches
missing_ours = [n for n in missing_ours if cmp.loc[n, "ot_macs"] > 0]
cmp["diff_%"] = 100 * (cmp["macs"] - cmp["ot_macs"]) / cmp["ot_macs"].replace(0, np.nan)

os.makedirs("out/compare", exist_ok=True)
cmp.to_csv("out/compare/comparison.csv")
pd.set_option("display.width", 200)
print(cmp.to_string(float_format=lambda v: f"{v:,.4g}"))

print()
if missing_ours:
    print("Only in onnx-tool (name mismatch or not counted by rooftool):", missing_ours)
if missing_ot:
    print("Only in rooftool (not in onnx-tool profile):", missing_ot)
# Only MAC-based ops are a fair comparison. Elementwise/Pool/Norm/Softmax count ops differently
# (rooftool = FLOPs per element; onnx-tool = a different per-element convention), so they are listed separately.
MAC_OPS = ["Conv", "Gemm"]
conv_gemm = cmp[cmp["op"].isin(MAC_OPS) & (cmp["ot_macs"] > 0)]
bad = conv_gemm[conv_gemm["diff_%"].abs() > 3]
print(f"Conv/Gemm layers compared: {len(conv_gemm)}, off by more than 3%: {len(bad)}")
other = cmp[~cmp["op"].isin(MAC_OPS) & cmp["op"].notna()]
print("Non-MAC ops (convention differences, not compared):", other.index.tolist())
print(f"Total MACs  rooftool={cmp['macs'].sum():,.0f}  onnx-tool={cmp['ot_macs'].sum():,.0f}")
