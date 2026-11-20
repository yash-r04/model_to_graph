from __future__ import annotations
import json, shutil, subprocess
from pathlib import Path
import pandas as pd

COL = {"memory": "#E8743B", "compute": "#2E86AB", "data-movement": "#9B9B9B", "free": "#DDDDDD", None: "#CCCCCC"}


def write_json(G, res, path):
    graph = dict(nodes=[dict(name=n.name, op=n.op_type, orig_op=n.orig_op, inputs=n.inputs, outputs=n.outputs)
                        for n in G.nodes],
                 edges=[dict(src=u, dst=v, tensor=d["tensor"]) for u, v, d in G.to_networkx().edges(data=True)],
                 tensors={k: dict(shape=t.shape, dtype=t.dtype, is_weight=t.is_weight) for k, t in G.tensors.items()},
                 inputs=G.inputs, outputs=G.outputs)
    json.dump(dict(model=G.name, graph=graph, **res), open(path, "w"), indent=2, default=float)


def write_csv(res, path):
    pd.DataFrame(res["layers"]).to_csv(path, index=False)


def write_dag(G, res, out: Path):
    by = {l["name"]: l for l in res["layers"]}
    maxf = max([l["flops"] for l in res["layers"]] + [1])
    lines = ["digraph G {", "rankdir=TB; node [shape=box, style=filled, fontname=Helvetica, fontsize=10];"]
    for n in G.nodes:
        l = by[n.name]
        lab = f"{n.name}\\n{n.orig_op or n.op_type}\\n{l['flops']/1e6:.1f} MFLOPs"
        if l["ai_ideal"] is not None: lab += f"\\nAI={l['ai_ideal']:.1f}"
        pw = 1 + 3 * (l["flops"] / maxf)
        lines.append(f'"{n.name}" [label="{lab}", fillcolor="{COL.get(l["bound"])}", penwidth={pw:.1f}, fontcolor=white];')
    for u, v, d in G.to_networkx().edges(data=True):
        lines.append(f'"{u}" -> "{v}";')
    lines.append("}")
    (out / "dag.dot").write_text("\n".join(lines))
    if shutil.which("dot"):
        subprocess.run(["dot", "-Tsvg", str(out / "dag.dot"), "-o", str(out / "dag.svg")], check=False)


def write_roofline(res, path):
    import plotly.graph_objects as go
    import numpy as np
    hw = res["hw"]; bw = hw["bw_gbs"]
    L = [l for l in res["layers"] if l["ai_ideal"] and l["attainable_gflops"]]
    if not bw or not L:
        Path(path).write_text("<html><body>No roofline: bandwidth or peak unknown.</body></html>"); return
    xs = [l["ai_ideal"] for l in L] + [l["ai_nocache"] for l in L]
    x = np.logspace(np.floor(np.log10(min(xs))) - 0.5, np.ceil(np.log10(max(xs))) + 0.5, 200)
    fig = go.Figure()
    for d in sorted({l["peak_dtype_used"] for l in L if l["peak_dtype_used"]}):
        peak = hw["peak_gflops"].get(d)
        if peak is None: continue
        fig.add_trace(go.Scatter(x=x, y=np.minimum(peak, x * bw), mode="lines", name=f"roof {d} ({peak/1000:g} T)",
                                 line=dict(width=2)))
    maxf = max(l["flops"] for l in L)
    for bound, col in (("memory", COL["memory"]), ("compute", COL["compute"])):
        S = [l for l in L if l["bound"] == bound]
        if not S: continue
        fig.add_trace(go.Scatter(
            x=[l["ai_ideal"] for l in S], y=[l["attainable_gflops"] for l in S], mode="markers", name=f"{bound}-bound",
            marker=dict(color=col, size=[8 + 30 * (l["flops"] / maxf) ** 0.5 for l in S], opacity=0.8,
                        line=dict(width=1, color="white")),
            text=[f"{l['name']} ({l['op']}, {l['dtype']})<br>FLOPs={l['flops']:.3g}<br>AI={l['ai_ideal']:.2f}"
                  f"<br>bytes={l['bytes_ideal']:.3g}" for l in S], hoverinfo="text"))
    fig.add_trace(go.Scatter(
        x=[l["ai_nocache"] for l in L], y=[min(hw["peak_gflops"].get(l["peak_dtype_used"], 1e12), l["ai_nocache"] * bw) for l in L],
        mode="markers", name="no-reuse worst case", visible="legendonly",
        marker=dict(symbol="x", color="#555", size=7), text=[l["name"] for l in L], hoverinfo="text"))
    fig.update_layout(title=f"Roofline - {hw['name']} (BW {bw:g} GB/s)", template="plotly_white",
                      xaxis=dict(type="log", title="Arithmetic intensity (FLOP/byte)"),
                      yaxis=dict(type="log", title="Attainable GFLOP/s"), height=650)
    fig.write_html(path, include_plotlyjs="cdn")


def write_report(G, res, out: Path):
    """One self-contained index.html: summary + warnings + embedded DAG + roofline link + table"""
    s = res["summary"]; hw = res["hw"]
    df = pd.DataFrame(res["layers"])[["name", "op", "dtype", "flops", "bytes_ideal", "ai_ideal", "bound",
                                      "attainable_gflops", "latency_us_lb", "fits_onchip"]]
    svg = (out / "dag.svg").read_text() if (out / "dag.svg").exists() else "<i>install graphviz for DAG</i>"
    roof = (out / "roofline.html").read_text() if (out / "roofline.html").exists() else ""
    import html
    warns = "".join(f"<li>{html.escape(w)}</li>" for w in res["warnings"]) or "<li>none</li>"
    page = f"""<html><head><meta charset='utf-8'><title>rooftool report</title>
<style>body{{font-family:Helvetica,Arial;margin:24px}} td,th{{border:1px solid #ccc;padding:3px 8px;font-size:12px}}
table{{border-collapse:collapse}} .w{{background:#fff4e5;padding:8px}}</style></head><body>
<h2>{html.escape(G.name)} on {html.escape(hw['name'])}</h2>
<p>{s['n_layers']} layers | {s['total_flops']/1e9:.2f} GFLOPs | params {s['params_bytes']/1e6:.2f} MB |
memory-bound {s['n_memory_bound']} / compute-bound {s['n_compute_bound']} |
latency lower bound {s['latency_ms_lower_bound'] if s['latency_ms_lower_bound'] is None else round(s['latency_ms_lower_bound'],3)} ms</p>
<div class='w'><b>Warnings</b><ul>{warns}</ul></div>
<h3>Roofline</h3><iframe srcdoc="{html.escape(roof)}" width='100%' height='700' style='border:0'></iframe>
<h3>DAG</h3><div style='overflow:auto;max-height:900px'>{svg}</div>
<h3>Per-layer table</h3>{df.to_html(index=False, float_format=lambda v: f'{v:.3g}')}
</body></html>"""
    (out / "index.html").write_text(page)
