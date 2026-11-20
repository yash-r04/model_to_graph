from __future__ import annotations
from .ir import Graph
from .ops import analyze_node
from .hw import HardwareSpec


def _layer_dtype(G, n):
    for t in n.inputs:
        tt = G.tensors.get(t)
        if tt is not None and not tt.is_weight:
            return tt.dtype
    return G.tensors[n.outputs[0]].dtype if n.outputs else "fp32"


def run(G: Graph, hw: HardwareSpec, bw_override=None):
    bw = bw_override or hw.bw_gbs
    warnings = list(G.warnings) + [f"[{hw.id}] {w}" for w in hw.warnings]
    if not bw:
        warnings.append(f"[{hw.id}] memory bandwidth unknown -> roofline bound not computed. Pass --bw.")
    layers = []
    onchip_b = hw.onchip_mb * 1e6 if hw.onchip_mb else None
    for n in G.nodes:
        s = analyze_node(G, n)
        dt = _layer_dtype(G, n)
        peak, used = hw.peak_for(dt)
        b_ideal = s.weight_bytes + s.act_in_bytes + s.out_bytes
        L = dict(name=n.name, op=n.op_type, orig_op=n.orig_op, dtype=dt, peak_dtype_used=used,
                 macs=s.macs, flops=s.flops, weight_bytes=s.weight_bytes, in_bytes=s.act_in_bytes,
                 out_bytes=s.out_bytes, bytes_ideal=b_ideal, bytes_nocache=s.nocache_bytes,
                 ai_ideal=(s.flops / b_ideal) if b_ideal else None,
                 ai_nocache=(s.flops / s.nocache_bytes) if s.nocache_bytes else None,
                 note=s.note, bound=None, attainable_gflops=None, ridge=None, latency_us_lb=None,
                 fits_onchip=(b_ideal <= onchip_b) if onchip_b else None)
        if peak is None:
            warnings.append(f"[{hw.id}] no peak for dtype '{dt}' (e.g. layer {n.name}); layer not classified")
        if peak and used != dt:
            warnings.append(f"[{hw.id}] model dtype {dt} has no peak in the spec; using {used} peak instead "
                            f"(optimistic if {used} is lower precision)")
        if peak and bw:
            ridge = peak / bw
            L["ridge"] = ridge
            if s.flops == 0:
                L["bound"] = "data-movement" if b_ideal else "free"
                L["attainable_gflops"] = 0.0
            else:
                L["attainable_gflops"] = min(peak, L["ai_ideal"] * bw)
                L["bound"] = "memory" if L["ai_ideal"] < ridge else "compute"
            # lower-bound latency: limited by whichever of compute / memory takes longer
            L["latency_us_lb"] = max(s.flops / (peak * 1e9), b_ideal / (bw * 1e9)) * 1e6
        layers.append(L)

    tot_f = sum(l["flops"] for l in layers)
    tot_b = sum(l["bytes_ideal"] for l in layers)
    lat = [l["latency_us_lb"] for l in layers if l["latency_us_lb"] is not None]
    summary = dict(total_flops=tot_f, total_macs=sum(l["macs"] for l in layers),
                   total_bytes_ideal=tot_b, params_bytes=sum(l["weight_bytes"] for l in layers),
                   model_ai=(tot_f / tot_b if tot_b else None),
                   n_layers=len(layers),
                   n_memory_bound=sum(l["bound"] == "memory" for l in layers),
                   n_compute_bound=sum(l["bound"] == "compute" for l in layers),
                   latency_ms_lower_bound=(sum(lat) / 1e3 if lat else None),
                   unsupported_ops=sorted({l["orig_op"] for l in layers if l["note"] == "UNSUPPORTED"}))
    hw_d = dict(id=hw.id, name=hw.name, bw_gbs=bw, mem_gb=hw.mem_gb, onchip_mb=hw.onchip_mb,
                tdp_w=hw.tdp_w, peak_gflops=hw.peak_gflops)
    return dict(layers=layers, summary=summary, hw=hw_d, warnings=sorted(set(warnings)))
