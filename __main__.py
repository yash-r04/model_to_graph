import argparse, sys
from .pipeline import analyze, DEFAULT_XLSX, DEFAULT_OVR


def main():
    p = argparse.ArgumentParser(prog="rooftool", description="Model -> DAG + layer-wise roofline")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="analyze a model")
    a.add_argument("model"); a.add_argument("--hw", help="catalogue ID/name (e.g. HW-GPU-002, 'A100 80GB PCIe') or a .yaml")
    a.add_argument("--peak-gflops", type=float); a.add_argument("--bw", type=float, help="GB/s (also overrides the catalogue)")
    a.add_argument("--out", default="out"); a.add_argument("--xlsx"); a.add_argument("--overrides")
    a.add_argument("--dynamic-dim", type=int, default=1, help="value for dynamic dims e.g. batch (default 1)")
    h = sub.add_parser("hw-list", help="show hardware after repair + warnings")
    h.add_argument("--xlsx", default=str(DEFAULT_XLSX)); h.add_argument("--overrides", default=str(DEFAULT_OVR))
    args = p.parse_args()
    if args.cmd == "hw-list":
        from .hw import load_catalogue
        for h in load_catalogue(args.xlsx, args.overrides).values():
            pk = ", ".join(f"{k}={v/1000:g}T" for k, v in h.peak_gflops.items()) or "no peaks"
            print(f"{h.id:12} {h.name[:38]:38} BW={h.bw_gbs} GB/s | {pk}")
            for w in h.warnings: print(f"{'':12}   ! {w}")
        return
    G, r = analyze(args.model, args.hw, args.out, args.xlsx, args.overrides, args.bw, args.peak_gflops, args.dynamic_dim)
    s = r["summary"]
    print(f"{s['n_layers']} layers, {s['total_flops']/1e9:.3f} GFLOPs, memory-bound {s['n_memory_bound']}, "
          f"compute-bound {s['n_compute_bound']}, latency LB {s['latency_ms_lower_bound']} ms")
    for w in r["warnings"]: print("WARN:", w)
    print(f"-> open {args.out}/index.html")


main()
