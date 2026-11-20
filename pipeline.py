from __future__ import annotations
from pathlib import Path
from .frontends import load_graph
from .hw import resolve, custom_spec
from . import roofline, render


def _find_root():
    for parent in Path(__file__).resolve().parents:
        if (parent / "hw" / "PE_FW_Hardware_Specification_Catalogue_v1.xlsx").exists():
            return parent
    raise FileNotFoundError(
        "Cannot find hw/PE_FW_Hardware_Specification_Catalogue_v1.xlsx. "
        "Make sure the hw/ folder is next to the rooftool package, "
        "or pass --xlsx explicitly.")


HERE = _find_root()                      # ← was: Path(__file__).resolve().parent.parent
DEFAULT_XLSX = HERE / "hw" / "PE_FW_Hardware_Specification_Catalogue_v1.xlsx"
DEFAULT_OVR  = HERE / "hw" / "overrides.yaml"

def analyze(model, hw=None, out="out", xlsx=None, overrides=None, bw=None, peak_gflops=None,
            dynamic_dim=1, dtype=None):
    """The one-call API. hw can be a catalogue ID/name, a .yaml file, or None (then give peak_gflops+bw)."""
    xlsx = xlsx or str(DEFAULT_XLSX); overrides = overrides or str(DEFAULT_OVR)
    if hw:
        spec = resolve(hw, xlsx, overrides)
    elif peak_gflops and bw:
        spec = custom_spec(peak_gflops, bw)
    else:
        raise SystemExit("give --hw <id|name|file.yaml>  or  --peak-gflops and --bw")
    G = load_graph(model, dynamic_dim)
    res = roofline.run(G, spec, bw_override=bw)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    render.write_json(G, res, out / "result.json")
    render.write_csv(res, out / "layers.csv")
    render.write_dag(G, res, out)
    render.write_roofline(res, out / "roofline.html")
    render.write_report(G, res, out)
    return G, res