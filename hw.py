"""Hardware spec loading. Sources: (a) the PE FW catalogue xlsx, (b) a tiny YAML, (c) two CLI numbers."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import yaml

PEAK_COLS = {"FP64 Peak": "fp64", "FP32 Peak": "fp32", "TF32 Peak": "tf32", "BF16 Peak": "bf16",
             "FP16 Peak": "fp16", "FP8 Peak": "fp8", "INT8 Peak": "int8", "INT4 Peak": "int4"}
# if the model dtype has no peak, fall back through this chain (and report it)
FALLBACK = {"fp32": ["fp32", "tf32", "fp16", "bf16"], "fp16": ["fp16", "bf16", "fp32"], "bf16": ["bf16", "fp16", "fp32"],
            "int8": ["int8", "fp8", "fp16", "fp32"], "uint8": ["int8", "fp8", "fp16", "fp32"],
            "int4": ["int4", "int8", "fp16"], "fp8": ["fp8", "int8", "fp16"],
            "int32": ["fp32"], "int64": ["fp32"], "int16": ["fp16", "fp32"], "fp64": ["fp64", "fp32"],
            "tf32": ["tf32", "fp32"], "bool": ["int8", "fp32"]}


@dataclass
class HardwareSpec:
    id: str
    name: str
    category: str = ""
    peak_gflops: Dict[str, float] = field(default_factory=dict)   # dtype -> GFLOPS/GOPS (dense)
    bw_gbs: Optional[float] = None
    mem_gb: Optional[float] = None
    onchip_mb: Optional[float] = None
    tdp_w: Optional[float] = None
    warnings: List[str] = field(default_factory=list)

    def peak_for(self, dtype: str):
        """returns (peak_gflops, dtype_used) or (None, None)"""
        for d in FALLBACK.get(dtype, [dtype, "fp32"]):
            if d in self.peak_gflops:
                return self.peak_gflops[d], d
        return None, None


def _num(v):
    if v is None: return None
    try:
        return float(str(v).replace(",", "").strip())
    except ValueError:
        return None


_SUSPECT = re.compile(r"[A-Za-z]{3,}")


def load_catalogue(xlsx: str, overrides: Optional[str] = None) -> Dict[str, HardwareSpec]:
    import openpyxl
    ov = yaml.safe_load(open(overrides)) if overrides and Path(overrides).exists() else {}
    wb = openpyxl.load_workbook(xlsx, data_only=True, read_only=True)
    rows = list(wb["Hardware Catalogue"].iter_rows(values_only=True))
    hdr = [str(h).strip() if h else "" for h in rows[0]]
    out: Dict[str, HardwareSpec] = {}
    for r in rows[1:]:
        d = dict(zip(hdr, r))
        hid = d.get("Hardware ID")
        if not hid: continue
        v, m = d.get("Vendor") or "", d.get("Model / SKU") or ""
        h = HardwareSpec(id=hid, name=(m if m.lower().startswith(v.lower()) else f"{v} {m}").strip(),
                         category=d.get("Hardware Category") or "")
        # shifted-row detection: text sitting in a numeric-only column
        bad = [c for c in list(PEAK_COLS) + ["Memory Capacity (GB)", "Memory Bandwidth (GB/s)", "Cache (MB)",
               "TDP / Board Power (W)"] if d.get(c) is not None and _num(d.get(c)) is None]
        o = ov.get(hid, {})
        if bad and "set" not in o:
            h.warnings.append(f"row looks misaligned (text in numeric columns {bad}); numeric fields ignored")
        elif bad:
            h.warnings.append(f"row was misaligned in the xlsx; repaired from overrides ({o.get('reason','')})")
        if not bad:
            for c, k in PEAK_COLS.items():
                v = _num(d.get(c))
                if v is not None: h.peak_gflops[k] = v * 1000.0        # TFLOPS/TOPS -> GFLOPS/GOPS
            h.mem_gb = _num(d.get("Memory Capacity (GB)"))
            h.bw_gbs = _num(d.get("Memory Bandwidth (GB/s)"))
            h.onchip_mb = _num(d.get("Cache (MB)"))
            h.tdp_w = _num(d.get("TDP / Board Power (W)"))
        # overrides
        for src, dst in (o.get("move") or {}).items():
            if not bad and src in h.peak_gflops: h.peak_gflops[dst] = h.peak_gflops.pop(src)
        for k, v in (o.get("set") or {}).items():
            if k in PEAK_COLS.values(): h.peak_gflops[k] = float(v) * 1000.0
            elif k == "bw_gbs": h.bw_gbs = float(v)
            elif k == "mem_gb": h.mem_gb = float(v)
            elif k == "onchip_mb": h.onchip_mb = float(v)
            elif k == "tdp_w": h.tdp_w = float(v)
        for k, f in (o.get("dense_factor") or {}).items():
            if k in h.peak_gflops: h.peak_gflops[k] *= f
        if o.get("warn"): h.warnings.append(o["warn"])
        out[hid] = h
    return out


def custom_spec(peak_gflops: float, bw_gbs: float, name="custom", dtype="fp32") -> HardwareSpec:
    """The 'two numbers' mode: one peak applies to every dtype."""
    return HardwareSpec(id="custom", name=name, bw_gbs=bw_gbs,
                        peak_gflops={d: peak_gflops for d in FALLBACK})


def load_yaml_spec(path: str) -> HardwareSpec:
    y = yaml.safe_load(open(path))
    pk = {k: float(v) for k, v in y.get("peak_gflops", {}).items()}
    return HardwareSpec(id=y.get("id", Path(path).stem), name=y.get("name", Path(path).stem),
                        peak_gflops=pk, bw_gbs=y.get("bw_gbs"), mem_gb=y.get("mem_gb"),
                        onchip_mb=y.get("onchip_mb"), tdp_w=y.get("tdp_w"))


def resolve(hw: str, xlsx: Optional[str], overrides: Optional[str]) -> HardwareSpec:
    if hw.endswith((".yaml", ".yml")):
        return load_yaml_spec(hw)
    cat = load_catalogue(xlsx, overrides)
    if hw in cat: return cat[hw]
    m = [h for h in cat.values() if hw.lower() in h.name.lower() or hw.lower() == h.id.lower()]
    if len(m) == 1: return m[0]
    raise SystemExit(f"hardware '{hw}' not found or ambiguous ({[h.id for h in m]}). Try `rooftool hw-list`.")
