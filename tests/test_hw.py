"""Layer 3: hardware catalogue loader and the repairs for the known xlsx problems."""
import pytest
from rooftool.pipeline import DEFAULT_XLSX, DEFAULT_OVR
from rooftool.hw import load_catalogue, resolve, HardwareSpec

CAT = load_catalogue(str(DEFAULT_XLSX), str(DEFAULT_OVR))
RAW = load_catalogue(str(DEFAULT_XLSX), None)           # without repairs


def test_every_row_loads_with_sane_numbers():
    assert len(CAT) >= 10
    for h in CAT.values():
        assert all(v > 0 for v in h.peak_gflops.values()), h.id
        assert h.bw_gbs is None or h.bw_gbs > 0, h.id


def test_t4_repaired_from_shifted_row():
    t4 = CAT["HW-GPU-001"]
    assert t4.peak_gflops["fp16"] == 65_000 and t4.peak_gflops["int8"] == 130_000 and t4.bw_gbs == 320
    assert any("misaligned" in w for w in RAW["HW-GPU-001"].warnings)     # raw sheet is detected as broken


def test_h100_and_a100_sxm_converted_to_dense():
    assert CAT["HW-GPU-004"].peak_gflops["bf16"] == pytest.approx(989_500)
    assert CAT["HW-GPU-004"].peak_gflops["fp32"] == 67_000                 # fp32 has no sparsity: untouched
    assert CAT["HW-GPU-003"].peak_gflops["bf16"] == CAT["HW-GPU-002"].peak_gflops["bf16"]   # SXM == PCIe dense


def test_a100_pcie_untouched():
    assert CAT["HW-GPU-002"].peak_gflops["bf16"] == 312_000


def test_unit_conversion_tflops_to_gflops():
    assert CAT["HW-GPU-002"].peak_gflops["fp32"] == 19_500


def test_peak_for_fallback_chain():
    h = HardwareSpec(id="x", name="x", peak_gflops={"fp16": 10.0})
    assert h.peak_for("fp16") == (10.0, "fp16")
    assert h.peak_for("fp32") == (10.0, "fp16")         # last-resort fallback (warned by roofline)
    assert h.peak_for("int4") == (10.0, "fp16")
    assert HardwareSpec(id="y", name="y").peak_for("fp32") == (None, None)


def test_resolve_by_id_and_by_name_and_errors():
    assert resolve("HW-GPU-002", str(DEFAULT_XLSX), str(DEFAULT_OVR)).id == "HW-GPU-002"
    assert resolve("T4", str(DEFAULT_XLSX), str(DEFAULT_OVR)).id == "HW-GPU-001"
    with pytest.raises(SystemExit):
        resolve("no-such-chip", str(DEFAULT_XLSX), str(DEFAULT_OVR))
    with pytest.raises(SystemExit):
        resolve("NVIDIA", str(DEFAULT_XLSX), str(DEFAULT_OVR))             # ambiguous


def test_sheet_guard_every_row_either_clean_or_overridden():
    """FAILS if someone edits the xlsx and a misaligned row has no repair. This is the sheet's regression test."""
    for hid, h in RAW.items():
        if any("misaligned" in w and "ignored" in w for w in CAT[hid].warnings):
            pytest.fail(f"{hid} is misaligned in the xlsx and has no entry in overrides.yaml")
