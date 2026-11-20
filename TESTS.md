# Test suite

Run all tests:
    pip install pytest onnx-tool
    pytest -q

## Files

### tests/helpers.py
Shared utilities for building tiny ONNX models from scratch. No test assertions here.

### tests/test_ops.py  — Layer 1: op formulas
Every assertion is a number worked out by hand, independent of the code.
- Conv plain (no bias)
- Conv with bias: adds one MAC per output element
- Depthwise conv (group == channels)
- Grouped conv
- Strided conv on 224x224 input (matches the known ResNet conv1 figure: 118,013,952)
- Gemm without bias / with bias
- Attention MatMul (activation × activation, no weights)
- Flatten: zero bytes, zero FLOPs
- Transpose: zero FLOPs, full data movement
- ReLU6 / Clip: two ops per element (onnx-tool convention)
- Unsupported op: zero FLOPs + a warning

### tests/test_roofline.py  — Layer 2: roofline math
Hardware is hand-chosen so the expected bound is obvious (ridge = 1000/50 = 20 FLOP/byte).
- Large weights → memory-bound; attainable = AI × BW
- Large square MatMul → compute-bound; attainable = peak; latency = compute time
- Ridge point value
- Shape op → "free" or "data-movement", not memory/compute
- Missing BW → warns, does not crash
- Missing dtype peak → falls back to fp16 and warns "optimistic"
- Summary totals = sum of layer values

### tests/test_hw.py  — Layer 3: hardware catalogue
- Every row loads with positive numbers after repairs
- T4 row: repaired from xlsx shift (fp16=65T, int8=130T, BW=320)
- H100: bf16 halved from sparse to dense (989.5T)
- A100 SXM: equals A100 PCIe after sparsity repair (both bf16=312T)
- A100 PCIe: untouched (19.5T fp32)
- TFLOPS → GFLOPS conversion (19.5T → 19500G)
- Fallback chain (fp32 model on fp16-only hardware)
- Resolve by exact ID, by partial name, error on unknown, error on ambiguous
- Sheet regression guard: fails if the xlsx has a misaligned row with no repair in overrides.yaml

### tests/test_frontends.py  — Layer 4: format readers
- Dynamic batch scales FLOPs × 8 linearly
- DAG edges include skip connections
- Weights marked as is_weight; activations are not; graph.inputs correct
- CSV → same FLOPs as equivalent ONNX model
- CSV inputs column creates correct DAG branches
- Unknown file extension raises ValueError
- TFLite and CSV sample tests live in their own files (below)

### tests/test_pipeline.py  — Layer 5: end-to-end contract
- All five output files are created: result.json, layers.csv, dag.dot, roofline.html, index.html
- result.json has the keys teammates depend on: graph, layers, summary, hw, warnings
- Every layer dict has all required keys (name, op, flops, macs, bound, ai_ideal, ...)
- DAG is a valid directed acyclic graph (networkx check)
- Two-number mode (--peak-gflops / --bw) needs no catalogue

### tests/test_golden_onnx_tool.py  — Layer 6: cross-check vs onnx-tool
Runs onnx-tool independently and compares Conv/Gemm MACs per layer.
- Runs automatically on every .onnx in examples/ (tiny_resnet, bias_dw_test)
- Extend by setting ROOFTOOL_MODELS to a folder with more models:
    set ROOFTOOL_MODELS=C:\edge-AI\models
    pytest tests\test_golden_onnx_tool.py -q
- Tolerance: 0.1% per layer. Elementwise/Pool/Norm are excluded (different conventions).

### tests/test_tflite.py  — Layer 4b: TFLite frontend
Runs on examples/tiny_cnn.tflite (fp32) and examples/tiny_cnn_int8.tflite, built by examples/make_test_tflite.py (no TensorFlow needed).
- Op order, canonical op names, graph inputs/outputs, DAG is a linear chain
- Weights vs activations flagged correctly (incl. the RESHAPE shape constant)
- Hand-computed MACs: conv+bias 7168, depthwise 2304 (Cout is the last axis), FC+bias 2570; relu 256, softmax 50 FLOPs; reshape free
- Weight bytes fp32; int8 model: same FLOPs, 4x fewer activation bytes, int32 biases read from the file
- Dynamic batch (-1 in file) scales FLOPs x8 with dynamic_dim=8
- Committed .tflite files are byte-identical to what the generator produces
- Corrupt file raises; end-to-end outputs written
Limitation: the samples carry no builtin-options (padding/strides), so they are for the frontend, not the TFLite interpreter.

### tests/test_csv.py  — Layer 4c: CSV frontend
Runs on examples/tiny_cnn.csv (same model as the TFLite sample, no bias column) and examples/model_template.csv.
- Hand-computed MACs/bytes, default "previous row" wiring, `inputs` branches, graph input/output
- CSV total MACs = TFLite total MACs minus the 256 + 10 bias terms
- dtype column (int8 vs fp32, blank = fp32), op aliases/case, shape separators, blank layer names, pool kernel, unknown op warning, extra columns
- Bad input fails loudly: unknown/forward layer reference, missing column
- 3 xfail (strict) tests record known gaps: blank weight_shape on Conv/Gemm gives 0 FLOPs silently (docstring says it is computed), duplicate layer names collide. When you fix them the tests flip to XPASS and fail, which is the cue to delete the xfail mark.

## Extending the golden test with larger models
    python make_models.py                      # exports resnet50.onnx, mobilenet_v2.onnx, vit_b_16.onnx
    set ROOFTOOL_MODELS=.
    pytest tests\test_golden_onnx_tool.py -v
