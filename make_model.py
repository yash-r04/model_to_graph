# make_models.py  —  exports standard models with random weights (no download of pretrained weights)
# Run once: python make_models.py
# Then:     python compare_onnx_tool.py resnet50.onnx 1
#           python -m rooftool analyze resnet50.onnx --hw HW-GPU-002 --out out\resnet50

import torch
import torchvision

x = torch.randn(1, 3, 224, 224)

models = [
    ("resnet50",      torchvision.models.resnet50),
    ("mobilenet_v2",  torchvision.models.mobilenet_v2),
    ("vit_b_16",      torchvision.models.vit_b_16),
]

for name, ctor in models:
    m = ctor(weights=None).eval()
    torch.onnx.export(
        m, x, f"{name}.onnx",
        opset_version=17,
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
    )
    print(f"saved {name}.onnx")