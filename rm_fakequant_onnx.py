import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torch.quantization import prepare_qat, get_default_qat_qconfig

# === 1. モデル構築・QAT設定 ===
model = efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 288)
model.qconfig = get_default_qat_qconfig('fbgemm')
model_prepared = prepare_qat(model)

# === 2. QAT済み重み読み込み ===
model_prepared.load_state_dict(torch.load("qat_model_final.pth", map_location="cpu"))
model_prepared.eval()

# === 3. convert() を使わずに FakeQuant だけ除去（floatモデルとして残す）
# → "擬似量子化された float モデル" が得られる（ONNX対応）

def remove_fakequant(model):
    for name, module in model.named_children():
        if hasattr(module, "activation_post_process"):
            module.activation_post_process = nn.Identity()
        remove_fakequant(module)  # 再帰的に
    return model

model_cleaned = remove_fakequant(model_prepared)

# === FakeQuant層だけ削除（安全版） ===
def remove_fakequant_safe(module):
    for name, child in module.named_children():
        if isinstance(child, (FakeQuantize, FusedMovingAvgObsFakeQuantize)):
            setattr(module, name, nn.Identity())
        else:
            remove_fakequant_safe(child)
    return module

model_cleaned = remove_fakequant_safe(model_prepared)
# === 4. ONNXエクスポート ===
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model_cleaned,
    dummy_input,
    "qat_cleaned_float.onnx",
    opset_version=13,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
