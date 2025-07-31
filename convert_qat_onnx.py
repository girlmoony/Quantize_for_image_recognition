import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0
from torch.quantization import get_default_qat_qconfig, prepare_qat, convert

# === モデル再構成 ===
num_classes = 288
model = efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.qconfig = get_default_qat_qconfig('fbgemm')
model_prepared = prepare_qat(model)

# === 重み読み込み ===
state_dict = torch.load("qat_model_final.pth", map_location='cpu')
model_prepared.load_state_dict(state_dict)

# === 推論モードに切り替え（これが必要）===
model_prepared.eval()

# === FakeQuantとObserverを削除（INT8変換はしない）===
model_converted = convert(model_prepared, inplace=False)

# === ONNXエクスポート ===
model_converted.eval()  # 念のためもう一度eval（冪等）
dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model_converted,
    dummy_input,
    "qat_model_cleaned.onnx",
    export_params=True,
    opset_version=13,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("✅ QATモデル（FakeQuant削除済）をONNXへ変換完了")
