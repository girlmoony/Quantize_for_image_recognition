#量子化モデルと通常モデルを自動判定してONNX変換
import torch
import torch.nn as nn
import torch.quantization
import torch.onnx
from torch.quantization import convert
import types

def is_qat_model(model):
    """
    モデルがQAT（量子化-aware training）済みかどうかを自動判定
    """
    for module in model.modules():
        # FakeQuant層を持っていればQAT
        if hasattr(module, 'activation_post_process') or 'FakeQuantize' in str(type(module)):
            return True
    return False

def export_model_to_onnx(model, onnx_path, dummy_input_shape=(1, 3, 224, 224)):
    """
    モデルがQAT済みかfloatかを自動判定し、適切にONNX変換を行う

    Args:
        model: PyTorchモデル（ロード済）
        onnx_path: 保存先のONNXファイル名
        dummy_input_shape: ダミー入力の形状（例：224×224画像）
    """
    model.eval()
    dummy_input = torch.randn(dummy_input_shape)

    if is_qat_model(model):
        print("✅ QATモデルを検出しました。convert() を適用してONNX変換します。")
        model = convert(model)
    else:
        print("✅ 通常のfloatモデルとしてONNX変換します。")

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=13,
        do_constant_folding=True
    )
    print(f"🎉 ONNX変換完了：{onnx_path}")


from torchvision.models import efficientnet_b0

# モデル定義と読み込み
model = efficientnet_b0(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 288)

# 重みをロード（QATでもfloatでもOK）
model.load_state_dict(torch.load("model.pth", map_location="cpu"))

# 自動判定 → ONNX変換
export_model_to_onnx(model, "model_auto.onnx")

