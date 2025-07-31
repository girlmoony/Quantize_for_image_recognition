import onnxruntime as ort
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# === 1. パラメータ設定 ===
onnx_model_path = "model_cleaned_sim.onnx"  # 変換済ONNXファイル
val_dir = "val_path"                        # 検証データフォルダ（ImageFolder構造）
batch_size = 32
num_classes = 288
input_size = (224, 224)

# === 2. ONNX Runtimeセッション作成 ===
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# === 3. データローダ（PyTorchと同じTransform）===
transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    # ONNXで前処理の正規化を学習時にしていた場合はここに入れる
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225]),
])
val_dataset = datasets.ImageFolder(val_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# === 4. 精度評価ループ ===
total_correct = 0
total_samples = 0

for images, labels in val_loader:
    # PyTorch tensor → numpy に変換
    images_np = images.numpy()

    # ONNX推論
    outputs = session.run([output_name], {input_name: images_np})[0]
    preds = np.argmax(outputs, axis=1)

    # 精度計算
    total_correct += (preds == labels.numpy()).sum()
    total_samples += labels.size(0)

accuracy = total_correct / total_samples
print(f"✅ ONNXモデル精度: {accuracy:.4f}")
