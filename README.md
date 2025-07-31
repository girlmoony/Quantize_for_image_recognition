EfficientNet-B0を使ったImagenetファインチューニング後、
GPU上では非常に高精度なのに、
# ONNX→bin変換（量子化含む）後のiPro環境では精度が大きく劣化している問題について

## 考えられる原因
### 1. 量子化による精度劣化（特にUnfreezeした層）
   - EfficientNet-B0は軽量かつ高精度ですが、量子化（特にINT8/FIX16など）により精度が落ちやすい構造です。
   - 微調整した深い層（低レイヤ）やBatchNormのパラメータが、量子化によって不正確になり、推論時にズレを生じる可能性があります
   - 学習はfloat32で行われているが、ONNX変換で量子化される際にそれらが不安定化しやすい

### 2. 量子化-awareでの学習をしていない
   - PyTorchで通常のfine-tuning後、ONNX→OpenVINO bin形式にすると **Post Training Quantization (PTQ)** が行われますが、これは精度を保証しません。
   - 量子化に不適な活性化分布を持つレイヤーや小さなデータセットでの新規クラスがあると、量子化精度が落ちやすい。

## 推奨される対応策
### 1. 量子化-aware Training (QAT)を導入
   - PyTorchで量子化-aware training（QAT）を使ってからONNXに変換してください。
   - QATでは量子化のノイズを学習中にシミュレートするため、binでの精度が大きく向上します。
```
# PyTorch QAT準備（簡略例）
from torch.quantization import prepare_qat, convert

model.train()
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
model_fused = torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']])  # 必要に応じて
model_prepared = prepare_qat(model_fused)

# 通常通りfine-tuning
# ...
# その後
model_quantized = convert(model_prepared.eval())
```
```
 QAT（Quantization‑Aware Training）
PyTorch 学習中に「fake quantization」モジュールを挿入し、量子化ノイズを学習時に模擬します。
その後 ONNX に書き出す際には、すでに量子化を考慮したパラメータを含んだモデルとなるため、bin への変換時の誤差が軽減され、高精度な INT8/FIX16 推論が可能になります。
```

