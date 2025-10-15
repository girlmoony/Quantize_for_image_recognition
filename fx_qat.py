import torch
import torch.ao.quantization as tq
from torch.ao.quantization import get_default_qat_qconfig
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx

# 0) backend
torch.backends.quantized.engine = "fbgemm"  # x86/サーバ向け

# 1) EfficientNet B0 を作成してヘッドを寿司クラス数に付け替え
model = build_efficientnet_b0(num_classes=N_SUSHI)  # あなたの実装でOK
model.train()

# 2) QAT用 qconfig
qconfig = get_default_qat_qconfig("fbgemm")  # ここは合ってます

# 3) FX Graph Mode で QAT 準備
qconfig_mapping = tq.qconfig_mapping.QConfigMapping().set_global(qconfig)
example_input = torch.randn(1, 3, 224, 224)  # 入力形状に合わせる
prepared = prepare_qat_fx(model, qconfig_mapping, example_input)  # ← これが“FX版prepare_qat”

# 4) QAT 学習
#  前半：Observer/Bn統計 有効のまま (量子化ノイズを学習に反映)
for epoch in range(warmup_epochs):
    train_one_epoch(prepared, ...)
    validate(prepared, ...)  # 量子化ノイズ込みの精度を見る

#  中盤以降：Observerを止め、BN統計も凍結して安定化
tq.disable_observer(prepared)
prepared.apply(torch.ao.quantization.freeze_bn_stats)  # or 自作でBN.eval()相当
for epoch in range(finetune_epochs):
    train_one_epoch(prepared, ...)
    validate(prepared, ...)

# 5) ここで“int8化しない”なら convert_fx は呼ばない
#    FP32のまま使う（FakeQuantだけ載っている状態）なら、保存前に無効化するとクリーン
tq.disable_fake_quant(prepared)
tq.disable_observer(prepared)
torch.save(prepared.state_dict(), "sushi_efficientnet_qat_fp32.pth")
