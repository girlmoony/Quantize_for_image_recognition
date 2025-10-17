# test_qat_infer.py
# PyTorch 2.0.1 / torch.ao.quantization (FX QAT) 前提
import argparse
import torch
import torch.nn as nn
from torch.ao.quantization import get_default_qat_qconfig, QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx

# -----------------------------
# ここをあなたのモデル定義に合わせて書き換え
# 学習時、「prepare_qat_fx する前」の float モデルを返す関数
def build_efficientnet_float(num_classes=1000):
    # 例:
    # from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    # m = efficientnet_b0(weights=None, num_classes=num_classes)
    # return m
    raise NotImplementedError("あなたの EfficientNet 初期化に置き換えてください")
# -----------------------------

def strip_module_prefix(state_dict):
    # DDP/DataParallel の 'module.' を外す
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v
    return new_sd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="学習時に保存した state_dict ファイル（.pth）")
    ap.add_argument("--classes", type=int, default=1000)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--backend", default="fbgemm", choices=["fbgemm", "qnnpack"])
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    torch.backends.quantized.engine = args.backend

    # 1) float モデルを学習時と同じ構成で作る（fuse を入れていないなら入れない）
    float_model = build_efficientnet_float(num_classes=args.classes)
    float_model.eval()  # FX は eval でも可（train でも可）

    # 2) QAT の設定（学習時と同じ qconfig を使う）
    qconfig = get_default_qat_qconfig(args.backend)
    qconfig_mapping = QConfigMapping().set_global(qconfig)

    # 3) FX の prepare（学習時と同じ「例入力」形状を用意）
    example_input = torch.randn(1, 3, args.imgsz, args.imgsz)
    prepared = prepare_qat_fx(float_model, qconfig_mapping, example_input)

    # 4) チェックポイントを読み込み
    sd = torch.load(args.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]  # Lightning などのラッパー形式に対応
    sd = strip_module_prefix(sd)

    # 5) prepared モデルにロード（QATの fake_quant/observer を含むことを想定）
    res = prepared.load_state_dict(sd, strict=False)
    print("[load_state_dict] missing:", res.missing_keys[:20])
    print("[load_state_dict] unexpected:", res.unexpected_keys[:20])

    # 6) 推論（prepared のままでも動くが、最終推論は convert 後が一般的）
    prepared.eval()
    quantized_model = convert_fx(prepared)  # 量子化済み（int8演算を含む）モデル

    # ダミー入力で通し、Top-k を表示
    with torch.inference_mode():
        x = torch.randn(1, 3, args.imgsz, args.imgsz)
        logits = quantized_model(x)
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        prob = torch.softmax(logits, dim=1)
        topk = min(args.topk, prob.shape[1])
        vals, idxs = torch.topk(prob, k=topk, dim=1)
        print("Top-{} indices:".format(topk), idxs.tolist())
        print("Top-{} probs:".format(topk), vals.tolist())

if __name__ == "__main__":
    main()
