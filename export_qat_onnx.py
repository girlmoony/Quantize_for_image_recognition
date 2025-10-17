# export_qat_onnx.py
# QAT（FX）チェックポイントを読み、reference 形式に変換して ONNX へ
import argparse
import torch
from torch.ao.quantization import get_default_qat_qconfig, QConfigMapping
from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx

# -----------------------------
# ここをあなたのモデル定義に合わせて書き換え
def build_efficientnet_float(num_classes=1000):
    raise NotImplementedError("あなたの EfficientNet 初期化に置き換えてください")
# -----------------------------

def strip_module_prefix(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v
    return new_sd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--classes", type=int, default=1000)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--backend", default="fbgemm", choices=["fbgemm", "qnnpack"])
    ap.add_argument("--opset", type=int, default=13)
    ap.add_argument("--dynamic", action="store_true", help="可変バッチをONNXで許可")
    args = ap.parse_args()

    torch.backends.quantized.engine = args.backend

    # 1) float モデル
    float_model = build_efficientnet_float(num_classes=args.classes)
    float_model.eval()

    # 2) QAT 準備（学習時と同じ）
    qconfig = get_default_qat_qconfig(args.backend)
    qconfig_mapping = QConfigMapping().set_global(qconfig)
    example_input = torch.randn(1, 3, args.imgsz, args.imgsz)
    prepared = prepare_qat_fx(float_model, qconfig_mapping, example_input)

    # 3) チェックポイント読み込み
    sd = torch.load(args.ckpt, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = strip_module_prefix(sd)
    res = prepared.load_state_dict(sd, strict=False)
    print("[load_state_dict] missing:", res.missing_keys[:20])
    print("[load_state_dict] unexpected:", res.unexpected_keys[:20])

    prepared.eval()
    # 4) reference 形式へ変換（ONNX で扱いやすい Q/DQ 構成）
    ref_model = convert_fx(prepared, is_reference=True)

    # 5) ONNX エクスポート
    ref_model.eval()
    dummy = torch.randn(1, 3, args.imgsz, args.imgsz)

    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {"input": {0: "batch"}, "output": {0: "batch"}}

    torch.onnx.export(
        ref_model,
        dummy,
        args.onnx,
        input_names=["input"],
        output_names=["output"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamic_axes=dynamic_axes,
    )
    print(f"Exported ONNX -> {args.onnx}")

if __name__ == "__main__":
    main()
