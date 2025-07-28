#QATモデルの重み分布を表示するスクリプト



import torch
import matplotlib.pyplot as plt

def plot_qat_weight_histograms(model, layer_names=None, bins=100):
    """
    QATモデルの重み（FakeQuantされた）のヒストグラムを描画

    Args:
        model: PyTorchのQAT準備済またはconvert()済みモデル
        layer_names: 可視化したい層の名前（例: ['features.0.0']）
        bins: ヒストグラムのbin数
    """
    print("🔍 可視化対象のFakeQuant層の重みを収集中...")
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            # layer指定があればスキップ処理
            if layer_names and not any(target in name for target in layer_names):
                continue

            try:
                # FakeQuant後の重み取得
                weights = module.weight.detach().cpu().numpy().flatten()
                plt.figure(figsize=(6, 3))
                plt.hist(weights, bins=bins, color='gray')
                plt.title(f"Weight Histogram: {name}")
                plt.xlabel("Weight value")
                plt.ylabel("Frequency")
                plt.grid(True)
                plt.tight_layout()
                plt.show()
            except Exception as e:
                print(f"[!] Skip {name}: {e}")


# QAT or convert済みモデルを読み込んだ後に実行
model = torch.load("qat_model_full.pt")  # or 定義して load_state_dict()

# 可視化（全Conv/Linear層）
plot_qat_weight_histograms(model)

# 一部の層だけを対象に（例: features.0.0 など）
# plot_qat_weight_histograms(model, layer_names=['features.0.0'])

#--------------------------------------------------
# QATなしモデル
model_float = ...  # floatモデルのロード
plot_qat_weight_histograms(model_float, layer_names=["features.0.0"])

# QAT済モデル
model_qat = ...  # QATモデル
plot_qat_weight_histograms(model_qat, layer_names=["features.0.0"])


