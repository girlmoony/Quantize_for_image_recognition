import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision.models import efficientnet_b0
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch.ao.quantization import get_default_qat_qconfig, prepare_qat, convert
import copy

# ==== 環境設定 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
image_size = 256
num_classes = 10
total_epochs = 300
qat_starts = [200, 220, 250]  # 🔁 試したい QAT 開始エポック一覧

# ==== ダミーデータ（代替で ImageNet 等に切り替え可能）====
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_dataset = datasets.FakeData(size=1000, image_size=(3, image_size, image_size), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ==== QAT 用ヘルパー ====
def enable_qat(model):
    model.train()
    model.qconfig = get_default_qat_qconfig("fbgemm")
    model = prepare_qat(model)

    # dummy forward で observer 初期化
    dummy_input = torch.randn(batch_size, 3, image_size, image_size).to(device)
    with torch.no_grad():
        for _ in range(5):
            model(dummy_input)
    return model

# ==== 学習実験ループ ====
for qat_start_epoch in qat_starts:
    print(f"\n=== QAT開始エポック: {qat_start_epoch} ===")

    # モデル初期化
    model = efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.1)

    for epoch in range(total_epochs):
        print(f"\nEpoch [{epoch+1}/{total_epochs}]")

        if epoch == qat_start_epoch:
            print("🔁 QAT 開始")
            model.eval()
            model = enable_qat(model)
            model.to(device)
            model.train()

            # 🔽 Optimizerはそのまま、学習率だけ下げる
            for param_group in optimizer.param_groups:
                param_group['lr'] = 1e-5

        # === 学習フェーズ ===
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        acc = correct / total
        print(f"Loss: {total_loss:.4f} | Accuracy: {acc:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

    # ==== 最終QAT変換 & 保存 ====
    model.cpu()
    model.eval()
    model_int8 = convert(model)
    torch.save(model_int8.state_dict(), f"qat_{qat_start_epoch}_int8.pth")
    print(f"✅ モデル保存: qat_{qat_start_epoch}_int8.pth")
