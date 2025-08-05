import torch
import torch.nn as nn
import torch.optim as optim
from torch.quantization import get_default_qat_qconfig, prepare_qat, convert
from torch.optim.lr_scheduler import StepLR

# === 初期設定 ===
qat_start_epoch = 5
total_epochs = 20
freeze_observer_epoch = 8  # Observer無効化タイミング
best_val_acc = 0.0

# 通常学習用 optimizer / scheduler
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

# === QAT 準備関数 ===
def enable_qat(model, device):
    model.train()
    model.qconfig = get_default_qat_qconfig("fbgemm")
    model = prepare_qat(model)
    model.to(device)

    # dummy inputで observer 初期化
    dummy_input = torch.randn(1, 3, image_size, image_size).to(device)
    with torch.no_grad():
        for _ in range(5):
            model(dummy_input)
    return model

# === 学習ループ ===
for epoch in range(total_epochs):
    print(f"\nEpoch [{epoch+1}/{total_epochs}]")

    # === QAT開始処理 ===
    if epoch == qat_start_epoch:
        print("🔁 QAT 開始")

        model.eval()
        model = enable_qat(model, device)
        model.train()

        # QAT用の optimizer/scheduler に変更
        optimizer = optim.SGD(model.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    # === Observer 無効化（QAT中盤以降）===
    if epoch == freeze_observer_epoch:
        print("🚫 Disable observer for QAT")
        torch.quantization.disable_observer(model)

    # === 通常学習 ===
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        # optional: 勾配爆発対策（必要に応じて）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100. * correct / total
    train_loss = total_loss / len(train_loader)

    # === バリデーション ===
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100. * correct / total
    val_loss /= len(val_loader)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.2f}%")

    # === 学習率調整 ===
    if scheduler:
        scheduler.step()

    # === モデル保存 ===
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_qat_model.pth")
        print(f"✅ Saved new best model (Val Acc: {val_acc:.2f}%)")
