# ====== 設定 ======
total_epochs = 300
qat_start_epoch = 200  # 🔧 ← ここを変えるだけでQAT開始タイミングを変更可能
batch_size = 128
image_size = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== データローダ ======
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
train_dataset = datasets.FakeData(size=1000, image_size=(3, image_size, image_size), transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ====== モデルと損失 ======
model = efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ====== QAT 準備関数 ======
def enable_qat(model):
    model.train()
    model.qconfig = get_default_qat_qconfig("fbgemm")
    model_prepared = prepare_qat(model)
    
    # dummy inputでFakeQuantizer初期化（重要）
    dummy_input = torch.randn(batch_size, 3, image_size, image_size).to(device)
    with torch.no_grad():
        for _ in range(5):
            model_prepared(dummy_input)
    return model_prepared

# ====== 学習ループ ======
for epoch in range(total_epochs):
    print(f"\n[Epoch {epoch+1}/{total_epochs}]")
    
    # QAT切り替え
    if epoch == qat_start_epoch:
        print("🔁 QAT 開始")
        model = enable_qat(model)
        optimizer = optim.Adam(model.parameters(), lr=1e-5)  # 学習率を小さくするのが一般的

    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Loss: {running_loss:.4f}")
