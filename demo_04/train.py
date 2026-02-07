import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pickle  # 提前导入pickle

# ===================== 1. 强制PyTorch使用CPU（无GPU适配） =====================
torch.cuda.is_available = lambda: False
device = torch.device("cpu")
print(f"使用设备: {device}")

# ===================== 2. 连接远程MLflow Server =====================
mlflow.set_tracking_uri("http://8.130.215.237:8081")
mlflow.set_experiment("PyTorch-鸢尾花分类-纯CPU")


# ===================== 3. 定义PyTorch模型（适配鸢尾花分类） =====================
class IrisClassifier(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=16, output_dim=3):
        super(IrisClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)


# ===================== 4. 数据预处理（PyTorch格式） =====================
# 加载数据并标准化
iris = load_iris()
X = iris.data
y = iris.target

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===== 关键修正1：提前保存scaler.pkl，确保log_model时文件存在 =====
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# 划分训练/测试集，转为PyTorch张量
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# 构建DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# ===================== 5. 训练参数配置 =====================
params = {
    "input_dim": 4,
    "hidden_dim": 16,
    "output_dim": 3,
    "lr": 0.01,
    "epochs": 50,
    "batch_size": 8,
    "dropout": 0.2
}

# ===================== 6. MLflow日志+模型训练 =====================
with mlflow.start_run():
    # 记录当前脚本源码
    mlflow.log_artifact(__file__, "training_source")

    # 记录训练参数
    mlflow.log_params(params)

    # 初始化模型、损失函数、优化器
    model = IrisClassifier(
        input_dim=params["input_dim"],
        hidden_dim=params["hidden_dim"],
        output_dim=params["output_dim"]
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params["lr"])

    # 训练循环
    model.train()
    for epoch in range(params["epochs"]):
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)

    # 测试集评估
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        _, preds = torch.max(test_outputs, 1)
        accuracy = (preds == y_test_tensor).float().mean().item()

    # 记录核心指标
    mlflow.log_metric("test_accuracy", accuracy)

    # ===== 关键修正2：此时scaler.pkl已存在，可正常传入extra_files =====
    mlflow.pytorch.log_model(
        pytorch_model=model,
        name="pytorch_iris_model",
        extra_files=["scaler.pkl"]  # 现在文件存在，不会报错
    )

    # 可选：额外将scaler.pkl记录到artifacts目录（便于查看）
    mlflow.log_artifact("scaler.pkl", "preprocessor")

print(f"训练完成！测试集准确率: {accuracy:.4f}")
print("所有日志（源码、参数、指标、模型）已记录到远程MLflow Server！")