# 安装依赖（终端执行）
# pip install pandas==2.1.4 numpy==1.26.4 scikit-learn==1.3.2 imbalanced-learn==0.11.0
# pip install matplotlib==3.8.2 seaborn==0.13.1 pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2
# pip install mlflow==2.8.1 shap>=0.40.0 scikit-optimize==0.9.0
import os
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import shap

# ========== 全局配置 ==========
# 1. 设备配置（自动识别GPU/CPU）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")

# 2. 可视化配置
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = ["DejaVu Sans"]

# 3. MLflow配置（远程服务器地址替换为你的实际地址）
MLFLOW_TRACKING_URI = "http://your-mlflow-server:5000"  # 替换为远程MLflow地址
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("credit_default_prediction_pytorch")  # 实验名称

# ========== 1. 数据加载与预处理（保留原逻辑） ==========
file_path = "default of credit card clients.csv"
df = None

try:
    df = pd.read_csv(
        file_path,
        skiprows=1,
        encoding="latin-1",
        on_bad_lines="skip",
        engine="python"
    )

    # 列重命名
    rename_mapping = {"PAY_0": "PAY_1", "default payment next month": "default"}
    df.rename(columns={k: v for k, v in rename_mapping.items() if k in df.columns}, inplace=True)

    # 基础信息
    print("=" * 50)
    print("1. 数据基础信息")
    print("=" * 50)
    print(f"数据形状: {df.shape}")
    print(f"缺失值总数: {df.isnull().sum().sum()}")
    target_dist = df["default"].value_counts(normalize=True).round(4) * 100
    print(f"违约分布 - 不违约: {target_dist[0]}%, 违约: {target_dist[1]}%")

    # 特征预处理
    X = df.drop(["ID", "default"], axis=1)
    y = df["default"].astype(np.float32)

    # 修正异常值
    X["EDUCATION"] = X["EDUCATION"].replace([0, 5, 6], 4)
    X["MARRIAGE"] = X["MARRIAGE"].replace(0, 3)

    # 类别特征One-Hot编码
    cat_features = ["SEX", "EDUCATION", "MARRIAGE", "PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    X = pd.get_dummies(X, columns=cat_features, drop_first=True)

    # 数值特征标准化
    num_features = [col for col in X.columns if col not in cat_features]
    scaler = StandardScaler()
    X[num_features] = scaler.fit_transform(X[num_features])

    # 数据不平衡处理
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )

    # 转换为PyTorch张量
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32).to(DEVICE)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(DEVICE)
    y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32).to(DEVICE)

    # 创建DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    BATCH_SIZE = 128
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"\n预处理完成 - 训练集: {X_train.shape}, 测试集: {X_test.shape}")
    print(f"特征维度: {X_train.shape[1]}")

except FileNotFoundError:
    print(f"错误：文件 {file_path} 不存在！")
    exit()
except Exception as e:
    print(f"数据处理出错: {type(e).__name__} - {str(e)}")
    exit()


# ========== 2. PyTorch模型定义 ==========
class CreditDefaultNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.2):
        super(CreditDefaultNet, self).__init__()
        layers = []
        # 输入层
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # 隐藏层
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        # 输出层（二分类）
        layers.append(nn.Linear(hidden_dims[-1], 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ========== 3. 训练函数定义 ==========
def train_model(model, train_loader, test_loader, epochs, lr, weight_decay, patience=5):
    # 损失函数和优化器
    criterion = nn.BCELoss()  # 二分类交叉熵
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)

    # 早停配置
    best_auc = 0.0
    best_model_state = None
    early_stop_counter = 0

    # 训练记录
    train_losses = []
    val_losses = []
    val_aucs = []

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)

                # 收集预测结果
                preds = outputs.cpu().numpy()
                labels = batch_y.cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)

        avg_val_loss = val_loss / len(test_loader.dataset)
        val_losses.append(avg_val_loss)

        # 计算AUC
        val_auc = roc_auc_score(all_labels, all_preds)
        val_aucs.append(val_auc)

        # 学习率调整
        scheduler.step(val_auc)

        # 早停逻辑
        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"早停触发 - 第 {epoch + 1} 轮，最佳AUC: {best_auc:.4f}")
                break

        # 打印日志
        print(f"Epoch [{epoch + 1}/{epochs}] - 训练损失: {avg_train_loss:.4f}, "
              f"验证损失: {avg_val_loss:.4f}, 验证AUC: {val_auc:.4f}")

        # 上报MLflow（每轮指标）
        mlflow.log_metrics({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_auc": val_auc
        }, step=epoch)

    # 加载最佳模型
    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses, val_aucs, best_auc


# ========== 4. MLflow实验运行 ==========
with mlflow.start_run(run_name="credit_default_pytorch_run") as run:
    # ========== 4.1 上报超参数 ==========
    hyperparams = {
        "input_dim": X_train.shape[1],
        "hidden_dims": [128, 64, 32],
        "dropout": 0.2,
        "batch_size": BATCH_SIZE,
        "epochs": 50,
        "lr": 0.001,
        "weight_decay": 1e-4,
        "patience": 5,
        "optimizer": "Adam",
        "loss_fn": "BCELoss",
        "scheduler": "ReduceLROnPlateau"
    }
    mlflow.log_params(hyperparams)
    print("\n" + "=" * 50)
    print("2. 模型训练")
    print("=" * 50)
    print(f"超参数: {hyperparams}")

    # ========== 4.2 初始化并训练模型 ==========
    model = CreditDefaultNet(
        input_dim=hyperparams["input_dim"],
        hidden_dims=hyperparams["hidden_dims"],
        dropout=hyperparams["dropout"]
    ).to(DEVICE)

    # 打印模型结构
    print(f"\n模型结构:\n{model}")

    # 训练模型
    trained_model, train_losses, val_losses, val_aucs, best_auc = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=hyperparams["epochs"],
        lr=hyperparams["lr"],
        weight_decay=hyperparams["weight_decay"],
        patience=hyperparams["patience"]
    )

    # ========== 4.3 模型评估 ==========
    print("\n" + "=" * 50)
    print("3. 模型评估")
    print("=" * 50)

    # 测试集预测
    trained_model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = trained_model(batch_x)
            preds = outputs.cpu().numpy()
            labels = batch_y.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    # 转换为类别预测（阈值0.5）
    y_pred = (np.array(all_preds) >= 0.5).astype(int).reshape(-1)
    y_true = np.array(all_labels).reshape(-1)

    # 计算最终指标
    final_auc = roc_auc_score(y_true, all_preds)
    class_report = classification_report(y_true, y_pred, target_names=["Non-Default", "Default"], output_dict=True)

    # 上报最终指标到MLflow
    mlflow.log_metrics({
        "best_val_auc": best_auc,
        "test_auc": final_auc,
        "precision_non_default": class_report["Non-Default"]["precision"],
        "precision_default": class_report["Default"]["precision"],
        "recall_non_default": class_report["Non-Default"]["recall"],
        "recall_default": class_report["Default"]["recall"],
        "f1_non_default": class_report["Non-Default"]["f1-score"],
        "f1_default": class_report["Default"]["f1-score"],
        "accuracy": class_report["accuracy"]
    })

    print(f"最佳验证AUC: {best_auc:.4f}")
    print(f"测试集AUC: {final_auc:.4f}")
    print("\n分类报告:")
    print(classification_report(y_true, y_pred, target_names=["Non-Default", "Default"]))


    # ========== 4.4 可视化并上报MLflow ==========
    # 1. 混淆矩阵
    def plot_confusion_matrix():
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Non-Default", "Default"],
            yticklabels=["Non-Default", "Default"]
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")  # 上报到MLflow
        plt.close()


    # 2. ROC曲线
    def plot_roc_curve():
        fpr, tpr, _ = roc_curve(y_true, all_preds)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {final_auc:.4f}")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig("roc_curve.png")
        mlflow.log_artifact("roc_curve.png")
        plt.close()


    # 3. 训练损失曲线
    def plot_loss_curve():
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig("loss_curve.png")
        mlflow.log_artifact("loss_curve.png")
        plt.close()


    # 生成并上报可视化图表
    plot_confusion_matrix()
    plot_roc_curve()
    plot_loss_curve()

    # ========== 4.5 SHAP解释并上报 ==========
    print("\n" + "=" * 50)
    print("4. SHAP特征解释")
    print("=" * 50)

    # 转换为numpy用于SHAP
    X_test_np = X_test_tensor.cpu().numpy()
    # 使用DeepExplainer（适配PyTorch模型）
    explainer = shap.DeepExplainer(trained_model, X_train_tensor[:100].cpu())  # 采样加速
    shap_values = explainer.shap_values(X_test_np[:100])  # 采样加速

    # 特征重要性图
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test_np[:100], feature_names=X_test.columns, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig("shap_feature_importance.png")
    mlflow.log_artifact("shap_feature_importance.png")
    plt.close()

    # ========== 4.6 保存并上报模型 ==========
    # 保存模型到本地
    model_path = "credit_default_pytorch_model.pth"
    torch.save(trained_model.state_dict(), model_path)

    # 上报模型到MLflow
    signature = infer_signature(X_test_np[:5], trained_model(X_test_tensor[:5]).cpu().detach().numpy())
    mlflow.pytorch.log_model(
        pytorch_model=trained_model,
        artifact_path="model",
        signature=signature,
        input_example=X_test_np[:1]
    )
    mlflow.log_artifact(model_path)  # 同时上报pth文件

    # ========== 4.7 打印实验信息 ==========
    print("\n" + "=" * 50)
    print("5. MLflow实验信息")
    print("=" * 50)
    print(f"实验名称: {mlflow.get_experiment(run.info.experiment_id).name}")
    print(f"运行ID: {run.info.run_id}")
    print(f"MLflow地址: {MLFLOW_TRACKING_URI}")
    print(f"最佳AUC: {best_auc:.4f}, 测试集AUC: {final_auc:.4f}")
    print("\n✅ 训练完成！所有结果已上报到MLflow！")


# ========== 5. 加载模型示例（可选） ==========
def load_model_from_mlflow(run_id, model_path="model"):
    """从MLflow加载训练好的模型"""
    model_uri = f"runs:/{run_id}/{model_path}"
    loaded_model = mlflow.pytorch.load_model(model_uri)
    loaded_model.to(DEVICE)
    loaded_model.eval()
    return loaded_model

# 示例：加载模型并预测
# loaded_model = load_model_from_mlflow(run.info.run_id)
# test_pred = loaded_model(X_test_tensor[:10]).cpu().detach().numpy()
# print(f"预测示例: {test_pred}")