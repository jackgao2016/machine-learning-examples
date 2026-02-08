# 端到端的信用风险预测流程（数据加载→预处理→平衡→建模→评估→调优→解释）

# 安装依赖（终端执行）
# pip install pandas==2.1.4 numpy==1.26.4 scikit-learn==1.3.2 imbalanced-learn==0.11.0
# pip install matplotlib==3.8.2 seaborn==0.13.1 xgboost==2.0.3 shap==0.44.1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap

# ========== 全局配置（核心修正：移除中文依赖，统一英文） ==========
# 仅保留负号适配，避免中文乱码/字体警告
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.figsize"] = (10, 6)  # 统一图表大小
plt.rcParams["font.size"] = 10  # 统一字体大小

# ========== 1. 数据加载与基础校验 ==========
file_path = "default of credit card clients.csv"  # 替换为你的文件路径
df = None  # 初始化DataFrame，避免后续变量未定义报错

try:
    # 强制使用 latin-1 编码读取（万能兼容）
    df = pd.read_csv(
        file_path,
        skiprows=1,  # 跳过UCI数据集第一行冗余标题
        encoding="latin-1",
        on_bad_lines="skip",  # 跳过错误行
        engine="python"  # 解决C解析器缓冲区溢出问题
    )

    # 重命名列（修正PAY_0笔误，简化目标列）
    rename_mapping = {
        "PAY_0": "PAY_1",
        "default payment next month": "default"
    }
    df.rename(columns={k: v for k, v in rename_mapping.items() if k in df.columns}, inplace=True)

    # 基础信息输出
    print("=" * 50)
    print("1. 数据基础信息")
    print("=" * 50)
    print(f"数据形状（行数, 列数）: {df.shape}")
    print(f"缺失值总数: {df.isnull().sum().sum()} (该数据集无缺失值)")

    if "default" in df.columns:
        target_dist = df["default"].value_counts(normalize=True).round(4) * 100
        print(f"违约分布 - 不违约(0): {target_dist[0]}%, 违约(1): {target_dist[1]}%")
    else:
        raise ValueError("未找到目标列 'default'，请检查数据集完整性！")

except FileNotFoundError:
    print(f"错误：文件 {file_path} 不存在，请确认路径正确！")
    print("提示：从UCI下载地址获取数据集：https://archive.ics.uci.edu/ml/machine-learning-databases/00350/")
    exit()  # 文件不存在时终止运行
except Exception as e:
    print(f"数据加载出错：{type(e).__name__} - {str(e)}")
    exit()

# ========== 2. 特征预处理 ==========
print("\n" + "=" * 50)
print("2. 特征预处理")
print("=" * 50)

# 特征/目标分离（排除ID列）
X = df.drop(["ID", "default"], axis=1)
y = df["default"]

# 修正异常类别值（教育/婚姻状态）
X["EDUCATION"] = X["EDUCATION"].replace([0, 5, 6], 4)  # 0/5/6→4（其他）
X["MARRIAGE"] = X["MARRIAGE"].replace(0, 3)  # 0→3（其他）

# 类别特征One-Hot编码
cat_features = ["SEX", "EDUCATION", "MARRIAGE", "PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
X = pd.get_dummies(X, columns=cat_features, drop_first=True)
print(f"One-Hot编码后特征数: {X.shape[1]}")

# 数值特征标准化
num_features = [col for col in X.columns if col not in cat_features]
scaler = StandardScaler()
X[num_features] = scaler.fit_transform(X[num_features])

# ========== 3. 数据不平衡处理（SMOTE） ==========
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)
balance_dist = pd.Series(y_balanced).value_counts(normalize=True).round(4) * 100
print(f"SMOTE平衡后分布 - 不违约: {balance_dist[0]}%, 违约: {balance_dist[1]}%")

# 划分训练/测试集（分层抽样）
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
)
print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

# ========== 4. 模型训练（XGBoost） ==========
print("\n" + "=" * 50)
print("3. 模型训练与评估")
print("=" * 50)

# 基础模型训练
xgb_model = XGBClassifier(
    random_state=42, learning_rate=0.1, n_estimators=100, max_depth=5, scale_pos_weight=1
)
xgb_model.fit(X_train, y_train)

# 预测结果
y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]


# ========== 5. 模型评估可视化 ==========
# 5.1 混淆矩阵绘制函数
def plot_confusion_matrix(y_true, y_pred, labels=["Non-Default", "Default"]):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, cbar=True
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Credit Default Prediction - Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.show()


# 5.2 ROC曲线绘制函数
def plot_roc_curve(y_true, y_pred_proba):
    auc = roc_auc_score(y_true, y_pred_proba)
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()
    return auc


# 执行可视化
plot_confusion_matrix(y_test, y_pred_xgb)
auc_score = plot_roc_curve(y_test, y_pred_proba_xgb)

# 分类报告（仅输出一次，修正重复问题）
print("基础模型分类报告:")
print(classification_report(y_test, y_pred_xgb, target_names=["Non-Default", "Default"]))
print(f"基础模型AUC-ROC: {auc_score:.4f}")

# ========== 6. 模型调优（网格搜索） ==========
print("\n" + "=" * 50)
print("4. 模型超参数调优")
print("=" * 50)

# 网格搜索配置
param_grid = {
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 5, 7],
    "n_estimators": [50, 100, 200]
}

grid_search = GridSearchCV(
    estimator=XGBClassifier(random_state=42),
    param_grid=param_grid,
    cv=5, scoring="roc_auc", n_jobs=-1
)
grid_search.fit(X_train, y_train)

# 最优模型评估
best_xgb = grid_search.best_estimator_
y_pred_best = best_xgb.predict(X_test)
best_auc = roc_auc_score(y_test, best_xgb.predict_proba(X_test)[:, 1])

print(f"最优参数: {grid_search.best_params_}")
print(f"交叉验证最优AUC: {grid_search.best_score_:.4f}")
print(f"调优后测试集AUC: {best_auc:.4f}")
print("调优后分类报告:")
print(classification_report(y_test, y_pred_best, target_names=["Non-Default", "Default"]))

# ========== 7. 模型解释（SHAP） ==========
print("\n" + "=" * 50)
print("5. 模型特征解释（SHAP）")
print("=" * 50)

# 计算SHAP值（针对最优模型）
explainer = shap.TreeExplainer(best_xgb)
shap_values = explainer.shap_values(X_test)

# 1. 特征重要性条形图
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.title("SHAP Feature Importance (Credit Default Prediction)", fontsize=14)
plt.tight_layout()
plt.show()

# 2. 核心特征依赖图（PAY_1_1：第一个还款状态为1的特征）
# 自动匹配PAY_1相关特征（避免特征名硬编码错误）
pay1_feature = [col for col in X_test.columns if "PAY_1_" in col][0]
shap.dependence_plot(
    pay1_feature, shap_values, X_test,
    xlabel=f"Feature: {pay1_feature} (Payment Status 1)",
    ylabel="SHAP Value (Impact on Default Probability)",
    show=True
)

print("\n✅ 信用风险预测流程执行完成！")