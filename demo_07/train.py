# 安装依赖（终端执行）
# pip install pandas==2.1.4 numpy==1.26.4 scikit-learn==1.3.2 imbalanced-learn==0.11.0
# pip install matplotlib==3.8.2 seaborn==0.13.1 xgboost==2.0.3 shap>=0.40.0
# pip install mlflow==2.8.1  # 新增MLflow依赖
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap

# ========== 全局配置 ==========
# 可视化配置（移除中文依赖）
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 10
plt.rcParams["font.family"] = ["DejaVu Sans"]

# MLflow配置（核心：连接Server，创建实验）
MLFLOW_TRACKING_URI = "http://8.130.215.237:8081"  # 替换为你的MLflow Server地址
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("CreditDefault-XGBoost-Sklearn-MLflow2")  # 实验名称
mlflow.sklearn.autolog(disable=True)  # 关闭自动日志，手动控制更灵活

# ========== 工具函数（可视化+临时文件保存） ==========
def plot_confusion_matrix(y_true, y_pred, save_path="confusion_matrix.png"):
    """绘制混淆矩阵并保存（用于MLflow日志）"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Non-Default", "Default"], yticklabels=["Non-Default", "Default"], cbar=True
    )
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Credit Default Prediction - Confusion Matrix", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()  # 关闭画布，避免内存泄漏
    return save_path

def plot_roc_curve(y_true, y_pred_proba, save_path="roc_curve.png"):
    """绘制ROC曲线并保存，返回AUC值"""
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
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return auc, save_path

def plot_shap_summary(shap_values, X_test, save_path="shap_summary.png"):
    """绘制SHAP特征重要性并保存"""
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance (Credit Default Prediction)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path

def plot_shap_dependence(shap_values, X_test, feature_name, save_path="shap_dependence.png"):
    """绘制SHAP依赖图并保存"""
    plt.figure(figsize=(10, 6))
    shap.dependence_plot(
        feature_name, shap_values, X_test, show=False
    )
    plt.title(f"Impact of {feature_name} on Credit Default", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    return save_path

# ========== 主流程 ==========
if __name__ == "__main__":
    # ========== 1. 数据加载与基础校验 ==========
    file_path = "default of credit card clients.csv"
    df = None

    try:
        # 加载数据
        df = pd.read_csv(
            file_path, skiprows=1, encoding="latin-1", on_bad_lines="skip", engine="python"
        )
        # 重命名列
        rename_mapping = {"PAY_0": "PAY_1", "default payment next month": "default"}
        df.rename(columns={k: v for k, v in rename_mapping.items() if k in df.columns}, inplace=True)

        # 基础信息输出
        print("=" * 50)
        print("1. 数据基础信息")
        print("=" * 50)
        print(f"数据形状（行数, 列数）: {df.shape}")
        print(f"缺失值总数: {df.isnull().sum().sum()}")
        target_dist = df["default"].value_counts(normalize=True).round(4) * 100
        print(f"违约分布 - 不违约(0): {target_dist[0]}%, 违约(1): {target_dist[1]}%")

    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在！")
        print("下载地址：https://archive.ics.uci.edu/ml/machine-learning-databases/00350/")
        exit()
    except Exception as e:
        print(f"数据加载出错：{type(e).__name__} - {str(e)}")
        exit()

    # ========== 2. 特征预处理 ==========
    print("\n" + "=" * 50)
    print("2. 特征预处理")
    print("=" * 50)

    # 特征/目标分离
    X = df.drop(["ID", "default"], axis=1)
    y = df["default"]

    # 修正异常值
    X["EDUCATION"] = X["EDUCATION"].replace([0, 5, 6], 4)
    X["MARRIAGE"] = X["MARRIAGE"].replace(0, 3)

    # One-Hot编码
    cat_features = ["SEX", "EDUCATION", "MARRIAGE", "PAY_1", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
    X = pd.get_dummies(X, columns=cat_features, drop_first=True)
    print(f"One-Hot编码后特征数: {X.shape[1]}")

    # 标准化
    scaler = StandardScaler()
    num_features = [col for col in X.columns if col not in cat_features]
    X[num_features] = scaler.fit_transform(X[num_features])

    # SMOTE平衡数据
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y)
    balance_dist = pd.Series(y_balanced).value_counts(normalize=True).round(4) * 100
    print(f"SMOTE平衡后分布 - 不违约: {balance_dist[0]}%, 违约: {balance_dist[1]}%")

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )
    print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

    # ========== 3. MLflow启动Run，记录全流程 ==========
    with mlflow.start_run(run_name="XGBoost-CreditDefault-FullProcess") as run:
        run_id = run.info.run_id
        print(f"\nMLflow Run ID: {run_id}")
        print(f"MLflow UI地址: {MLFLOW_TRACKING_URI}/#/experiments/-1/runs/{run_id}")

        # ---------- 3.1 日志预处理参数 ----------
        preprocess_params = {
            "smote_random_state": 42,
            "test_size": 0.2,
            "stratify": True,
            "cat_features": cat_features,
            "num_features": num_features,
            "one_hot_drop_first": True,
            "scaler_type": "StandardScaler"
        }
        mlflow.log_params(preprocess_params)

        # ---------- 3.2 训练基础XGBoost模型并日志 ----------
        print("\n" + "=" * 50)
        print("3. 基础模型训练与评估")
        print("=" * 50)

        # 基础模型参数
        base_xgb_params = {
            "random_state": 42,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "max_depth": 5,
            "scale_pos_weight": 1
        }
        mlflow.log_params({"base_model_" + k: v for k, v in base_xgb_params.items()})

        # 训练模型
        xgb_model = XGBClassifier(**base_xgb_params)
        xgb_model.fit(X_train, y_train)

        # 预测与评估
        y_pred_xgb = xgb_model.predict(X_test)
        y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

        # 日志基础模型指标
        base_auc, roc_path = plot_roc_curve(y_test, y_pred_proba_xgb)
        cm_path = plot_confusion_matrix(y_test, y_pred_xgb)
        mlflow.log_metric("base_model_auc", base_auc)

        # 提取分类报告关键指标并日志
        cls_report = classification_report(y_test, y_pred_xgb, target_names=["Non-Default", "Default"], output_dict=True)
        mlflow.log_metrics({
            "base_model_accuracy": cls_report["accuracy"],
            "base_model_precision_non_default": cls_report["Non-Default"]["precision"],
            "base_model_precision_default": cls_report["Default"]["precision"],
            "base_model_recall_non_default": cls_report["Non-Default"]["recall"],
            "base_model_recall_default": cls_report["Default"]["recall"],
            "base_model_f1_non_default": cls_report["Non-Default"]["f1-score"],
            "base_model_f1_default": cls_report["Default"]["f1-score"]
        })

        # 日志基础模型与可视化图表
        signature = infer_signature(X_test, y_pred_xgb)  # 生成模型输入输出签名
        mlflow.sklearn.log_model(
            sk_model=xgb_model,
            artifact_path="base_xgb_model",
            signature=signature,
            input_example=X_test.head(5)  # 输入示例
        )
        mlflow.log_artifact(roc_path, "visualizations")
        mlflow.log_artifact(cm_path, "visualizations")

        # 打印基础模型结果
        print("基础模型分类报告:")
        print(classification_report(y_test, y_pred_xgb, target_names=["Non-Default", "Default"]))
        print(f"基础模型AUC-ROC: {base_auc:.4f}")

        # ---------- 3.3 网格搜索调优并日志最优模型 ----------
        print("\n" + "=" * 50)
        print("4. 模型超参数调优")
        print("=" * 50)

        # 网格搜索参数
        param_grid = {
            "learning_rate": [0.01, 0.1],
            "max_depth": [3, 5, 7],
            "n_estimators": [50, 100, 200]
        }
        mlflow.log_params({"grid_search_" + k: v for k, v in param_grid.items()})

        # 网格搜索
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

        # 日志调优结果
        mlflow.log_params({"best_model_" + k: v for k, v in grid_search.best_params_.items()})
        mlflow.log_metrics({
            "best_model_cv_auc": grid_search.best_score_,
            "best_model_test_auc": best_auc,
            "best_model_accuracy": cls_report["accuracy"]  # 复用分类报告准确率
        })

        # 日志最优模型
        mlflow.sklearn.log_model(
            sk_model=best_xgb,
            artifact_path="best_xgb_model",
            signature=signature,
            input_example=X_test.head(5)
        )

        # 打印调优结果
        print(f"最优参数: {grid_search.best_params_}")
        print(f"交叉验证最优AUC: {grid_search.best_score_:.4f}")
        print(f"调优后测试集AUC: {best_auc:.4f}")
        print("调优后分类报告:")
        print(classification_report(y_test, y_pred_best, target_names=["Non-Default", "Default"]))

        # ---------- 3.4 SHAP解释并日志 ----------
        print("\n" + "=" * 50)
        print("5. 模型特征解释（SHAP）")
        print("=" * 50)
        print(f"当前SHAP版本：{shap.__version__}")

        # 计算SHAP值
        explainer = shap.TreeExplainer(best_xgb)
        shap_values = explainer.shap_values(X_test)

        # 绘制并日志SHAP图表
        shap_summary_path = plot_shap_summary(shap_values, X_test)
        pay1_feature = [col for col in X_test.columns if "PAY_1_" in col][0]
        shap_dependence_path = plot_shap_dependence(shap_values, X_test, pay1_feature)
        mlflow.log_artifact(shap_summary_path, "shap_plots")
        mlflow.log_artifact(shap_dependence_path, "shap_plots")

        # ---------- 3.5 日志其他关键文件 ----------
        # 保存并日志Scaler
        scaler_path = "scaler.pkl"
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, "preprocessor")

        # 日志数据集基础信息
        data_info_path = "data_info.txt"
        with open(data_info_path, "w") as f:
            f.write(f"原始数据形状: {df.shape}\n")
            f.write(f"平衡后数据形状: {X_balanced.shape}\n")
            f.write(f"训练集形状: {X_train.shape}\n")
            f.write(f"测试集形状: {X_test.shape}\n")
            f.write(f"原始违约分布: {target_dist.to_dict()}\n")
            f.write(f"平衡后违约分布: {balance_dist.to_dict()}\n")
        mlflow.log_artifact(data_info_path, "data_info")

        # ---------- 3.6 清理临时文件 ----------
        temp_files = [roc_path, cm_path, shap_summary_path, shap_dependence_path, scaler_path, data_info_path]
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)

        print("\n✅ 信用风险预测流程执行完成！")
        print(f"MLflow实验地址: {MLFLOW_TRACKING_URI}/#/experiments/-1/runs/{run_id}")