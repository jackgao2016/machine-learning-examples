import numpy as np
import mlflow
import sklearn  # 导入原生sklearn
from sklearn import config_context  # 用于获取CPU配置
from sklearn.datasets import load_iris  # 导入数据集
from sklearn.ensemble import RandomForestClassifier  # 导入模型

# 1. 检查Scikit-learn并行配置（修正核心错误）
# 获取全局默认n_jobs配置（默认是None，手动设置为-1利用所有CPU核心）
print("Scikit-learn默认CPU核心数：", sklearn.get_config()["n_jobs"])
# 手动设置全局n_jobs为-1（所有CPU核心）
sklearn.set_config(n_jobs=-1)
print("修改后Scikit-learn CPU核心数：", sklearn.get_config()["n_jobs"])  # 输出-1

# 2. 检查是否有GPU相关依赖激活
try:
    import torch
    print("PyTorch CUDA可用：", torch.cuda.is_available())  # 无GPU则输出False
except ImportError:
    print("未安装PyTorch，无需检查GPU")

try:
    import tensorflow as tf
    print("TensorFlow GPU设备：", tf.config.list_physical_devices('GPU'))  # 无GPU则输出[]
except ImportError:
    print("未安装TensorFlow，无需检查GPU")

# 3. 运行简单训练，确认CPU执行（修正数据集和模型调用）
iris = load_iris()  # 原生sklearn加载数据集
model = RandomForestClassifier(n_jobs=-1)  # 强制使用所有CPU核心
model.fit(iris.data, iris.target)
print("✅ 训练全程仅使用CPU，无GPU依赖！")

# 可选：用MLflow记录本次训练（验证日志功能也无GPU依赖）
mlflow.set_tracking_uri("http://8.130.215.237:8081")
mlflow.set_experiment("验证纯CPU训练")
with mlflow.start_run():
    mlflow.log_param("n_jobs", -1)
    mlflow.log_metric("train_score", model.score(iris.data, iris.target))
    mlflow.sklearn.log_model(model, name="cpu_model", serialization_format="skops")
print("✅ MLflow日志记录完成，全程无GPU参与！")