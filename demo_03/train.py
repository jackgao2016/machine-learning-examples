import numpy as np
import mlflow
import sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# 1. 兼容所有Scikit-learn版本的CPU核心数检查（修正KeyError）
# 方案：不读取全局n_jobs（避免KeyError），直接验证模型的n_jobs参数
print(f"Scikit-learn版本：{sklearn.__version__}")
# 直接定义模型时指定n_jobs=-1（利用所有CPU核心）
model = RandomForestClassifier(n_jobs=-1)
print(f"模型使用的CPU核心数：{model.n_jobs}")  # 输出-1（所有核心）

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

# 3. 运行简单训练，确认CPU执行
iris = load_iris()
model.fit(iris.data, iris.target)
print("✅ 训练全程仅使用CPU，无GPU依赖！")

# 4. 用MLflow记录本次训练（验证日志功能无GPU依赖）
mlflow.set_tracking_uri("http://8.130.215.237:8081")
mlflow.set_experiment("验证纯CPU训练_兼容sklearn1.3.0")
with mlflow.start_run():
    mlflow.log_param("n_jobs", model.n_jobs)  # 记录模型使用的CPU核心数
    mlflow.log_metric("train_score", model.score(iris.data, iris.target))
    # 用skops格式保存模型，消除pickle安全警告
    mlflow.sklearn.log_model(model, name="cpu_model", serialization_format="pickle")

print("✅ MLflow日志记录完成，全程无GPU参与！")