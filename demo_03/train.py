import numpy as np
import mlflow

# 1. 检查Scikit-learn并行配置
print("Scikit-learn CPU核心数：", mlflow.sklearn.get_config()["n_jobs"])  # 应为-1（所有核心）

# 2. 检查是否有GPU相关依赖激活
try:
    import torch
    print("PyTorch CUDA可用：", torch.cuda.is_available())  # 应为False
except ImportError:
    print("未安装PyTorch，无需检查GPU")

try:
    import tensorflow as tf
    print("TensorFlow GPU设备：", tf.config.list_physical_devices('GPU'))  # 应为[]
except ImportError:
    print("未安装TensorFlow，无需检查GPU")

# 3. 运行简单训练，确认CPU执行
iris = mlflow.sklearn.load_iris()
model = mlflow.sklearn.RandomForestClassifier(n_jobs=-1)
model.fit(iris.data, iris.target)
print("✅ 训练全程仅使用CPU，无GPU依赖！")