import mlflow.pytorch
import torch
import pickle
import numpy as np

mlflow.set_tracking_uri("http://8.130.215.237:8081")
mlflow.set_experiment("PyTorch-鸢尾花分类-纯CPU")

# 加载模型（替换为你的Run ID）
run_id = "f4328eac7493473394b9c5097c4f2a33"
loaded_model = mlflow.pytorch.load_model(f"runs:/{run_id}/pytorch_iris_model")
loaded_model.to("cpu")  # 确保加载到CPU

# 加载scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# 推理示例
sample_data = np.array([[5.1, 3.5, 1.4, 0.2]])  # 鸢尾花样本
sample_data_scaled = scaler.transform(sample_data)
sample_tensor = torch.tensor(sample_data_scaled, dtype=torch.float32).to("cpu")

loaded_model.eval()
with torch.no_grad():
    output = loaded_model(sample_tensor)
    _, pred = torch.max(output, 1)
    print(f"预测类别: {pred.item()}")  # 输出0/1/2（对应鸢尾花三个类别）