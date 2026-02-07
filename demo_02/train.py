import torch
# 强制PyTorch使用CPU（即使安装了CUDA也会忽略）
torch.cuda.is_available = lambda: False  # 伪装成无CUDA
device = torch.device("cpu")  # 所有张量/模型都部署到CPU

# 示例：PyTorch模型训练（仅CPU）
model = torch.nn.Linear(4, 3).to(device)  # 模型放到CPU
X = torch.tensor([[1.0, 2.0, 3.0, 4.0]]).to(device)  # 数据放到CPU
output = model(X)
print(output)