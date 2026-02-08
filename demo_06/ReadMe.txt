基于 XGBoost 的信用风险预测代码重构为 PyTorch 实现，并接入 MLflow 完成训练过程和结果的远程上报。
以下是完整的 PyTorch+MLflow 重构代码，包含神经网络搭建、训练流程、MLflow 日志上报（参数、指标、模型、图表），
完全适配信用风险预测场景。

核心重构思路
模型层：用 PyTorch 搭建全连接神经网络（适配表格数据）；
训练层：实现完整的训练 / 验证循环（早停、学习率调整）；
MLflow 层：日志上报超参数、训练指标、模型文件、可视化图表；
兼容层：保留原数据预处理逻辑，确保输入数据格式适配 PyTorch 张量。

完整代码（PyTorch + MLflow)见 train.py

关键说明
1. MLflow 配置（必改）
替换 MLFLOW_TRACKING_URI 为你的远程 MLflow 服务器地址（如 http://192.168.1.100:5000）；
确保 MLflow 服务器已启动，且当前机器能访问该地址；
实验名称 credit_default_prediction_pytorch 可根据需要修改。

2. 模型设计适配性
采用全连接神经网络（MLP），适合表格型信用数据；
加入 Dropout 和 L2 正则（weight_decay）防止过拟合；
早停机制（Early Stopping）避免训练过度，保存最佳模型。

3. MLflow 上报内容
上报类型	具体内容
超参数	网络结构、训练参数（批次、学习率、epochs 等）
指标	训练 / 验证损失、AUC、精确率、召回率、F1、准确率
模型	PyTorch 模型文件（.pth）、MLflow 标准模型格式
可视化	混淆矩阵、ROC 曲线、损失曲线、SHAP 特征重要性图
签名	模型输入输出签名（便于批量预测）

4. 运行注意事项
GPU 加速：若有 NVIDIA GPU，安装 CUDA 版本的 PyTorch，训练速度提升 5-10 倍；
数据采样：SHAP 部分使用了采样（[:100]）加速，若需更精准可移除采样；
依赖兼容：确保 MLflow 服务器版本与客户端版本一致（推荐 2.8.1）；
端口开放：远程 MLflow 服务器需开放 5000 端口，确保当前机器能访问。

训练结果查看
访问 MLflow UI（http://your-mlflow-server:5000）；
选择实验 credit_default_prediction_pytorch；
查看对应 Run 的：
Parameters：所有超参数；
Metrics：训练 / 验证指标曲线；
Artifacts：模型文件、可视化图表；
Model：可直接下载或部署模型。

核心优势
PyTorch 原生支持：完整的神经网络训练流程，支持 GPU 加速；
MLflow 全链路上报：从超参数到最终模型，实现实验可追溯；
工业级训练逻辑：早停、学习率调整、正则化，保证模型泛化能力；
兼容原数据逻辑：保留原数据预处理流程，确保结果可比。