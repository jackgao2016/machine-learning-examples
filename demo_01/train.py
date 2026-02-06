import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1. 连接远程MLflow Server
mlflow.set_tracking_uri("http://8.130.215.237:8081")
mlflow.set_experiment("修复源码记录-鸢尾花分类5")

# 2. 开启autolog（移除log_source参数，避免报错）
mlflow.sklearn.autolog()  # 核心：删除log_source=True

# 3. 加载数据+定义参数
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)
params = {"n_estimators": 100, "max_depth": 5}

# 4. 启动Run，手动记录源码（替代log_source的核心逻辑）
with mlflow.start_run():
    # 关键：手动记录当前训练脚本的源码（__file__是当前脚本路径）
    mlflow.log_artifact(__file__, "training_source")  # 源码存到training_source目录

    # 可选：记录其他依赖文件（如utils.py、config.yaml）
    # mlflow.log_artifact("utils.py", "training_source")

    # 训练模型+自动记录参数/指标（autolog的核心功能不受影响）
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    # 新增：用skops格式保存模型，消除安全警告
    mlflow.sklearn.log_model(model, name="random_forest_model", serialization_format="skops")

print("训练完成，源码已手动记录到远程MLflow Server！")