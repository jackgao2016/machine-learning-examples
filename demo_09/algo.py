import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

# 生成基础月亮数据集并查看基本信息
X, y = datasets.make_moons()
print("基础数据集形状 - X:", X.shape, " y:", y.shape)
print("X前5行:\n", X[:5])
print("y前5行:\n", y[:5])

# 可视化基础月亮数据集
plt.scatter(X[y==0,0], X[y==0,1], label='Class 0')
plt.scatter(X[y==1,0], X[y==1,1], label='Class 1')
plt.title("Basic Moons Dataset (No Noise)")
plt.legend()
plt.show()

# 生成带噪声的月亮数据集（更贴近真实场景）
X, y = datasets.make_moons(noise=0.15, random_state=777)
plt.scatter(X[y==0,0], X[y==0,1], label='Class 0')
plt.scatter(X[y==1,0], X[y==1,1], label='Class 1')
plt.title("Moons Dataset with Noise (0.15)")
plt.legend()
plt.show()

# 构建多项式特征+标准化+线性SVM的流水线
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

def PolynomialSVC(degree, C=1.0):
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),  # 生成多项式特征
        ("std_scaler", StandardScaler()),            # 标准化
        # 显式设置dual=False消除警告，增加max_iter避免收敛问题
        ("linearSVC", LinearSVC(C=C, dual=False, max_iter=10000))
    ])

# 训练3次多项式特征的SVM模型
poly_svc = PolynomialSVC(degree=3)
poly_svc.fit(X, y)

# 定义修复后的决策边界可视化函数（移除linewidth参数）
def plot_decision_boundary(model, axis):
    # 生成网格点覆盖可视化区域
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]  # 拼接成模型可预测的格式

    # 预测网格点类别
    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

    # 移除无效的linewidth参数，消除警告
    plt.contourf(x0, x1, zz, cmap=custom_cmap)

# 可视化多项式SVM的决策边界
plot_decision_boundary(poly_svc, axis=[-1.5, 2.5, -1.0, 1.5])
plt.scatter(X[y==0,0], X[y==0,1], label='Class 0')
plt.scatter(X[y==1,0], X[y==1,1], label='Class 1')
plt.title("Polynomial SVM (degree=3) Decision Boundary")
plt.legend()
plt.show()