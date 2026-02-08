import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

# 加载并预处理数据
iris = datasets.load_iris()
X = iris.data
y = iris.target
X = X[y<2,:2]
y = y[y<2]

# 可视化原始数据
plt.scatter(X[y==0,0], X[y==0,1], color='red')
plt.scatter(X[y==1,0], X[y==1,1], color='blue')
plt.title("Original Iris Data (2 classes, 2 features)")
plt.show()

# 数据标准化
from sklearn.preprocessing import StandardScaler
standardScaler = StandardScaler()
standardScaler.fit(X)
X_standard = standardScaler.transform(X)

# 训练LinearSVM模型
from sklearn.svm import LinearSVC
# 硬间隔SVM (C很大)
svc = LinearSVC(C=1e9)
svc.fit(X_standard, y)
# 软间隔SVM (C很小)
svc2 = LinearSVC(C=0.01)
svc2.fit(X_standard, y)

# 定义决策边界可视化函数（修复了linewidth警告）
def plot_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

    # 移除了无效的linewidth参数
    plt.contourf(x0, x1, zz, cmap=custom_cmap)

# 可视化硬间隔SVM决策边界
plot_decision_boundary(svc, axis=[-3, 3, -3, 3])
plt.scatter(X_standard[y==0,0], X_standard[y==0,1], label='Class 0')
plt.scatter(X_standard[y==1,0], X_standard[y==1,1], label='Class 1')
plt.title("Hard Margin SVM (C=1e9) Decision Boundary")
plt.legend()
plt.show()

# 可视化软间隔SVM决策边界
plot_decision_boundary(svc2, axis=[-3, 3, -3, 3])
plt.scatter(X_standard[y==0,0], X_standard[y==0,1], label='Class 0')
plt.scatter(X_standard[y==1,0], X_standard[y==1,1], label='Class 1')
plt.title("Soft Margin SVM (C=0.01) Decision Boundary")
plt.legend()
plt.show()

# 输出模型参数
print("Hard Margin SVM coef:", svc.coef_)
print("Hard Margin SVM intercept:", svc.intercept_)

# 定义带间隔边界的SVM可视化函数（修复了linewidth警告）
def plot_svc_decision_boundary(model, axis):
    x0, x1 = np.meshgrid(
        np.linspace(axis[0], axis[1], int((axis[1] - axis[0]) * 100)).reshape(-1, 1),
        np.linspace(axis[2], axis[3], int((axis[3] - axis[2]) * 100)).reshape(-1, 1),
    )
    X_new = np.c_[x0.ravel(), x1.ravel()]

    y_predict = model.predict(X_new)
    zz = y_predict.reshape(x0.shape)

    from matplotlib.colors import ListedColormap
    custom_cmap = ListedColormap(['#EF9A9A', '#FFF59D', '#90CAF9'])

    # 移除了无效的linewidth参数
    plt.contourf(x0, x1, zz, cmap=custom_cmap)

    w = model.coef_[0]
    b = model.intercept_[0]

    # 计算决策边界和间隔边界
    plot_x = np.linspace(axis[0], axis[1], 200)
    up_y = -w[0] / w[1] * plot_x - b / w[1] + 1 / w[1]
    down_y = -w[0] / w[1] * plot_x - b / w[1] - 1 / w[1]

    # 过滤超出可视化范围的点
    up_index = (up_y >= axis[2]) & (up_y <= axis[3])
    down_index = (down_y >= axis[2]) & (down_y <= axis[3])
    plt.plot(plot_x[up_index], up_y[up_index], color='black')
    plt.plot(plot_x[down_index], down_y[down_index], color='black')

# 可视化硬间隔SVM的决策边界+间隔
plot_svc_decision_boundary(svc, axis=[-3, 3, -3, 3])
plt.scatter(X_standard[y==0,0], X_standard[y==0,1], label='Class 0')
plt.scatter(X_standard[y==1,0], X_standard[y==1,1], label='Class 1')
plt.title("Hard Margin SVM with Margin Boundaries")
plt.legend()
plt.show()

# 可视化软间隔SVM的决策边界+间隔
plot_svc_decision_boundary(svc2, axis=[-3, 3, -3, 3])
plt.scatter(X_standard[y==0,0], X_standard[y==0,1], label='Class 0')
plt.scatter(X_standard[y==1,0], X_standard[y==1,1], label='Class 1')
plt.title("Soft Margin SVM with Margin Boundaries")
plt.legend()
plt.show()
