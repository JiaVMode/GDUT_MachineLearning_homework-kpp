import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

# 加载数据
iris = datasets.load_iris()
X = iris['data'][:, (2, 3)]  # 使用花瓣长度和花瓣宽度
y = (iris['target'] == 2).astype(np.float64)  # 二分类（是否Iris-Virginica）

# 生成网格点
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# 定义可视化参数
colors = ['blue', 'red']       # 负类/正类颜色
markers = ['o', '^']           # 负类/正类形状
labels = ['Not Virginica', 'Virginica']  # 图例标签

# 创建参数组合
svm_params = [
    {'degree': 3, 'coef0': 1, 'C': 5},
    {'degree': 4, 'coef0': 10, 'C': 5}
]

# 创建子图
fig, axes = plt.subplots(1, 2, figsize=(20, 7))

for ax, params in zip(axes, svm_params):
    # 训练模型
    svm = SVC(
        kernel='poly',
        degree=params['degree'],
        coef0=params['coef0'],
        C=params['C'],
        gamma='scale'
    )
    svm.fit(X, y)
    
    # 预测决策边界
    Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    
    # 绘制决策区域
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Pastel2)
    
    # 绘制样本点
    for i in [0, 1]:  # 遍历负类和正类
        ax.scatter(
            X[y == i, 0],
            X[y == i, 1],
            c=colors[i],
            marker=markers[i],
            edgecolors='k',
            s=60,
            label=labels[i]
        )
    
    # 添加标注
    ax.set_xlabel('Petal length')
    ax.set_ylabel('Petal width')
    ax.set_title(
        f"Degree={params['degree']}, Coef0={params['coef0']}, C={params['C']}\n"
        f"Accuracy: {svm.score(X, y):.2f}"
    )
    ax.legend()

plt.tight_layout()
plt.show()