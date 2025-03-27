import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# 加载数据并预处理
iris = datasets.load_iris()
X = iris.data[:, 2:4]  # 花瓣长度和宽度
y = iris.target

# 标准化特征
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 创建子图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
plt.subplots_adjust(wspace=0.3, hspace=0.3)

# 参数组合
gamma_values = [0.1, 5]
C_values = [0.001, 1000]

# 生成网格数据
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 不同类别对应的形状
markers = ['o', 's', '^']  # 圆形、方形、三角形

# 遍历所有参数组合
for i, gamma in enumerate(gamma_values):
    for j, C in enumerate(C_values):
        ax = axes[i, j]
        
        # 训练模型
        svm = SVC(kernel='rbf', gamma=gamma, C=C)
        svm.fit(X_scaled, y)
        
        # 预测网格点
        Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        # 绘制决策边界
        ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
        
        # 绘制不同形状的数据点
        for k, marker in enumerate(markers):
            ax.scatter(X_scaled[y == k, 0], 
                       X_scaled[y == k, 1], 
                       marker=marker, 
                       edgecolor='k', 
                       s=30,
                       label=iris.target_names[k])
        
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xlabel('Petal Length')
        ax.set_ylabel('Petal Width')
        ax.set_title(f'gamma={gamma}, C={C}')

# 添加统一图例
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=3)

plt.show()