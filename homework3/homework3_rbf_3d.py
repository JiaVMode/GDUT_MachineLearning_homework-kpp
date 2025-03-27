import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

def plot_comparison_modified_colors(X, y, param_combinations, features=(0,1,2)):
    """
    多参数组合三维对比可视化（修改数据点颜色）
    
    参数:
    X : 原始特征矩阵
    y : 标签向量
    param_combinations : 参数组合列表 [(gamma1, C1), (gamma2, C2), ...]
    features : 选择可视化的三个特征索引
    """
    # 特征选择与标准化
    X_vis = X[:, features]
    X_scaled = StandardScaler().fit_transform(X_vis)
    
    # 创建子图
    n = len(param_combinations)
    rows = int(np.sqrt(n))
    cols = int(np.ceil(n / rows))
    fig = plt.figure(figsize=(cols*6, rows*6))
    
    # 生成网格
    x_min, x_max = X_scaled[:,0].min()-1, X_scaled[:,0].max()+1
    y_min, y_max = X_scaled[:,1].min()-1, X_scaled[:,1].max()+1
    z_min, z_max = X_scaled[:,2].min()-1, X_scaled[:,2].max()+1
    xx, yy, zz = np.mgrid[x_min:x_max:30j, 
                         y_min:y_max:30j, 
                         z_min:z_max:30j]
    grid_test = np.stack((xx.ravel(), yy.ravel(), zz.ravel()), axis=1)

    # 修改后的颜色配置
    cm_light = matplotlib.colors.ListedColormap(['#FF6666', '#6666FF', '#66FF66'])  # 红/蓝/绿加深版
    cm_dark = matplotlib.colors.ListedColormap(['#CC0000', '#0000CC', '#00CC00'])    # 更饱和的颜色
    markers = ['^', 's', 'o']  # 对应三类

    for idx, (gamma, C) in enumerate(param_combinations, 1):
        ax = fig.add_subplot(rows, cols, idx, projection='3d')
        
        # 训练模型
        svm = SVC(kernel='rbf', gamma=gamma, C=C).fit(X_scaled, y)
        
        # 预测并绘制决策区域
        grid_hat = svm.predict(grid_test).reshape(xx.shape)
        ax.scatter(xx.ravel(), yy.ravel(), zz.ravel(),
                  c=grid_hat, cmap=cm_light, alpha=0.03, s=8)
        
        # 绘制原始数据点（不同形状和颜色）
        for k, marker in enumerate(markers):
            mask = (y == k)
            ax.scatter(X_scaled[mask,0], X_scaled[mask,1], X_scaled[mask,2],
                      marker=marker, edgecolor='k', s=30,
                      c=y[mask], cmap=cm_dark, vmin=0, vmax=2)  # 确保颜色映射范围正确
        
        # 坐标轴标签设置
        feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
        ax.set_xlabel(feature_names[features[0]])
        ax.set_ylabel(feature_names[features[1]])
        ax.set_zlabel(feature_names[features[2]])
        ax.set_title(f'gamma={gamma}, C={C}', fontsize=10)

    plt.tight_layout()
    plt.show()

# 使用示例
iris = datasets.load_iris()
param_combinations = [(0.1,0.001), (0.1,1000), (1,0.001), (1,1000)]

# 可视化花瓣长度+宽度+萼片长度（修改颜色后的版本）
plot_comparison_modified_colors(iris.data, iris.target, param_combinations, features=(2,3,0))