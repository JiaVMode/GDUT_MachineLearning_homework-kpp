from sklearn.pipeline import Pipeline
from sklearn.svm import SVC  # 修改为导入SVC
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
iris = datasets.load_iris()
X = iris['data'][:, (2, 3)]
y = (iris['target'] == 2).astype(np.float64)

scaler = StandardScaler()
# 使用SVC代替LinearSVC
svm_clf_c1 = SVC(kernel='linear', C=1, random_state=42)
svm_clf_c2 = SVC(kernel='linear', C=100, random_state=42)

# 创建并训练Pipeline
svm_clf_c1 = Pipeline([
    ('std', scaler),
    ('linear_svc', svm_clf_c1)
])
svm_clf_c1.fit(X, y)

svm_clf_c2 = Pipeline([
    ('std', scaler),
    ('linear_svc', svm_clf_c2)
])
svm_clf_c2.fit(X, y)

# 修改后的绘图函数
def plot_svm_decision_boundary(svm_clf, xmin, xmax, ax, sv=True):
    # 从Pipeline中获取scaler和模型
    scaler = svm_clf.named_steps['std']
    model = svm_clf.named_steps['linear_svc']
    w = model.coef_[0]
    b = model.intercept_[0]
    mean = scaler.mean_
    scale = scaler.scale_
    
    # 调整参数到原始空间
    w_original = w / scale
    b_original = b - (w * (mean / scale)).sum()
    
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = (-w_original[0] * x0 - b_original) / w_original[1]
    margin = 1 / np.linalg.norm(w_original)
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin
    
    # # 绘制支持向量（逆标准化到原始空间）
    # if sv:
    #     svs = model.support_vectors_
    #     svs_original = scaler.inverse_transform(svs)
    #     ax.scatter(svs_original[:, 0], svs_original[:, 1], s=180, facecolors='#FFAAAA')
    
    ax.plot(x0, decision_boundary, 'k-', linewidth=2)
    ax.plot(x0, gutter_up, 'k--', linewidth=2)
    ax.plot(x0, gutter_down, 'k--', linewidth=2)

# 创建子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))

# 绘制数据点
ax1.plot(X[:, 0][y==0], X[:, 1][y==0], 'bs')
ax1.plot(X[:, 0][y==1], X[:, 1][y==1], 'g^')
ax2.plot(X[:, 0][y==0], X[:, 1][y==0], 'bs')
ax2.plot(X[:, 0][y==1], X[:, 1][y==1], 'g^')

# 绘制决策边界和支持向量
plot_svm_decision_boundary(svm_clf_c1, 4, 6, ax1, sv=False)
plot_svm_decision_boundary(svm_clf_c2, 4, 6, ax2, sv=False)

# 设置坐标轴和标题
ax1.set_xlim(4, 6)
ax1.set_ylim(1, 2.75)
ax1.set_xlabel('Petal length')
ax1.set_ylabel('Petal width')
ax1.set_title('C = 1')
ax1.legend()

ax2.set_xlim(4, 6)
ax2.set_ylim(1, 2.75)
ax2.set_xlabel('Petal length')
ax2.set_ylabel('Petal width')
ax2.set_title('C = 100')
ax2.legend()

plt.show()