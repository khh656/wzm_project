import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

# 主成分分析法作用:选中了20个指标,你觉得都很重要,但是20个指标对于你的分析确实太过繁琐,
# 这时候,你就可以采用主成分分析的方法进行降维.
# 输入待降维数据 (5 * 6) 矩阵，6个维度，5个样本值
# A = np.array([[84, 65, 61, 72, 79, 81], [64, 77, 77, 76, 55, 70], [65, 67, 63, 49, 57, 67], [74, 80, 69, 75, 63, 74],[84, 74, 70, 80, 74, 82]])
#导入数据
data_ori = pd.read_csv(r'E:\work\ai_work\oss_health\data\project_1486.csv')
data = data_ori.drop(columns=['project_id', 'date', 'committer_id', 'req_merged'])
target = data_ori['req_merged']
# 直接使用PCA进行降维
pca = PCA(n_components=9)  # 降到 2 维
pca.fit(data)

pca.transform(data)  # 降维后的结果
res1 = pca.explained_variance_ratio_  # 降维后的各主成分的方差值占总方差值的比例，即方差贡献率

# res2 = pca.explained_variance_  # 降维后的各主成分的方差值
print(res1)
# print(res2)
