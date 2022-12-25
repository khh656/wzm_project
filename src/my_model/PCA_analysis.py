import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# 用这个  des_path = r'E:/work/ai_work/oss_health/assess_data_90/ten_16.csv'
def pca_scores(file_path):
    data_ori = pd.read_csv(file_path)
    df = data_ori.drop(columns=['project_id', 'date'])

    # 数据标准化, 标准化之后变为均值为0, 方差为1的数据
    x=df.values
    x = StandardScaler().fit_transform(x)
    num_fac = get_weidu(x)
    # 使用PCA降维
    pca = PCA(n_components=num_fac)
    principalComponents = pca.fit_transform(x)
    # 降维后的新数据--主成分的得分相加？？？
    # principalDf = pd.DataFrame(data=principalComponents, columns=['pc1', 'pc2', 'pc3', 'pc4'])
    # finalDf = pd.concat([principalDf, data_ori[['date']]], axis = 1)
    communicate_ratio = pca.explained_variance_ratio_  # 降维后的各主成分的方差值占总方差值的比例，即方差贡献率

    # 得到转换的关系(系数) 因子载荷矩阵？
    weight = pca.components_
    # 计算个主成分得分
    x_pca = np.dot(x, weight.T)  # 原始数据X与选取的特征向量相乘，得到k个主成分，即降维结果，即pca函数的调用结果，保存在X_pca中。
    score = pd.Series(x_pca.sum(axis=1))  # 项目得分
    # scores = score.sort_values(ascending=False)  # 得分和排序
    # print("各个项目得分和排名：\n", scores)
    pca_scores = dict(zip(data_ori['project_id'], score))
    return pca_scores


def get_weidu(x):
    # 查看降维后的维数
    # 选择降到多少维度比较合适
    pca = PCA(n_components=6)
    principalComponents = pca.fit_transform(x)
    ratio = pca.explained_variance_ratio_  # 降维后的各主成分的方差值占总方差值的比例，即方差贡献率
    print(ratio)  # 维度占比
    count = 0
    i = 1
    for r in ratio:
        count = count + r
        if count > 0.8:
            return i
        else:
            i += 1
    # 进行可视化
    # importance = pca.explained_variance_ratio_
    # plt.scatter(range(1,11),importance)
    # plt.plot(range(1,11),importance)
    # plt.title('Scree Plot')
    # plt.xlabel('Factors')
    # plt.ylabel('Eigenvalue')
    # plt.grid()
    # plt.show()
    return i

if __name__ == '__main__':
    file_path = r'E:/work/ai_work/oss_health/assess_data_90/ten_16.csv'
    pca_scores(file_path)