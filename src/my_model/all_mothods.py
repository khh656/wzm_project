
import numpy as np
import pandas as pd
import numpy.linalg as nlg
from factor_analyzer import calculate_kmo, calculate_bartlett_sphericity, FactorAnalyzer

weight = []

def ahp(data, rows, columns):
    f_score = pd.read_csv(data, 0)	# 专家对各项指标的重要度评分存储在feature_score中
    # print(评分：f_score)
    f_score = f_score.prod(axis=1)
    f_score = np.power(f_score, 1/2)
    W = f_score/sum(f_score)			# 计算权重
    # print("权重：\n", W)
    score = data
    for i in range(rows):
        for j in range(columns):
            score[i][j] = score[i][j]*W[i]		# 计算分数
    score = score.sum(axis=0)
    weight.append(W.tolist())
    return score

def E_j_fun(data, rows, columns):  #计算熵值
    E = np.array([[None] * columns for i in range(rows)])   # 新建空矩阵
    for i in range(rows):
        for j in range(columns):
            if data[i][j] == 0:
                e_ij = 0.0
            else:
                P_ij = data[i][j] / data.sum(axis=0)[j]  # 计算比重(列求和)
                e_ij = (-1 / np.log(rows)) * P_ij * np.log(P_ij)
            E[i][j] = e_ij
    # print(E)
    E_j=E.sum(axis=0)       # 求出每列信息熵（指标）列求和
    return E_j

def topsis(data, rows, columns):        # topsis综合评价
    Z_ij = np.array([[None] * columns for i in range(rows)])   # 新建空矩阵(加权标准化矩阵)
    E_j = E_j_fun(data, rows, columns)       # 第j个指标的信息熵
    # print(E_j)
    G_j = 1-E_j               # 信息差异度矩阵
    # print(G_j)
    W_j = G_j/sum(G_j)        # 计算权重
    for i in range(rows):
        for j in range(columns):
            Z_ij[i][j] = data[i][j] * W_j[j]
    Imax_j = Z_ij.max(axis=0)  # 最优解
    Imin_j = Z_ij.min(axis=0)  # 最劣解
    Dmax_ij = np.array([[None] * columns for i in range(rows)])
    Dmin_ij = np.array([[None] * columns for i in range(rows)])
    for i in range(rows):
        for j in range(columns):
            Dmax_ij[i][j] = (Imax_j[j] - Z_ij[i][j]) ** 2
            Dmin_ij[i][j] = (Imin_j[j] - Z_ij[i][j]) ** 2
    Dmax_i = Dmax_ij.sum(axis=1) ** 0.5  # 最优解欧氏距离
    Dmin_i = Dmin_ij.sum(axis=1) ** 0.5  # 最劣解欧氏距离
    C_i = Dmin_i / (Dmax_i + Dmin_i)  # 综合评价值
    weight.append(W_j.tolist())
    # print(C_i)
    return C_i

def E_evalution(data, rows, columns):        # 熵权法综合评价
    Z_ij = np.array([[None] * columns for i in range(rows)])  # 新建空矩阵(加权标准化矩阵)
    E_j = E_j_fun(data, rows, columns)  # 第j个指标的信息熵
    G_j = 1 - E_j  # 信息差异度矩阵
    W_j = G_j / (columns - sum(G_j))  # 计算权重
    for i in range(rows):
        for j in range(columns):
            Z_ij[i][j] = data[i][j] * W_j[j]
    ret = Z_ij.sum(axis=1)
    weight.append(W_j.tolist())
    # print("ret", ret)
    return ret

def critic(data, rows, columns):
    Z_ij = np.array([[None] * rows for i in range(columns)])
    data_std = np.std(data, axis=1, ddof=1)
    # print(data_std)
    data_rela = np.corrcoef(data)
    data_rela = data_rela.sum(axis=1)
    # print(data_std, "\n", data_rela)        # 样本标准差(n-1)
    C_i = data_rela * data_std              # 矩阵点乘
    W_i = C_i/sum(C_i)
    # print(W_i)
    for i in range(columns):
        for j in range(rows):
            Z_ij[i][j] = data[i][j] * W_i[i]
    ret = Z_ij.sum(axis=0)
    # print(ret)
    weight.append(W_i.tolist())
    return ret

def factor(data, columns, rows):
    df2_corr = np.corrcoef(data.T)  # 皮尔逊相关系数
    kmo = calculate_kmo(data)  # kmo值要大于0.7
    bartlett = calculate_bartlett_sphericity(data)  # bartlett球形度检验p值要小于0.05
    print('kmo:{},bartlett:{}'.format(kmo[1], bartlett))
    # 使用最大方差法旋转因子载荷矩阵
    fa = FactorAnalyzer(n_factors=rows, rotation='varimax', method='principal', impute='mean')
    fa.fit(data)
    fa_sd = fa.get_factor_variance()        # 得到贡献率fa_sd[1]
    fa_rotate = FactorAnalyzer(rotation='varimax', n_factors=rows, method='principal')
    fa_rotate.fit(data)
    # 查看旋转后的因子载荷
    # print("\n旋转后的因子载荷矩阵:\n", fa_rotate.loadings_)
    # 因子得分（回归方法）（系数矩阵的逆乘以因子载荷矩阵）
    X1 = np.mat(df2_corr)
    X1 = nlg.inv(X1)
    factor_score = np.dot(X1, fa_rotate.loadings_)
    # print("\n因子得分（每个样本的因子权重）：\n", factor_score)
    fa_t_score = np.dot(np.matrix(data), np.matrix(factor_score))
    # print("\n样本的因子得分：\n", pd.DataFrame(fa_t_score))
    # 综合得分(加权计算）
    fa_t_score = np.dot(fa_t_score, fa_sd[1]) / sum(fa_sd[1])
    weight .append((fa_sd[1] / sum(fa_sd[1])).tolist())
    return np.array(fa_t_score)[0]


#　1.1数据转化
def MinMax(df, cols):
    df_n = df.copy()
    for col in cols:
        ma = df_n[col].max()
        mi = df_n[col].min()
        df_n[col] = (df_n[col] - mi) / (ma - mi)
    return (df_n)

def main():
    path = 'shang_data.csv'
    data = pd.read_csv(path,header=None,skiprows=1)      # 读取excel并保存
    # print("源数据", data)
    for i in range(0, 5):
        data[i] = 1-data[i]/data[i].max()
    cols = data.columns
    Standard_data = MinMax(data,cols)      # 对每一列最大最小归一化数据
    rows = Standard_data.shape[0]
    columns = Standard_data.shape[1]
    # 客观
    ret_topsis = topsis(Standard_data.T, columns, rows)
    ret_E = E_evalution(Standard_data.T, columns, rows)
    ret_critic = critic(Standard_data, columns, rows)
    ret_factor = factor(Standard_data.T, columns, rows)
    # 主观
    ret_ahp = ahp(Standard_data, rows, columns)
    # 合并
    result = np.dstack((ret_ahp, ret_topsis, ret_E, ret_critic, ret_factor))
    print(result)

if __name__ == '__main__':
    main()
