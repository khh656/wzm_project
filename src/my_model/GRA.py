import pandas as pd
import numpy as np
# 暂用这个灰色关联分析法,评价方法对比的时候使用这个
from src.my_model.GRA_Health_Result import dimensionlessProcessing

'''

def gra_score(data_path):
    # 导入数据
    data_ori = pd.read_csv(data_path)
    x = data_ori.drop(columns=['project_id', 'date'])

    # csv_data = pd.read_csv(csv_file, low_memory=False)  # 防止弹出警告
    x_max = x.max(axis=0)
    x = x.T

    # 数据均值化处理
    x_mean = x.mean(axis=1)
    for i in range(x.index.size):
        x.iloc[i, :] = x.iloc[i, :]/x_mean[i]

    # 提取参考队列和比较队列
    ck = x_max
    cp = x
    # 比较队列与参考队列相减
    t = pd.DataFrame()
    for j in range(cp.index.size):
        temp = pd.Series(cp.iloc[:, j]-ck)
        t = t.append(temp, ignore_index=True)

    #求最大差和最小差
    mmax=t.abs().max().max()
    mmin=t.abs().min().min()
    rho=0.5
    #求关联系数
    ksi = ((mmin+rho*mmax)/(abs(t)+rho*mmax))
    #求关联度
    r = ksi.sum(axis=1)/ksi.columns.size
    #关联度排序
    # result = r.sort_values(ascending=False)
    dic_gra = dict(zip(data_ori['project_id'],r))
    return dic_gra
'''

# 暂用这个灰色关联分析法,评价方法对比的时候使用这个
def gra_score(data_path):
    # 导入数据
    data_ori = pd.read_csv(data_path)
    x = data_ori.drop(columns=['project_id', 'date'])
    # 数据均值化处理
    x_mean = dimensionlessProcessing(x)

    # csv_data = pd.read_csv(csv_file, low_memory=False)  # 防止弹出警告
    x_max = x_mean.max(axis=0)
    x_T = x_mean.T

    # 数据均值化处理
    # x_mean = x.mean(axis=1)
    # for i in range(x.index.size):
    #     x.iloc[i, :] = x.iloc[i, :]/x_mean[i]

    # 提取参考队列和比较队列
    ck = x_max
    cp = x_T
    # 比较队列与参考队列相减
    t = pd.DataFrame()
    for j in range(cp.columns.size):
        temp = pd.Series(cp.iloc[:, j]-ck)
        t = t.append(temp, ignore_index=True)

    #求最大差和最小差
    mmax=t.abs().max().max()
    mmin=t.abs().min().min()
    rho = 0.5
    #求关联系数
    beta = 1e-8
    ksi = ((mmin+rho*mmax)/(abs(t)+rho*mmax+beta))
    #求关联度--取均值就是关联度
    # r = ksi.sum(axis=1)/ksi.columns.size
    r = ksi.sum(axis=0)/len(ksi)
    #关联度排序
    # result = r.sort_values(ascending=False)
    # 综合评价
    # res = np.dot(r, cp)
    res_score = np.dot(r, x.T)
    res = res_score/res_score.sum()
    dic_gra = dict(zip(data_ori['project_id'],res))

    return dic_gra

if __name__ == '__main__':
    path = r'E:\work\ai_work\oss_health\assess_data_90\ten_0.csv'
    a = gra_score(path)
    print(a)