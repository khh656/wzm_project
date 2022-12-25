import pandas as pd
import numpy as np
import math
from numpy import array
import copy

# 熵值法函数 求指标权重 用这个 https://blog.51cto.com/u_15127589/2806339
def EWM_scores(file_path):
    '''熵值法计算变量的权重 '''
    data_ori = pd.read_csv(file_path)
    x = data_ori.drop(columns=['project_id', 'date'])
    # 标准化
    label = x.columns
    for key in label:
        # 得到列最大和最小
        maxium = np.max(x[key], axis=0)
        minium = np.min(x[key], axis=0)
        beta = 1e-8
        if key == 'issue':  # 反向
            x[key] = x.loc[:, key].apply(lambda x: ((maxium - x) / (maxium - minium+beta)))
        else:  # 正向
            x[key] = x.loc[:, key].apply(lambda x: ((x - minium) / (maxium - minium+beta)))

    m, n = x.shape
    # data = x.as_matrix(columns=None)  # 将dataframe格式转化为matrix格式
    data = x  # 将dataframe格式转化为matrix格式
    k = 1 / np.log(m)
    yij = data.sum(axis=0)

    # 计算pij
    pij = data / yij

    # 求各指标的信息熵
    mid1 = pij * np.log(pij)
    mid = np.nan_to_num(mid1)  # 使用0代替数组x中的nan元素，使用有限的数字代替inf元素(默认行为)
    ej = -k * (mid.sum(axis=0))
    # 计算每种指标的信息熵; 求各指标权重
    weight = (1 - ej) / np.sum(1 - ej)
    # 计算得分
    scores = (weight * x).sum(axis=1)
    ewm_scores_res = dict(zip(data_ori['project_id'],scores))
    print(scores)
    return ewm_scores_res, weight


if __name__ == '__main__':

    # data_ori2 = pd.read_csv(r'E:\work\ai_work\oss_health\assess_data_90\ten_6.csv')
    #
    # data1 = data_ori2.drop(columns=['project_id', 'date'])
    #
    # # # 求熵权法
    # df = pd.DataFrame(data1)
    # df.dropna()
    # label = df.columns
    # path = r'E:\work\ai_work\oss_health\assess_data_90\ten_6.csv'
    path = r'E:\work\ai_work\oss_health\data_30_10_series\project_772.csv'
    res_score = EWM_scores(path)  # 调用cal_weight
    # w.index = df.columns
    # w.columns = ['weight']  # 每一个指标的权重
    # print(w)
    print("项目得分：", res_score)
