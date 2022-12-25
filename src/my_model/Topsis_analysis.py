# -*- coding: utf-8 -*-
'''

用这个
'''
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import pandas as pd
import numpy as np
def Topsis_scores(file_path, weight, postive):
    data_ori = pd.read_csv(file_path)
    data = data_ori.drop(columns=['project_id', 'date'])

    A = data.values  # numpy.ndarray'
    B = np.zeros(A.shape)
    for i in range(0, A.shape[1]):
        mid = A[:, i]
        beta = 1e-5
        B[:, i] = mid/(np.linalg.norm(mid)+beta)  # 归一化
    
    w = np.array(weight)
    C = B*w.T  # 加权规范矩阵
    Cstar = C.max(axis=0)  # 先按列取最大值，求正理想解：
    print(Cstar)
    for i in postive:
        Cstar[i] = C[:, i].min()  # 但由于第四个指标是负向指标，即值越小越好，所以我们的正理想解的第四个指标应该取最小值：
    
    C0 = C.min(axis=0)  # 按列取最小值，求负理想解
    
    for i in postive:
        C0[i] = C[:, i].max()  # 取第i个指标是负理想指标（值越大越好）
    # print(C0)
    # 求各个样本到正负理想解的距离
    Sstar = np.zeros((1,C.shape[0]))
    S0 = np.zeros((1,C.shape[0]))
    for i in range(0, C.shape[0]):
        Sstar[:, i] = np.linalg.norm(C[i, :]-Cstar)  # 求各样本到正理想解的距离
        S0[:, i] = np.linalg.norm(C[i, :]-C0)  # 求各样本到负理想解的距离
    # 再根据各样本到正负理想解的距离计算每个待评价样本的评价参考值：
    f = S0/(S0 + Sstar)
    # 我们需要根据评价参考值，从大到小进行排序，展示出来即可
    ind = data.index.values
    result = np.insert(f.T, 0, values = ind, axis = 1)

    # return pd.DataFrame(result[np.lexsort(-result.T)], columns = ['project', 'score'])
    res = pd.DataFrame(result, columns=['project', 'score'])
    topsis_scores_dic = dict(zip(data_ori['project_id'], res['score']))
    return topsis_scores_dic


if __name__ == '__main__':
    file_path = r'E:\work\ai_work\oss_health\assess_data_90\ten_0.csv'
    topsis_scores_dic = Topsis_scores(file_path,
                    weight=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                    postive=[1])  # 根据Cstar结果选最小值所对应的索引，作为负想指标