import csv
# dic = {8024: 18.891547, 20088: -3.4166,51900: 8.515,  31738: -2.973,  23681: -3.92, 48168: 0.20713461647443837, 253850: -4.549309348541141, 323872: -5.311657932485378, 6624: -4.45110283760075, 235442: -3.1115771918881876}
# path = r'/result/test2.csv'
import pandas as pd
import os
# data.insert(0,"date",time_series)
# 字典存入csv
def dic_csv_score(dic, csv_path):
    f = open(csv_path, mode='w+', encoding='utf-8', newline='')  #  mode='a'追加
    columns = dic.keys()
    csv_writer = csv.DictWriter(f, fieldnames=columns)  # 列名
    csv_writer.writeheader()  # 列名写入csv
    csv_writer.writerow(dic)  # 数据写入csv文件

# result_scores
def csv_merge(dir_path, end_path_file):
    dir_list = []
    for file in os.listdir(dir_path):
        file_name = dir_path + file
        name = file.split('.')[0]
        df = pd.read_csv(file_name)
        df.insert(0, "project", name)  # 将方法名字加进去
        dir_list.append(df)
    res = pd.concat(dir_list, axis=0)
    res.to_csv(end_path_file, index=False)
    return res

def csv_T(ori_file,end_file):
    data = pd.read_csv(ori_file)
    data_T = data.T
    data_T.to_csv(end_file,header=None)

# list -- csv
def list_to_csv(columns,list_data, csv_file):

    df = pd.DataFrame(columns=columns, data=list_data)
    df.to_csv(csv_file, encoding="utf_8_sig", header=1, index=0)
    # df.to_csv(csv_file, mode="a", encoding="utf_8_sig", header=1, index=0)

# 计算信度
def caculate_xin(path):
    data = pd.read_csv(path, index_col='project')
    # data = data_ori.T
    # data = softmax(data)
    # data_ori = pd.read_csv(path,index_col='project')
    # data = data_ori.drop('Project', axis=1)
    xin_dic = {}

    for col in data.columns:
        data_mid = data.drop(col, axis=1)
        count = 0
        for col_j in data_mid.columns:
            kendall_weight = spearman(data[col].values, data[col_j].values)
            # kendall_weight = pearson(data[col].values, data[col_j].values)
            # kendall_weight = kendall(data[col].values, data[col_j].values)
            # kendall_weight = data[col].corr(data[col_j], method='kendall')
            count = count + kendall_weight
        xin_dic[col] = count/len(data_mid.columns)
    return xin_dic

# 用方法的排名计算评价方法的相似度;n为评价方法的数量
'''
{'NO_ewm': 0.9381818181818182,
 'NO_ewm_top': 0.9187878787878787,
 'NO_fa': 0.9478787878787879,
 'NO_gra': 0.9624242424242425,
 'NO_pca': 0.8703030303030304,
 'NO_top': 0.9042424242424243}
'''
def caculate_similary(path,n):

    data_ori = pd.read_csv(path)
    data = data_ori.drop('project', axis=1)
    # 皮尔逊相关系数
    spe = data.corr()
    spe_res = (spe.sum()-1)/(n-1)
    return dict(zip(data.columns, spe_res))

# 用方法的排名计算评价方法的离散度; n为评价方法的数量
'''
{'NO_ewm': 3.4,
 'NO_ewm_top': 4.0,
 'NO_fa': 4.0,
 'NO_gra': 4.2,
 'NO_pca': 3.8,
 'NO_top': 4.2}
'''
import numpy as np
def cac_Dispersion(path, n):
    data_ori = pd.read_csv(path)
    data = data_ori.drop('project', axis=1)
    lsd_dic = {}
    for col in data.columns:
        data_mid = data.drop(col, axis=1)
        count = 0
        for col_j in data_mid.columns:
            cha_no = data[col].values - data[col_j].values
            # 统计0的个数
            zero_array = np.where(cha_no, 0, 1)
            count = count + np.sum(zero_array)
        lsd_dic[col] = float(count / (n - 1))
    return lsd_dic


# 平均离差
def cac_mean_dis(path):
    data = pd.read_csv(path, index_col="project")
    pro_mean = data.mean(axis=1).values
    mean_cha = {}

    for col in data.columns:
        cha = abs(data[col] - pro_mean)
        mean_d_sum = cha.sum()
        mean_d = mean_d_sum / len(data)
        mean_cha[col] = mean_d
    return mean_cha

import math
from itertools import combinations
# 肯德尔系数、皮尔逊系数、斯皮尔曼系数
#Pearson algorithm
def pearson(x, y):
    assert len(x) == len(y) > 0
    q = lambda n: len(n) * sum(map(lambda i: i ** 2, n)) - (sum(n) ** 2)
    return (len(x) * sum(map(lambda a: a[0] * a[1], zip(x, y))) - sum(x) * sum(y)) / math.sqrt(q(x) * q(y))

#Spearman algorithm
def spearman(x, y):
    assert len(x) == len(y) > 0
    q = lambda n: map(lambda val: sorted(n).index(val) + 1, n)
    d = sum(map(lambda x, y: (x - y) ** 2, q(x), q(y)))
    return 1.0 - 6.0 * d / float(len(x) * (len(y) ** 2 - 1.0))

#Kendall algorithm
def kendall(x, y):
    assert len(x) == len(y) > 0
    c = 0  # concordant count
    d = 0  # discordant count
    t = 0  # tied count
    for (i, j) in combinations(range(len(x)), 2):
        s = (float(x[i]) - float(x[j])) * (float(y[i]) - float(y[j]))
        if s:
            c += 1
            d += 1
            if s > 0:
                t += 1
            elif s < 0:
                t -= 1
        else:
            if x[i] - x[j]:
                c += 1
            elif y[i] - y[j]:
                d += 1
    return t / math.sqrt(c * d)

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)
if __name__ == '__main__':
#     pd1 = pd.read_csv(r"E:\work\ai_work\oss_health\result\test.csv")
#     pd2 = pd.read_csv(r"E:\work\ai_work\oss_health\result\test2.csv")
#     res = pd.concat([pd1, pd2], axis=0)
#     print(res)

# read in file
    GSM = [6, 9, 3, 12, 8, 7, 10, 11, 2, 5, 4, 1]
    LGC = [6, 9, 3, 12, 7, 8, 11, 10, 2, 4, 5, 1]

    kendall_test = kendall(GSM, LGC)
    pearson_test = pearson(GSM, LGC)
    spearman_test = spearman(GSM, LGC)

    print("肯德尔系数：", kendall_test)
    print("皮尔逊系数：", pearson_test)
    print("斯皮尔曼系数：", spearman_test)