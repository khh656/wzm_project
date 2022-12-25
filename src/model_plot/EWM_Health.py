import pandas as pd
import numpy as np
import os
# 熵值法函数 求指标权重 用这个 https://blog.51cto.com/u_15127589/2806339
from src.utils.util import dic_csv_score

# 计算每个月616个项目的得分情况
def EWM_scores_alldata(dir_path, des_path):
    '''熵值法计算变量的权重 '''
    beta=1e-8

    for file in os.listdir(dir_path):
        file_path = dir_path+file
        data_ori = pd.read_csv(file_path)
        x = data_ori.drop(columns=['project_id', 'date'])
        # 标准化
        label = x.columns
        for key in label:
            # 得到列最大和最小
            maxium = np.max(x[key], axis=0)
            minium = np.min(x[key], axis=0)

            if key == 'issue':  # 反向
                x[key] = x.loc[:, key].apply(lambda x: ((maxium - x) / (maxium - minium+beta)))
            elif key == 'req_opened':
                x[key] = x.loc[:, key].apply(lambda x: ((maxium - x) / (maxium - minium + beta)))
            else:  # 正向
                x[key] = x.loc[:, key].apply(lambda x: ((x - minium) / (maxium - minium+beta)))

        m, n = x.shape
        # data = x.as_matrix(columns=None)  # 将dataframe格式转化为matrix格式
        data = x  # 将dataframe格式转化为matrix格式
        k = 1 / (np.log(m)+beta)
        yij = data.sum(axis=0)

        # 计算pij
        pij = data / (yij+beta)

        # 求各指标的信息熵
        mid1 = pij * np.log(pij)
        mid = np.nan_to_num(mid1)  # 使用0代替数组x中的nan元素，使用有限的数字代替inf元素(默认行为)
        ej = -k * (mid.sum(axis=0))
        # 计算每种指标的信息熵; 求各指标权重
        weight = (1 - ej) / (np.sum(1 - ej)+beta)
        # 计算得分
        scores = (weight * x).sum(axis=1)
        # scores = np.dot(weight, x.T).mean()
        # score_pro[data_ori['project_id'][0]] = scores
        ewm_scores_res = dict(zip(data_ori['project_id'], scores))  # {961: 0.019997059669618435}
        file_name = des_path+file
        dic_csv_score(ewm_scores_res, file_name)

# 计算一个项目每个月的得分、权重情况
def EWM_scores_one(dir_path, des_path):
    '''熵值法计算变量的权重 '''
    beta=1e-8

    for file in os.listdir(dir_path):
        file_path = dir_path+file
        data_ori = pd.read_csv(file_path)
        # x = data_ori.drop(columns=['project_id', 'date'])
        x = data_ori.drop(columns=['date'])
        column_name = x.columns
        # 标准化
        label = x.columns
        for key in label:
            # 得到列最大和最小
            maxium = np.max(x[key], axis=0)
            minium = np.min(x[key], axis=0)

            if key == 'issue':  # 反向
                x[key] = x.loc[:, key].apply(lambda x: ((maxium - x) / (maxium - minium+beta)))
            elif key == 'req_opened':
                x[key] = x.loc[:, key].apply(lambda x: ((maxium - x) / (maxium - minium + beta)))
            else:  # 正向
                x[key] = x.loc[:, key].apply(lambda x: ((x - minium) / (maxium - minium+beta)))

        m, n = x.shape
        # data = x.as_matrix(columns=None)  # 将dataframe格式转化为matrix格式
        data = x  # 将dataframe格式转化为matrix格式
        k = 1 / (np.log(m)+beta)
        yij = data.sum(axis=0)

        # 计算pij,第j个指标下第i个样本占该指标的比重
        pij = data / (yij+beta)

        # 求各指标的信息熵
        mid1 = pij * np.log(pij)
        mid = np.nan_to_num(mid1)  # 使用0代替数组x中的nan元素，使用有限的数字代替inf元素(默认行为)
        ej = -k * (mid.sum(axis=0))  # 计算第j个指标的熵值
        # 计算每种指标的信息熵; 求各指标权重
        weight = (1 - ej) / (np.sum(1 - ej)+beta)
        # 计算得分
        scores = (weight * x).sum(axis=1)
        # scores = np.dot(weight, x.T).mean()
        # score_pro[data_ori['project_id'][0]] = scores
        # 每个指标，每个时刻的权重值
        w_time = -k*mid
        file_name = des_path + file
        df_d = pd.DataFrame(w_time, columns=column_name)
        df_d.insert(0, 'date', data_ori['date'])
        # df_d.insert(1, 'project_id', data_ori['project_id'])
        filename = file.split('.')[0]
        df_d.insert(1, 'project_id', filename)
        # df_d.insert(11, 'score', scores)
        df_d.insert(9, 'score', scores)
        df_d.to_csv(file_name, index=None)

        # ewm_scores_res = dict(zip(data_ori['project_id'], scores))  # {961: 0.019997059669618435}
        # file_name = des_path + file
        # dic_csv_score(ewm_scores_res, file_name)


# 计算一个项目每个月权重情况
def EWM_scores_weight(one_path_file, des_path_file):
    '''熵值法计算变量的权重 '''
    beta = 1e-8
    windows = 30
    i = 0

    data_ori = pd.read_csv(one_path_file)[1:]
    # data_res = data_ori.drop(columns=['committer_id','project_id', 'date'])
    data_res = data_ori.drop(columns=[ 'date'])
    column_name = data_res.columns
    # 标准化
    label = data_res.columns
    len_data = len(data_res)
    weight_list = []
    while i <= (len_data-windows):
        x = data_res.iloc[i:i+windows, :]
        for key in label:
            # 得到列最大和最小
            maxium = np.max(x[key], axis=0)
            minium = np.min(x[key], axis=0)

            if key == 'issue':  # 反向
                x[key] = x.loc[:, key].apply(lambda x: ((maxium - x) / (maxium - minium + beta)))
            elif key == 'req_opened':
                x[key] = x.loc[:, key].apply(lambda x: ((maxium - x) / (maxium - minium + beta)))
            else:  # 正向
                x[key] = x.loc[:, key].apply(lambda x: ((x - minium) / (maxium - minium + beta)))

        m, n = x.shape
        # data = x.as_matrix(columns=None)  # 将dataframe格式转化为matrix格式
        data = x  # 将dataframe格式转化为matrix格式
        k = 1 / (np.log(m) + beta)
        yij = data.sum(axis=0)

        # 计算pij,第j个指标下第i个样本占该指标的比重
        pij = data / (yij + beta)

        # 求各指标的信息熵
        mid1 = pij * np.log(pij)
        mid = np.nan_to_num(mid1)  # 使用0代替数组x中的nan元素，使用有限的数字代替inf元素(默认行为)
        ej = -k * (mid.sum(axis=0))  # 计算第j个指标的熵值
        # 计算每种指标的信息熵; 求各指标权重
        weight = (1 - ej) / (np.sum(1 - ej) + beta)
        weight_list.append(weight)
        i = i + windows
        # 计算得分
        # scores = (weight * x).sum(axis=1)
    df_d = pd.DataFrame(weight_list, columns=column_name)
    df_d.to_csv(des_path_file, index=None)


if __name__ == '__main__':
    # 文件夹下的项目提取权重和得分
    import os
    # ori_path = r'E:\work\ai_work\oss_health\data_30_10_EWM\\'
    # des_path = r'E:/work/ai_work/oss_health/result_weight_score_0610/'
    # EWM_scores_one(ori_path, des_path)

    # 选取一个项目，计算指标与权重关系
    one_path_file = r'E:\work\ai_work\oss_health\data_spyder\data_merge\redis.csv'
    des_path_file = r'E:/work/ai_work/oss_health/result_weight_score_0610/one_file_weight2.csv'
    EWM_scores_weight(one_path_file, des_path_file)

# E:\work\ai_work\oss_health\data_30_10_series