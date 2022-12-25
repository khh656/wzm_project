# 将所有项目原始数据按照每30天计算一个得分，合并到一起
from src.utils.util import list_to_csv
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

def EWM_scores_pros(ori_dir, des_path_file):
    '''熵值法计算变量的权重 30天的数据计算一次'''
    beta = 1e-8
    windows = 30
    i = 0
    columns_pro = []
    score_list = []
    while tqdm(i <= 50*30):
        score_pro = []
        for file in os.listdir(ori_dir):
            file_path = ori_dir + file
            if file_path.endswith('.csv'):
                # data_ori = pd.read_csv(file_path)[1:]
                # 0712 start
                data_ori = pd.read_csv(file_path)
                # data_res = data_ori.drop(columns=['committer_id', 'project_id', 'date'])
                data_res = data_ori.drop(columns=['date'])
                # 标准化
                label = data_res.columns
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
                # 计算得分
                scores = np.dot(weight, x.T).sum()
                # if i == 0:
                #     columns_pro.append(data_ori['project_id'].iloc[0])
                col_name = file.split('.')[0]
                columns_pro.append(col_name)
                score_pro.append(scores)
        score_list.append(score_pro)
        i = i + windows
    # df_d = pd.DataFrame(weight_list, columns=column_name)
    # df_d.to_csv(des_path_file, index=None)
    list_to_csv(columns_pro[:int((len(columns_pro)/51))], score_list, des_path_file)

def plot_plzft(path, des_path):
    data = pd.read_csv(path)
    data_T = data.T / 10
    months = [1, 10, 20, 30, 40, 49]
    for month in months:
        plt.figure()  # 初始化一张图range=(0,width),
        x = data_T[month].tolist()
        # bins_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        list_width = np.arange(0, 0.7, 0.02)
        bins_list = list_width.tolist()
        width = len(list_width) - 1
        count, bins, patches = plt.hist(x, bins=bins_list, color='b', alpha=0.8)

        for a, b in zip(bins, count):
            plt.text(a, b, int(b), ha='center', va='bottom')

        # print("数量:", count)
        # print("bins:", bins)
        # print('patches:', patches)

        plt.title('Distribution Histogram of Project Health in %s-th Month' % month)
        plt.xlabel('Health')
        plt.ylabel('Project Numbers')

        plt.xticks(np.arange(0, 0.7, 0.1))

        plt.plot(bins[0:width] + ((bins[1] - bins[0]) / 2.0), count, color='red', linestyle='--')  # 利用返回值来绘制区间中点连线
        des_file = des_path+"第"+str(month)+"月频率分布直方图.pdf"
        plt.savefig(des_file)
        plt.show()
if __name__ == '__main__':
    # 抽取数据放入csv，准备画图使用。运行一次即可。
    # ori_dir = r'E:\work\ai_work\oss_health\process_data_0604\\'
    # des_path_file = r'E:/work/ai_work/oss_health/result_weight_score_0610/all_file_scores.csv'
    # EWM_scores_pros(ori_dir, des_path_file)
    # 画频率直方图。
    path_file = r'E:/work/ai_work/oss_health/result_weight_score_0610/all_file_scores.csv'
    dest_path = r'E:\work\ai_work\oss_health\result_picture_0607\\'
    plot_plzft(path_file, dest_path)