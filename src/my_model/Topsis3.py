import numpy as np

import pandas as pd
'''
topsis综合评价法即根据有限个评价对象与理想化目标的接近程度进行排序的方法，是在现有的对象中进行相对优劣的评价，是一种逼近于理想解的排序法。TOPSIS法是多目标决策分析中一种常用的有效方法，又称为优劣解距离法。
TOPSIS法是一种常用的综合评价方法，能充分利用原始数据的信息，其结果能精确地反映各评价方案之间的差距。
'''

# 从excel文件中读取数据
def read(file):
    # wb = xlrd.open_workbook(filename=file)  # 打开文件
    df = pd.read_csv(file, encoding='gb2312')
    return df

# 极小型指标 -> 极大型指标
def dataDirection_1(datas):
    return np.max(datas) - datas  # 套公式

# 中间型指标 -> 极大型指标
def dataDirection_2(datas, x_best):
    temp_datas = datas - x_best
    M = np.max(abs(temp_datas))
    answer_datas = 1 - abs(datas - x_best) / M  # 套公式
    return answer_datas

# 正向化矩阵标准化
def temp2(datas):
    K = np.power(np.sum(pow(datas, 2), axis=1), 0.5)
    for i in range(0, K.size):
        for j in range(0, datas[i].size):
            datas[i, j] = datas[i, j] / K[i]  # 套用矩阵标准化的公式
    return datas


# 计算得分并归一化
def temp3(answer2):
    list_max = np.array(
        [np.max(answer2[0, :]), np.max(answer2[1, :]), np.max(answer2[2, :])])  # 获取每一列的最大值
    list_min = np.array(
        [np.min(answer2[0, :]), np.min(answer2[1, :]), np.min(answer2[2, :])])  # 获取每一列的最小值
    max_list = []  # 存放第i个评价对象与最大值的距离
    min_list = []  # 存放第i个评价对象与最小值的距离
    answer_list = []  # 存放评价对象的未归一化得分
    for k in range(0, np.size(answer2, axis=1)):  # 遍历每一列数据
        max_sum = 0
        min_sum = 0
        for q in range(0, 3):  # 有三个指标
            max_sum += np.power(answer2[q, k] - list_max[q], 2)  # 按每一列计算Di+
            min_sum += np.power(answer2[q, k] - list_min[q], 2)  # 按每一列计算Di-
        max_list.append(pow(max_sum, 0.5))
        min_list.append(pow(min_sum, 0.5))
        answer_list.append(min_list[k] / (min_list[k] + max_list[k]))  # 套用计算得分的公式 Si = (Di-) / ((Di+) +(Di-))
        max_sum = 0
        min_sum = 0
    answer = np.array(answer_list)  # 得分归一化
    return (answer / np.sum(answer))


def main():
    file = r'E:\work\ai_work\oss_health\des_data\project_390.csv'
    answer1 = pd.read_csv(file, encoding='gb2312')
    answer1 = answer1.drop(columns=['project_id', 'date', 'committer_id'])
    answer2 = []
    for i in range(0, 3):  # 按照不同的列，根据不同的指标转换为极大型指标，因为只有四列
        answer = None
        if (i == 0):  # 本来就是极大型指标，不用转换
            answer = answer1[0]
        elif (i == 1):  # 中间型指标
            answer = dataDirection_2(answer1[1], 1)
        elif (i == 2):  # 极小型指标
            answer = dataDirection_1(answer1[2])
        answer2.append(answer)
    answer2 = np.array(answer2)  # 将list转换为numpy数组
    answer3 = temp2(answer2)  # 数组正向化
    answer4 = temp3(answer3)  # 标准化处理去钢
    data = pd.DataFrame(answer4)  # 计算得分
    print(data)

main()

