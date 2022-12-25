import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# 健康性频率直方图
def plto_plzft(o_path, des_path, num):
    data = pd.read_csv(o_path, header=None)
    data_T = data.T

    plt.figure()  # 初始化一张图range=(0,width),
    x = data_T[1].tolist()
    # bins_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    list_width = np.arange(0, 0.7, 0.02)
    bins_list = list_width.tolist()
    width = len(list_width)-1
    count, bins, patches = plt.hist(x, bins=bins_list, color='b', alpha=0.8)

    for a, b in zip(bins, count):
        plt.text(a, b, int(b), ha='center', va='bottom')

    # print("数量:", count)
    # print("bins:", bins)
    # print('patches:', patches)

    plt.title('Distribution Histogram of Project Health in %s-th Month' % num)
    plt.xlabel('Health')
    plt.ylabel('Project Numbers')

    plt.xticks(np.arange(0, 0.8, 0.1))

    plt.plot(bins[0:width] + ((bins[1] - bins[0]) / 2.0), count, color='red', linestyle='--')  # 利用返回值来绘制区间中点连线
    des_file = des_path+"第"+num+"月频率分布直方图.pdf"
    plt.savefig(des_file)
    plt.show()

if __name__ == '__main__':
    import os
    ori_path = r'E:/work/ai_work/oss_health/ewm_scores_alldata/'
    des_path = r'E:/work/ai_work/oss_health/result_picture_0607/'
    for file in os.listdir(ori_path):
        num = file.split('.')[0]
        if file.endswith('.csv'):
            path = ori_path+file
            plto_plzft(path, des_path, num)


