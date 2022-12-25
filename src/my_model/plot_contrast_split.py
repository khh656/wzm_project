import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#用于显示中文
import matplotlib
import os
matplotlib.rcParams['font.family'] = 'SimHei'

# 对比指标
def plot_fac(path, name, despath):
    data_ori = pd.read_csv(path, index_col="method").drop(columns=['project_id', 'mape','r2'])
    data = data_ori.T.drop(columns=["Transformer"]).T
    x_tick = data.index.to_list()
    colors = ['blue', 'red', 'green', 'black', 'cyan', 'magenta', 'olive', 'darksalmon']
    x = [i for i in range(len(x_tick))]

    for col in data.columns:
        y = data[col]
        # 绘制柱状图
        plt.figure(figsize=(10,15))
        # plt.figure(figsize=(10, 8))
        plt.bar(x, y, width=0.5, align="center", color=colors, tick_label=x_tick)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=16)
        # plt.ylim(0, 2)
        plt.xlabel(col, color='r', fontsize=20)
        plt.ylabel("Values", fontsize=20)
        plt.title(name, color='indigo', fontsize=20)
        for i in range(len(x_tick)):
            val = "%.3f" % y[i]
            plt.text(i, y[i], val, fontsize=12, va="bottom", ha="center")
        pig_name = name+'_'+col
        des_path2 = despath + pig_name + '.pdf'
        plt.savefig(des_path2, bbox_inches='tight')
        plt.show()



if __name__ == '__main__':
    ori_path = r'E:\work\ai_model\wzm_prodict\result\plot_contrast_2\\'
    for file in os.listdir(ori_path):
        if file.endswith('.csv'):
            name1 = file.split('.')[0]
            name = name1.split("_")[-1]
            # name = "Project_" + name1.split("_")[-1]
            path = ori_path+file
            plot_fac(path, name)