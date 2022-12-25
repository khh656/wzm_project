# 画图
import os
# from result.plot_fig.plot_contrast_split import plot_fac
# 对比指标图
# from result.plot_fig.plot_pre_true import data_pro
# from result.plot_fig.plot_pre_true_2 import plot_fig
# from result.plot_fig.plot_remove import plot_remove
from src.my_model.plot_contrast_split import plot_fac
from src.my_model.plot_remove import plot_remove


def plot_fac_contrast(ori_path, des_dir):
    for file in os.listdir(ori_path):
        if file.endswith('.csv'):
            name1 = file.split('.')[0]
            name = "Project_" + name1.split("_")[-1]
            path = ori_path + file
            plot_fac(path, name, des_dir)

# 消融实验

def plot_rem(ori_path, dir_path):
    for file in os.listdir(ori_path):
        if file.endswith('.csv'):
            name = file.split('.')[0]
            id = name.split("_")[-1]
            path = ori_path+file
            plot_remove(path, name,id, dir_path)

if __name__ == '__main__':
    # 更改路径
    ori_path = r'data_0712\plot_contrast_2\\'
    des_dir = r'result\result_2\\'
    # 指标对比图
    for file in os.listdir(ori_path):
        if file.endswith('.csv'):
            name1 = file.split('.')[0]
            name = name1.split("_")[-1]
            path = ori_path + file
            plot_fac(path, name, despath=des_dir)
    # 消融实验图
    plot_rem(ori_path, des_dir)
