import os
import pandas as pd
import matplotlib.pyplot as plt
#用于显示中文
import matplotlib
matplotlib.rcParams['font.family'] = 'SimHei'
path = r'result_plot_6456.csv'

def process_data(x):
    return float('%.2f' % x)
def plot_remove(file,name,id, dir_path):
    data = pd.read_csv(file, index_col="method").drop(columns=['project_id', 'r2'])
    data_tf = data.T
    EFormer = data_tf['EWM_TFormer'].tolist()
    Transformer = data_tf['Transformer'].tolist()
    EFormer_list = list(map(process_data, EFormer))
    TF_list = list(map(process_data, Transformer))
    x = ['MAE', 'MSE', 'RMSE', 'MAPE']  # mae, mape, mse, rmse
    y = EFormer_list  # EFormer
    z = TF_list  # Transformer

    # 增加一个固定维度，长度与上述数据一样
    fix_value = []
    # 求出数据y和z的最大值，取其1/4的值作为固定长度
    value_max = max(max(y), max(z))
    fix_temp = value_max / 4
    for i in range(len(x)):
        fix_value.append(fix_temp)
    print(fix_value)
    # 将y，z其中一个维度变为负值，我们选择z
    z_ne = [-i for i in z]
    # print(z_ne)

    # 设置中文显示为微软雅黑
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    # 设置图表默认字体为12
    plt.rcParams['font.size'] = 20
    # 设置画布大小
    plt.figure(figsize=(15, 8))

    # 画条形图,设置颜色和柱宽，将fix_value,y,z_ne依次画进图中
    plt.barh(x, fix_value, color='w', height=0.5)
    plt.barh(x, y, left=fix_value, color='#037171', label='EFormer', height=0.5)
    plt.barh(x, z_ne, color='#FF474A', height=0.5, label='EFormer-EWM')

    # 添加数据标签，将fix_value的x标签显示出来，y和z_ne的数据标签显示出来
    for a, b in zip(x, fix_value):
        plt.text(b / 2.0, a, '%s' % str(a), ha='center', va='center', fontsize=20)
    for a, b in zip(x, y):
        plt.text(b + fix_temp + value_max / 20.0 + 0.01, a, '%s' % float(b), ha='center', va='center')
    for a, b in zip(x, z):
        plt.text(-b - value_max / 20.0 - 0.01, a, '%s' % float(b), ha='center', va='center')
    # 坐标轴刻度不显示
    plt.xticks([])
    plt.yticks([])
    # 添加图例，自定义位置
    plt.legend(bbox_to_anchor=(-0.02, 0.5), frameon=False)
    # 添加标题，并设置字体
    title_name = "消融实验--Project_"+id
    plt.title(label=title_name, fontsize=20, fontweight='bold')
    ax = plt.gca()
    ax.set_axisbelow(True)
    # 设置绘图区域边框不可见
    [ax.spines[loc_axis].set_visible(False) for loc_axis in ['bottom', 'top', 'right', 'left']]
    # 使布局更合理
    plt.tight_layout()

    des_path = dir_path + name + '_remove.pdf'
    plt.savefig(des_path, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    ori_path = r'E:\work\ai_model\wzm_prodict\result\plot_contrast\\'
    for file in os.listdir(ori_path):
        if file.endswith('.csv'):
            name = file.split('.')[0]
            id = name.split("_")[-1]
            path = ori_path+file
            plot_remove(path, name,id)