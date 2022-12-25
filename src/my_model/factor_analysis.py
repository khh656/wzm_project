# 数据处理
import pandas as pd
import numpy as np

# 绘图
import seaborn as sns
import matplotlib.pyplot as plt
# 因子分析
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import matplotlib as mpl


mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
'''
因子分析是指研究从变量群中提取共性因子的统计技术。他发现学生的各科成绩之间存在着一定的相关性,一科成绩好的学生,
往往其他各科成绩也比较好,从而推想是否存在某些潜在的共性因子,或称某些一般智力条件影响着学生的学习成绩。
因子分析可在许多变量中找出隐藏的具有代表性的因子。将相同本质的变量归入一个因子,可减少变量的数目,
还可检验变量间关系的假设。
fac_num: 变量数
'''
def f_analysis(file_path, fac_num):
    df_ori = pd.read_csv(file_path)
    data = df_ori.drop(columns=['project_id', 'date'])
    beta = 1e-5
    df = data.apply(lambda x:(x-x.mean())/(x.std()+beta))
    # 去掉空值
    df.dropna(inplace=True)
    # 充分性检测
    check(df)
    # 确定因子个数 自动确定
    n_factors = get_factors(df, fac_num)
    # 根据确定的因子个数进行因子分析，获取方差贡献率和因子得分
    var, fa2_score = get_variance(df_ori, df, n_factors)
    df_ori['factor_score'] = ((var[1] / var[1].sum()+beta) * fa2_score).sum(axis=1)
    # 排序
    data = df_ori.sort_values(by='factor_score', ascending=False).reset_index(drop=True)
    data['rank'] = data.index + 1
    # index = data['index']
    res = data.iloc[:, -(n_factors+3):]
    print(res)
    return res  #  project_id  fac1   fac2  fac3  factor_score  rank
    # return df_ori

# df_ori 原始数据
def get_variance(df_ori,df,n_factors):
    # 取旋转后的结果
    fa2 = FactorAnalyzer(n_factors, rotation='varimax', method='principal')
    fa2.fit(df)
    # 给出贡献率
    var = fa2.get_factor_variance()

    # 计算因子得分
    fa2_score = fa2.transform(df)

    # 得分表
    column_list = ['fac' + str(i) for i in np.arange(n_factors) + 1]
    fa_score = pd.DataFrame(fa2_score, columns=column_list)
    for col in fa_score.columns:
        df_ori[col] = fa_score[col]
    print("\n各因子得分:\n", fa_score)

    # 方差贡献表
    df_fv = pd.DataFrame()
    df_fv['因子'] = column_list
    df_fv['方差贡献'] = var[1]
    df_fv['累计方差贡献'] = var[2]
    df_fv['累计方差贡献占比'] = var[1] / var[1].sum()
    print("\n方差贡献表:\n", df_fv)
    return var, fa2_score

def check(df):
    # 充分性检测
    print('巴特利球形度检验')
    chi_square_value, p_value = calculate_bartlett_sphericity(df)
    print('卡方值：', chi_square_value, 'P值', p_value)
    # KMO检验 相关性检验kmo要大于0.6，也说明变量之间存在相关性，可以进行分析。
    kmo_all, kmo_model = calculate_kmo(df)
    print('KMO检验：', kmo_model)

def get_factors(df, fac_num):
    # 查看相关矩阵特征值 10 个变量
    fa = FactorAnalyzer(fac_num, rotation='varimax', method='principal', impute='mean')
    fa.fit(df)
    # 得到特征值ev、特征向量v
    ev, v = fa.get_eigenvalues()
    print('相关矩阵特征值：', ev)
    # Create scree plot using matplotlib # 同样的数据绘制散点图和折线图
    plt.figure(figsize=(8, 6.5))
    plt.scatter(range(1, df.shape[1] + 1), ev)
    plt.plot(range(1, df.shape[1] + 1), ev)
    plt.title('碎石图', fontdict={'weight': 'normal', 'size': 25})
    plt.xlabel('因子', fontdict={'weight': 'normal', 'size': 15})
    plt.ylabel('特征值', fontdict={'weight': 'normal', 'size': 15})
    plt.grid()
    # plt.savefig('E:/suishitu.jpg')
    plt.show()

    # 确定因子个数 自动确定
    n_factors = sum(ev > 1)
    return n_factors
if __name__ == '__main__':
    # import os
    # ori_path = r'E:/work/ai_work/oss_health/des_data/'
    # i=1
    # data_res = []
    # res = []
    # des_path = r'E:\work\ai_work\oss_health\result\check.csv'
    # res_path = r'E:\work\ai_work\oss_health\result\mean_result.csv'
    # n_factors = 10  # 变量数
    # for file in os.listdir(ori_path):
    #     if i % 11 != 0 and i <= 110:
    #         try:
    #             file = ori_path+file
    #             data_ori = pd.read_csv(file)
    #             data = data_ori.drop(columns=['project_id', 'date', 'committer_id'])[1:1500]
    #             data1 = data.mean()
    #             data_res.append(data1.to_list())
    #             i += 1
    #         except:
    #             continue
    #     elif i <= 110:
    #         columns = ['forks', 'commits', 'commit_comment', 'req_opened', 'req_closed',
    #                    'req_merged', 'other', 'issue', 'issue_comment', 'watchers']
    #         file2 = list_to_csv(columns, data_res, des_path)
    #         result = f_analysis(des_path,n_factors)
    #         res.append(result['factor_score'].to_list())
    #         data_res=[]
    #         i += 1
    #     else:
    #         break
    # column = ['p1','p2','p3','p4','p5','p6','p7','p8','p9','p10',]
    # res_csv = list_to_csv(column, res, res_path)
    # res_data = pd.read_csv(res_path).mean()
    # print(res_data)  # 项目平均得分
    des_path = r'E:/work/ai_work/oss_health/assess_data_90/ten_0.csv'
    n_factors = 10
    fa = f_analysis(des_path, n_factors)

