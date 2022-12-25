import os
import random
from tqdm import tqdm
import pandas as pd


from src.model_plot.plot_pic import plot_result
from src.model_plot.process_plzft import EWM_scores_pros, plot_plzft
from src.my_model.EWM_analysis import EWM_scores
from src.my_model.GRA import gra_score
from src.my_model.PCA_analysis import pca_scores

from src.my_model.Topsis_analysis import Topsis_scores
from src.my_model.factor_analysis import f_analysis
from src.utils.util import list_to_csv, dic_csv_score


# 评价得分存入csv 第二步骤
def dic_csv(ori_path,n_factors,des_path):
    fa_file = des_path + 'FactorAnalysis.csv'
    gra_file = des_path + 'GRA.csv'
    pca_file = des_path + 'PCA.csv'
    ewm_file = des_path + 'EWM.csv'
    topsis_file = des_path + 'Topsis.csv'
    ewm_top_file = des_path + 'EWM_Top.csv'
    fa_dic, gra_dic, pca_dic, ewm_dic, topsis_dic, ewm_topsis_dic = effec_comp(ori_path, n_factors)
    dic_csv_score(fa_dic, fa_file)
    dic_csv_score(gra_dic, gra_file)
    dic_csv_score(pca_dic, pca_file)
    dic_csv_score(ewm_dic, ewm_file)
    dic_csv_score(topsis_dic, topsis_file)
    dic_csv_score(ewm_topsis_dic, ewm_top_file)

# 不同分析方法对20个项目评分
def effec_comp(end_path, n_factors):
    fa_dic = {}
    gra_dic = {}
    pca_dic = {}
    ewm_dic = {}
    topsis_dic ={}
    ewm_topsis_dic = {}
    count_fa = 0
    count_gra = 0
    count_pca = 0
    count_ewm = 0
    count_topsis = 0
    count_ewm_topsis = 0
    for file in os.listdir(end_path):
        file_path = end_path + file
        # 因子分析法
        try:
           fa = f_analysis(file_path, n_factors)
           fa_id_score = dict(zip(fa['project_id'],fa['factor_score']))
           for key in fa_id_score.keys():
               if key not in fa_dic.keys():
                   fa_dic[key] = fa_id_score.get(key)
               else:
                   fa_dic[key] = float(fa_dic.get(key))+ float(fa_id_score.get(key))
           count_fa += 1  # 可以计算fa的有效文件个数
        except:
            pass

        # 灰色关联分析
        try:
           gra_score_dic = gra_score(file_path)
           for key in gra_score_dic.keys():
               if key not in gra_dic.keys():
                   gra_dic[key] = gra_score_dic.get(key)
               else:
                   gra_dic[key] = float(gra_dic.get(key)) + float(gra_score_dic.get(key))
           count_gra += 1
        except:
            pass

        # PCA 分析
        try:
           pca_score_dic = pca_scores(file_path)
           for key in pca_score_dic.keys():
               if key not in pca_dic.keys():
                   pca_dic[key] = pca_score_dic.get(key)
               else:
                   pca_dic[key] = float(pca_dic.get(key)) + float(pca_score_dic.get(key))
           count_pca += 1
        except:
            pass

        # 熵权法分析
        try:
           ewm_score_dic, topsis_weight = EWM_scores(file_path)
           for key in ewm_score_dic.keys():
               if key not in ewm_dic.keys():
                   ewm_dic[key] = ewm_score_dic.get(key)
               else:
                   ewm_dic[key] = float(ewm_dic.get(key)) + float(ewm_score_dic.get(key))
           count_ewm += 1
        except:
            pass

        # Topsis
        try:
           topsis_scores_dic = Topsis_scores(file_path,
                                              weight=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                                              postive=[1])
           for key in topsis_scores_dic.keys():
               if key not in topsis_dic.keys():
                   topsis_dic[key] = topsis_scores_dic.get(key)
               else:
                   topsis_dic[key] = float(topsis_dic.get(key)) + float(topsis_scores_dic.get(key))
           count_topsis += 1
        except:
            pass

        # ewm+Topsis
        try:
            ewm_topsis_scores_dic = Topsis_scores(file_path,
                                              weight= list(topsis_weight),
                                              postive=[1])
            for key in ewm_topsis_scores_dic.keys():
                if key not in ewm_topsis_dic.keys():
                    ewm_topsis_dic[key] = ewm_topsis_scores_dic.get(key)
                else:
                    ewm_topsis_dic[key] = float(ewm_topsis_dic.get(key)) + float(ewm_topsis_scores_dic.get(key))
            count_ewm_topsis += 1
        except:
            pass
    fa_dic = dic_div(fa_dic,count_fa)
    gra_dic = dic_div(gra_dic, count_gra)
    pca_dic = dic_div(pca_dic, count_pca)
    ewm_dic = dic_div(ewm_dic, count_ewm)
    topsis_dic = dic_div(topsis_dic, count_topsis)
    ewm_topsis_dic = dic_div(ewm_topsis_dic, count_ewm_topsis)
    return fa_dic, gra_dic, pca_dic, ewm_dic, topsis_dic, ewm_topsis_dic


# 将字典中的值平均化
def dic_div(dic, count):
    for k, v in dic.items():
     dic[k] = v/count
    return dic

def factors_month(ori_path, des_path):
    # data_res = []
    # 随机选取20个项目
    files = random.sample(os.listdir(ori_path), 18)
    for i in tqdm(range(30)):  # 1-17*3 个月
        data_res = []
        for file in files:
            file_path = ori_path + file
            data_ori = pd.read_csv(file_path)
            data = data_ori.iloc[i]
            data_res.append(data.to_list())
        # columns = ['date', 'forks', 'commits', 'commit_comment', 'req_opened', 'req_closed',
        #            'req_merged', 'other', 'issue', 'issue_comment', 'watchers','project_id']
        columns = ['date', 'commits', 'commit_count', 'forks', 'issue', 'issue_comment', 'req_closed',
                   'req_merged', 'req_opened', 'other','project_id']
        d_path = des_path + "ten_" + str(i)+'.csv'
        list_to_csv(columns, data_res, d_path)


# 画频率直方图,所有项目每个月下的数据放到一起
def factors_month_alldata(ori_path, des_path):
        for i in tqdm(range(17)):
            data_res = []  # 1-17*3 个月，即1-51个月的数据
            for file in os.listdir(ori_path):
                file_path = ori_path + file
                data_ori = pd.read_csv(file_path)
                try:
                    data = data_ori.iloc[i]
                    data_res.append(data.to_list())
                    print(file)
                except:
                    pass

            # columns = ['date', 'forks', 'commits', 'commit_comment', 'req_opened', 'req_closed',
            #                'req_merged', 'other', 'issue', 'issue_comment', 'watchers','project_id']
            columns = ['date','commits','commit_count','forks','issue','issue_comment','req_closed',
                       'req_merged','req_opened','other','project_id']
            d_path = des_path + str(abs(i))+'.csv'
            list_to_csv(columns, data_res, d_path)
if __name__ == '__main__':

    path_file = r'data_0712/result_weight_score_0712/all_file_scores.csv'
    dest_path = r'result\result_1\\'
    plot_plzft(path_file, dest_path)
    # 指标权重随时间变化情况
    ori_path2 = r'data_0712\one_file_resulrt\\'
    res_file2 = r'data_0712\one_file_resulrt\ace.csv'
    ori_file2 = r'data_0712\one_file\ace.csv'
    result_path2 = r'result\result_1\\'
    plot_result(ori_path2, ori_file2, res_file2, result_path2)
    print("Success!!!")
