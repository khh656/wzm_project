from src.model_plot.EWM_Health import EWM_scores_weight
from src.model_plot.plot_pic_utils import extract_data, \
    health_pro_one, health_pro2_one, one_file_plot_ewm, \
    response_health_one, press_health_one, state_health_one, press_weight, state_weight, response_weight, one_file_ewm


# 此文件针对论文评估部分
def plot_result(ori_path, ori_file, res_file, result_path):
    date_time0, pro_score = extract_data(ori_path)  # 随机提取6个项目
    # health_pro_one(date_time=date_time0, pro_score=pro_score, save_path=result_path)
    # health_pro2_one(date_time=date_time0, pro_score=pro_score, save_path=result_path)

    # 针对一个项目分析-压力层/状态层/响应层与健康性随时间的变化
    project_id, date_time, press_dic, state_dic, response_dic, fac_score, col = one_file_plot_ewm(ori_file, res_file)
    press_health_one(press_dic, date_time, fac_score, save_path=result_path)
    state_health_one(state_dic, date_time, fac_score, save_path=result_path)
    response_health_one(response_dic, date_time, fac_score, save_path=result_path)

    # 不同阶段的weight关联性说明
    # 首先选取一个项目，每30天计算一次权重值
    one_path_file = r'data_0712\redis.csv'
    des_path_file = r'data_0712\result_weight_score_0712\one_file_weight2.csv'
    EWM_scores_weight(one_path_file, des_path_file)
    # 其次，根据结果，画图
    ori_file = r'data_0712\result_weight_score_0712\one_file_weight2.csv'
    press_weight_dic, state_weight_dic, response_weight_dic, date_time = one_file_ewm(ori_file)
    press_weight(press_weight_dic, date_time, save_path=result_path)
    state_weight(state_weight_dic, date_time, save_path=result_path)
    response_weight(response_weight_dic, date_time, save_path=result_path)

if __name__ == '__main__':
    # FA
    # ori_path = r'E:/work/ai_work/oss_health/result_plot_0607/'
    # one_file = r'E:/work/ai_work/oss_health/result_plot_0607/project_25822.csv'
    # result_path = r'E:/work/ai_work/oss_health/result_picture/'
    # plot_result(ori_path, one_file, result_path)
    #
    # EWM
    ori_path = r'E:/work/ai_work/oss_health/result_weight_score_0610/'
    res_file = r'E:/work/ai_work/oss_health/result_weight_score_0610/project_1142.csv' # 结果数据
    ori_file = r'E:\work\ai_work\oss_health\data_30_10_EWM\project_1142.csv'  # 原始数据
    result_path = r'E:/work/ai_work/oss_health/result_picture_0607/'
    plot_result(ori_path, ori_file, res_file, result_path)