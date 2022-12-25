# 画实验图
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import pandas as pd
ori_path = r'E:/work/ai_work/oss_health/result_plot_0607/'
# 均值化
def norm_data(arr):
    res=[]
    for x in arr:
        x = float(x - np.min(arr))/(np.max(arr)- np.min(arr))
        res.append(x)
    return res

# 归一化
def norm_data2(arr):
    res=[]
    for x in arr:
        x = float(x - np.mean(arr))/(np.max(arr)- np.min(arr))
        res.append(x)
    return res

def extract_data(ori_path):
    files = os.listdir(ori_path) # 随机挑选6个项目，画柱状图和折线图
    pro_scores = {}

    for file in files:
        file_name = ori_path + file
        data = pd.read_csv(file_name)[:320]
        date_time = data['date']
        # pro_name = data['project_id'][0]
        pro_name = data['project_id'][0]
        health_score = data['score'].tolist()
        pro_scores[pro_name] = health_score

    return date_time, pro_scores

# GitHub 中开源软件项目健康性分布直方图 图3.6
def health_pro(date_time, pro_score, save_path):
    fig = plt.figure(figsize=(20, 20))
    i = 1
    for k, v in pro_score.items():
        num = str(23) + str(i)
        ax = fig.add_subplot(int(num))
        plt.title("Project_" + str(k))
        plt.xlabel("Time")

        xticks = list(range(0, len(date_time), 40))
        date_time_res = date_time.tolist()
        xlabels = [date_time_res[x] for x in xticks]
        xticks.append(len(date_time_res))
        xlabels.append(date_time_res[-1])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=40)
        plt.ylabel("Health")
        color = ['c', 'm', 'y', 'k', 'b', 'g', 'r']
        #     c = random.sample(color,1)[0]
        c = color[i]
        ax.plot(date_time_res, v, '%s-' % c)
        ax.bar(date_time_res, v, color=c)
        i += 1
    path = save_path+"开源软件项目健康性分布直方图.pdf"
    plt.savefig(path, bbox_inches='tight')
    plt.show()

# 单独出图  直方图
def health_pro_one(date_time, pro_score, save_path):
    # fig = plt.figure(figsize=(20, 20))
    i = 1
    for k, v in pro_score.items():
        fig = plt.figure(figsize=(8, 8))
        num = 111
        ax = fig.add_subplot(int(num))
        plt.title("Project_" + str(k), fontsize=20)
        # plt.xlabel("Time", fontsize=20)

        xticks = list(range(0, len(date_time), 40))
        date_time_res = date_time.tolist()
        xlabels = [date_time_res[x] for x in xticks]
        xticks.append(len(date_time_res))
        xlabels.append(date_time_res[-1])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=40, fontsize=20)
        plt.ylabel("Health", fontsize=20)
        plt.yticks(fontsize=20)
        color = ['c', 'm', 'y', 'k', 'b', 'g', 'r']
        #     c = random.sample(color,1)[0]
        c = color[i]
        ax.plot(date_time_res, v, '%s-' % c)
        ax.bar(date_time_res, v, color=c)
        path = save_path + "开源软件项目健康性分布直方图_%s.pdf" % str(i)
        plt.savefig(path, bbox_inches='tight')
        i += 1
        plt.show()

# 折线图
def health_pro2(date_time, pro_score,save_path):
    # 健康性随时间的变化，
    fig = plt.figure(figsize=(20, 20))
    i = 1
    for k, v in pro_score.items():
        num = str(23) + str(i)
        ax = fig.add_subplot(int(num))
        #     plt.title(k)
        plt.title("Project_" + str(k))
        xticks = list(range(0, len(date_time), 40))
        date_time_res = date_time.tolist()
        xlabels = [date_time_res[x] for x in xticks]
        xticks.append(len(date_time_res))
        xlabels.append(date_time_res[-1])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=40)
        plt.ylabel("Health")
        color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        #     c = random.sample(color,1)[0]
        c = color[i]
        ax.plot(date_time_res, v, '%s-' % c)
        i += 1
    path = save_path + "开源软件项目健康性分布折线图.pdf"
    plt.savefig(path, bbox_inches='tight')
    plt.show()

# 折线图
def health_pro2_one(date_time, pro_score,save_path):
    # 健康性随时间的变化
    i = 1
    for k, v in pro_score.items():
        fig = plt.figure(figsize=(8, 8))
        num = 111
        ax = fig.add_subplot(int(num))
        #     plt.title(k)
        plt.title("Project_" + str(k), fontsize=20)
        xticks = list(range(0, len(date_time), 40))
        date_time_res = date_time.tolist()
        xlabels = [date_time_res[x] for x in xticks]
        xticks.append(len(date_time_res))
        xlabels.append(date_time_res[-1])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=40, fontsize=20)
        plt.ylabel("Health", fontsize=20)
        color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        #     c = random.sample(color,1)[0]
        c = color[i]
        ax.plot(date_time_res, v, '%s-' % c)

        path = save_path + "开源软件项目健康性分布折线图_%d.pdf"%i
        i += 1
        plt.savefig(path, bbox_inches='tight')
        plt.show()

# 针对一个项目分析
def one_file_plot_fa(one_file):
    data = pd.read_csv(one_file)[:200]
    press_dic = {}  # 压力层
    state_dic = {}  # 状态层
    response_dic = {}  # 响应层
    col = data.columns
    columns = col[col.str.contains('fac')][:-1]  # 获取fac因子个数（自动确定数量）
    # num_fac = len(columns)  # 3
    fac_num = data[columns]

    date_time = data['date'].tolist()

    forks = norm_data(data['forks'].tolist())  # state
    # watchers = norm_data(data['watchers'].tolist())  # state
    commits = norm_data(data['commits'].tolist())  # state
    commit_comment = norm_data(data['commit_count'].tolist())  # state
    state_dic['forks'] = forks
    # state_dic['watchers'] = watchers
    state_dic['commits'] = commits
    state_dic['commit_comment'] = commit_comment

    req_opened = norm_data(data['req_opened'].tolist())  # response
    req_merged = norm_data(data['req_merged'].tolist())  # response
    req_closed = norm_data(data['req_closed'].tolist())  # response
    other = norm_data(data['other'].tolist())  # response
    response_dic['req_opened'] = req_opened
    # response_dic['req_mreged'] = req_merged
    response_dic['req_closed'] = req_closed
    # response_dic['other'] = other

    issue = norm_data(data['issue'].tolist())  # press
    issue_comment = norm_data(data['issue_comment'].tolist())  # press
    press_dic['issue'] = issue
    press_dic['issue_comment'] = issue_comment

    project_id = data['project_id'][0]

    fac_score = data['factor_score'].tolist()
    return project_id, date_time, press_dic, state_dic, response_dic, fac_num, fac_score, col

#
def one_file_plot_ewm(ori_file, res_file):
    # data = pd.read_csv(ori_file)[-200:]   #原始数据
    # data_res = pd.read_csv(res_file)[-200:]  #  权重数据
    data = pd.read_csv(ori_file)  # 原始数据
    data_res = pd.read_csv(res_file)
    press_dic = {}  # 压力层
    state_dic = {}  # 状态层
    response_dic = {}  # 响应层

    press_weight_dic = {}
    state_weight_dic = {}
    response_weight_dic = {}

    col = data.columns
    # columns = col[col.str.contains('fac')][:-1]  # 获取fac因子个数（自动确定数量）
    # num_fac = len(columns)  # 3
    # fac_num = data[columns]

    date_time = data['date'].tolist()

    forks = norm_data(data['forks'].tolist())  # state
    # watchers = norm_data(data['watchers'].tolist())  # state
    commits = norm_data(data['commits'].tolist())  # state
    # commit_comment = norm_data(data['commit_comment'].tolist())  # state
    commit_comment = norm_data(data['commit_count'].tolist())  # state
    state_dic['forks'] = forks
    # state_dic['watchers'] = watchers
    state_dic['commits'] = commits
    state_dic['commit_comment'] = commit_comment
    # 权重
    forks_weight = data_res['forks'].tolist()  # state
    # watchers_weight = data_res['watchers'].tolist()  # state
    commits_weight = data_res['commits'].tolist()  # state
    commit_comment_weight = data_res['commit_count'].tolist()  # state

    state_weight_dic['forks'] = forks_weight
    # state_weight_dic['watchers'] = watchers_weight
    state_weight_dic['commits'] = commits_weight
    state_weight_dic['commit_comment'] = commit_comment_weight

    # 原始值
    req_opened = norm_data(data['req_opened'].tolist())  # response
    # req_merged = norm_data(data['req_merged'].tolist())  # response
    req_closed = norm_data(data['req_closed'].tolist())  # response
    # other = norm_data(data['other'].tolist())  # response

    response_dic['req_opened'] = req_opened
    # response_dic['req_mreged'] = req_merged
    response_dic['req_closed'] = req_closed
    # response_dic['other'] = other

    # 权重
    req_opened_weight = data_res['req_opened'].tolist() # response
    # req_merged_weight = data_res['req_merged'].tolist()  # response
    req_closed_weight = data_res['req_closed'].tolist()  # response
    # other_weight = data_res['other'].tolist()  # response

    response_weight_dic['req_opened'] = req_opened_weight
    # response_weight_dic['req_mreged'] = req_merged_weight
    response_weight_dic['req_closed'] = req_closed_weight
    # response_weight_dic['other'] = other_weight

    # 原始
    issue = norm_data(data['issue'].tolist())  # press
    issue_comment = norm_data(data['issue_comment'].tolist())  # press
    press_dic['issue'] = issue
    press_dic['issue_comment'] = issue_comment

    # weight
    issue_weight = data_res['issue'].tolist()  # press
    issue_comment_weight = data_res['issue_comment'].tolist()  # press
    press_weight_dic['issue'] = issue_weight
    press_weight_dic['issue_comment'] = issue_comment_weight

    # project_id = data['project_id'][0]
    project_id = 'redis'
    fac_score = data_res['score'].tolist()
    return project_id, date_time, press_dic, state_dic, response_dic, fac_score, col


# 对一个项目，获取压力层、状态层和响应层
def one_file_ewm(ori_file):
    data_res = pd.read_csv(ori_file)[-49:]   #原始数据

    press_weight_dic = {} # 压力层
    state_weight_dic = {} # 状态层
    response_weight_dic = {}  # 响应层

    date_time = np.arange(1, 50, 1)


    # 权重
    forks_weight = data_res['forks'].tolist()  # state
    # watchers_weight = data_res['watchers'].tolist()  # state
    commits_weight = data_res['commits'].tolist()  # state
    commit_comment_weight = data_res['commit_count'].tolist()  # state

    state_weight_dic['forks'] = forks_weight
    # state_weight_dic['watchers'] = watchers_weight
    state_weight_dic['commits'] = commits_weight
    state_weight_dic['commit_comment'] = commit_comment_weight

    # 权重
    req_opened_weight = data_res['req_opened'].tolist() # response
    # req_merged_weight = data_res['req_merged'].tolist()  # response
    req_closed_weight = data_res['req_closed'].tolist()  # response
    # other_weight = data_res['other'].tolist()  # response

    response_weight_dic['req_opened'] = req_opened_weight
    # response_weight_dic['req_mreged'] = req_merged_weight
    response_weight_dic['req_closed'] = req_closed_weight
    # response_weight_dic['other'] = other_weight

    # weight
    issue_weight = data_res['issue'].tolist()  # press
    issue_comment_weight = data_res['issue_comment'].tolist()  # press
    press_weight_dic['issue'] = issue_weight
    press_weight_dic['issue_comment'] = issue_comment_weight

    return press_weight_dic, state_weight_dic, response_weight_dic, date_time

# 压力层与健康性随时间的变化，
def press_health(press_dic,date_time,fac_score,save_path):
    fig = plt.figure(figsize=(8, 8))
    i = 1
    for k, v in press_dic.items():
        num = str(12)+str(i)
        ax = fig.add_subplot(int(num))
        plt.title(k)
        xticks = list(range(0, len(date_time),50))
        xlabels = [date_time[x] for x in xticks]
        xticks.append(len(date_time))
        xlabels.append(date_time[-1])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=40)

        plt.ylabel(k)
        color=['b','g','c','m','y','k']
        c = random.sample(color,1)[0]
    #     c_health = color[-1]
    #     c=color[i]
        ax.plot(date_time,v,'%s--'%c, label = k)
        ax.plot(date_time,fac_score,'r-', label="Health")
        plt.legend(fontsize=18)
        i += 1
    path = save_path + "压力层与健康性随时间的变化趋势图.pdf"
    plt.savefig(path, bbox_inches='tight')
    plt.show()

## 压力层与健康性随时间的变化，
def press_health_one(press_dic,date_time,fac_score,save_path):

    i = 1
    for k, v in press_dic.items():
        fig = plt.figure(figsize=(8, 8))
        num = 111
        ax = fig.add_subplot(int(num))
        plt.title(k, fontsize=20)
        xticks = list(range(0, len(date_time),50))
        xlabels = [date_time[x] for x in xticks]
        xticks.append(len(date_time))
        xlabels.append(date_time[-1])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=40, fontsize=20)

        plt.ylabel("value", fontsize=20)
        plt.yticks(fontsize=20)
        color=['b','g','c','m','y','k']
        c = random.sample(color,1)[0]
    #     c_health = color[-1]
    #     c=color[i]
        ax.plot(date_time,v,'%s--'%c, label= k)
        ax.plot(date_time,fac_score,'r-', label="Health")
        plt.legend(fontsize=18)

        path = save_path + "压力层与健康性随时间的变化趋势图_%d.pdf"%i
        i += 1
        plt.savefig(path, bbox_inches='tight')
        plt.show()

def state_health(state_dic,date_time,fac_score,save_path):
    # 状态层与健康性随时间的变化，
    fig = plt.figure(figsize=(20, 20))
    i = 1
    for k, v in state_dic.items():
        num = str(22) + str(i)
        ax = fig.add_subplot(int(num))
        plt.title(k)
        xticks = list(range(0, len(date_time), 40))
        xlabels = [date_time[x] for x in xticks]
        xticks.append(len(date_time))
        xlabels.append(date_time[-1])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=40)

        plt.ylabel(k)
        color = ['b', 'g', 'c', 'm', 'y', 'k']
        c = random.sample(color, 1)[0]
        #     c_health = color[-1]
        #     c=color[i]
        ax.plot(date_time, v, '%s--' % c, label=k)
        ax.plot(date_time, fac_score, 'r-', label="Health")
        plt.legend(fontsize=18)
        i += 1
    path = save_path + "状态层与健康性随时间的变化趋势图.pdf"
    plt.savefig(path, bbox_inches='tight')
    plt.show()

def state_health_one(state_dic,date_time,fac_score,save_path):
    # 状态层与健康性随时间的变化，
    i = 1
    for k, v in state_dic.items():
        fig = plt.figure(figsize=(8, 8))
        num = 111
        ax = fig.add_subplot(int(num))
        plt.title(k, fontsize=20)
        xticks = list(range(0, len(date_time), 40))
        xlabels = [date_time[x] for x in xticks]
        xticks.append(len(date_time))
        xlabels.append(date_time[-1])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=40, fontsize=20)

        plt.ylabel("value", fontsize=20)
        plt.yticks(fontsize=20)
        color = ['b', 'g', 'c', 'm', 'y', 'k']
        c = random.sample(color, 1)[0]
        #     c_health = color[-1]
        #     c=color[i]
        ax.plot(date_time, v, '%s--' % c, label=k)
        ax.plot(date_time, fac_score, 'r-', label="Health")
        plt.legend(fontsize=18)

        path = save_path + "状态层与健康性随时间的变化趋势图_%d.pdf"%i
        i += 1
        plt.savefig(path, bbox_inches='tight')
        plt.show()

# 响应层与健康性随时间的变化，
def response_health(response_dic, date_time, fac_score,save_path):
    # 响应层与健康性随时间的变化，
    fig = plt.figure(figsize=(20, 20))
    i = 1
    for k, v in response_dic.items():
        num = str(22) + str(i)
        ax = fig.add_subplot(int(num))
        plt.title(k, fontsize=20)
        xticks = list(range(0, len(date_time), 40))
        xlabels = [date_time[x] for x in xticks]
        xticks.append(len(date_time))
        xlabels.append(date_time[-1])
        ax.set_xticks(xticks, fontsize=20)
        ax.set_xticklabels(xlabels, rotation=40, fontsize=20)

        plt.ylabel(k)
        color = ['b', 'g', 'c', 'm', 'y', 'k']
        c = random.sample(color, 1)[0]
        #     c_health = color[-1]
        #     c=color[i]
        ax.plot(date_time, v, '%s--' % c, label=k, fontsize=20)
        ax.plot(date_time, fac_score, 'r-', label="Value", fontsize=20)
        plt.legend(fontsize=18)
        i += 1
    path = save_path+"响应层与健康性随时间的变化趋势图.pdf"
    plt.savefig(path, bbox_inches='tight')
    plt.show()


# 响应层与健康性随时间的变化，
def response_health_one(response_dic, date_time, fac_score,save_path):
    # 响应层与健康性随时间的变化，

    i = 1
    for k, v in response_dic.items():
        fig = plt.figure(figsize=(8, 8))
        num = 111
        ax = fig.add_subplot(int(num))
        plt.title(k, fontsize=20)
        xticks = list(range(0, len(date_time), 40))
        xlabels = [date_time[x] for x in xticks]
        xticks.append(len(date_time))
        xlabels.append(date_time[-1])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=40, fontsize=20)

        plt.ylabel("value", fontsize=20)
        plt.yticks(fontsize=20)
        color = ['b', 'g', 'c', 'm', 'y', 'k']
        c = random.sample(color, 1)[0]
        #     c_health = color[-1]
        #     c=color[i]
        ax.plot(date_time, v, '%s--' % c, label=k)
        ax.plot(date_time, fac_score, 'r-', label="Health")
        plt.legend(fontsize=18)
        path = save_path+"响应层与健康性随时间的变化趋势图_%d.pdf"%i
        i += 1
        plt.savefig(path, bbox_inches='tight')
        plt.show()

# 因子分析，不同因子值与各个指标之间的关系图
# 压力层与因子随时间的变化，
def press_factor(fac_num,press_dic,columns,date_time,save_path):
    # 压力层与因子随时间的变化，
    fig = plt.figure(figsize=(20, 20))
    i = 1
    columns = columns[columns.str.contains('fac')][:-1]
    for k, v in press_dic.items():

        num = str(21) + str(i)
        ax = fig.add_subplot(int(num))
        plt.title(k + "_Factors")
        xticks = list(range(0, len(date_time), 40))
        xlabels = [date_time[x] for x in xticks]
        xticks.append(len(date_time))
        xlabels.append(date_time[-1])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=40)

        plt.ylabel("Press")
        ax.plot(date_time, v, 'r--', label=k)
        i += 1
        j = 0
        for col in columns:
            j += 1
            ax.plot(date_time, norm_data(fac_num[col].tolist()), '%s-' % j, label=col)
        plt.legend(fontsize=18)
    path = save_path+"压力层与因子随时间的变化趋势图_%d.pdf"%i
    plt.savefig(path, bbox_inches='tight')
    plt.show()

# 状态层与因子随时间的变化，
def state_factor(fac_num,state_dic,columns,date_time,save_path):
    # 状态层
    fig = plt.figure(figsize=(20, 20))
    i = 1
    columns = columns[columns.str.contains('fac')][:-1]
    for k, v in state_dic.items():
        num = str(41) + str(i)
        ax = fig.add_subplot(int(num))
        plt.title(k + "_Factors")
        xticks = list(range(0, len(date_time), 40))
        xlabels = [date_time[x] for x in xticks]
        xticks.append(len(date_time))
        xlabels.append(date_time[-1])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=40)

        plt.ylabel("State")
        ax.plot(date_time, v, 'r--', label=k)
        i += 1
        j = 0
        for col in columns:
            j += 1
            ax.plot(date_time, norm_data(fac_num[col].tolist()), '%s-' % j, label=col)

        plt.legend(fontsize=18)
    path = save_path + "状态层与因子随时间的变化趋势图.pdf"
    plt.savefig(path, bbox_inches='tight')
    plt.show()
    # 压力层和fac2走势吻合

# 响应层与因子随时间的变化，
def response_factor(fac_num,response_dic,columns,date_time,save_path):
    # 响应层与因子随时间的变化，
    fig = plt.figure(figsize=(20, 20))
    i = 1
    columns = columns[columns.str.contains('fac')][:-1]
    for k, v in response_dic.items():
        num = str(41) + str(i)
        ax = fig.add_subplot(int(num))
        plt.title(k + "_Factors")
        xticks = list(range(0, len(date_time), 40))
        xlabels = [date_time[x] for x in xticks]
        xticks.append(len(date_time))
        xlabels.append(date_time[-1])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, rotation=40)

        plt.ylabel("Response")
        ax.plot(date_time, v, 'r--', label=k)
        i += 1
        j = 0
        for col in columns:
            j += 1
            ax.plot(date_time, norm_data(fac_num[col].tolist()), '%s-' % j, label=col)

        plt.legend(fontsize=18)
    path = save_path + "响应层与因子随时间的变化趋势图.pdf"
    plt.savefig(path, bbox_inches='tight')
    plt.show()
    # 压力层和fac2走势吻合


# 压力层权重随时间的变化，weight
def press_weight(press_weight_dic, date_time, save_path):
    # 压力层与因子随时间的变化，
    i = 1
    for k, v in press_weight_dic.items():
        fig = plt.figure(figsize=(8, 8))
        num = 111
        ax = fig.add_subplot(int(num))
        plt.title(k + "_Weights", fontsize=20)
        xticks = [0,9,19,29,39]
        xlabels = [date_time[x] for x in xticks]
        xticks.append(len(date_time))
        xlabels.append(date_time[-1])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=20)
        plt.xlabel("month", fontsize=20)
        plt.ylabel("Weights")
        plt.yticks(fontsize=20)
        ax.plot(date_time, v, 'b--', label=k)
        plt.legend(fontsize=18)
        path = save_path+"压力层权重随时间的变化趋势图_%d.pdf" % i
        i += 1
        plt.savefig(path, bbox_inches='tight')
        plt.show()

# 状态层权重随时间的变化，
def state_weight(state_weight_dic,date_time,save_path):
    # 状态层
    i = 1
    for k, v in state_weight_dic.items():
        fig = plt.figure(figsize=(8, 8))
        num = 111
        ax = fig.add_subplot(int(num))
        plt.title(k + "_Weights")
        xticks =  [0,9,19,29,39]
        xlabels = [date_time[x] for x in xticks]
        xticks.append(len(date_time))
        xlabels.append(date_time[-1])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        plt.xlabel("month", fontsize=20)
        plt.ylabel("Weights")
        plt.yticks(fontsize=20)
        ax.plot(date_time, v, 'r--', label=k)

        plt.legend(fontsize=18)
        path = save_path + "状态层全权重随时间的变化趋势图_%d.pdf"%i
        i += 1
        plt.savefig(path, bbox_inches='tight')
        plt.show()

# 响应层权重随时间的变化，
def response_weight(response_weight_dic,date_time,save_path):
    i = 1
    for k, v in response_weight_dic.items():
        fig = plt.figure(figsize=(8, 8))
        num = 111
        ax = fig.add_subplot(int(num))
        plt.title(k + "_Weights", fontsize=20)
        xticks =  [0,9,19,29,39]
        xlabels = [date_time[x] for x in xticks]
        xticks.append(len(date_time))
        xlabels.append(date_time[-1])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels)
        plt.xlabel("month", fontsize=20)
        plt.ylabel("Weights")
        plt.yticks(fontsize=20)
        ax.plot(date_time, v, 'g--', label=k)

        path = save_path + "响应层权重随时间的变化趋势图_%d.pdf"%i
        i += 1
        plt.savefig(path, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':

    # 取一个项目分析
    one_file = r'E:/work/ai_work/oss_health/result_plot_0607/project_43097.csv'
    project_id, date_time, press_dic, state_dic, response_dic, fac_num, fac_score, col = one_file_plot(one_file)

