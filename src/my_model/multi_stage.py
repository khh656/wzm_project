import pandas as pd

path = r'E:\work\ai_work\oss_health\data_0712\result_weight_score_0712\one_file_weight2.csv'
data = pd.read_csv(path)
window = 12

def get_fac(data):
    weight_list = []
    index_list = []
    i = -85
    while i <= -window:
        j = 0
        data_res = data.iloc[i:i+window]
        i = i+window
        data_mean = data_res.mean(axis=0)
        data_sort = data_mean.sort_values(ascending=False,inplace=False)
        while True:
            if data_sort.iloc[:j].sum()/data_sort.sum() > 0.92:
                break
            j += 1
        weight_list.append(data_sort.values[:j].tolist())
        index_list.append(data_sort.index[:j].tolist())
    return weight_list,index_list

if __name__ == '__main__':
    val, index = get_fac(data)
    for i in val:
        print(i)
    print("***"*50)
    for i in index:
        print(i)
    # print("权重值：",val)
    # print("特征",index)