'''
import pandas as pd

csv_file = "shang_data.csv"

csv_data = pd.read_csv(csv_file, low_memory=False) # 防止弹出警告

x= pd.DataFrame(csv_data)

# x=x.iloc[:,1:].T

# 1、数据均值化处理

x_mean=x.mean(axis=1)

for i in range(x.index.size):

    x.iloc[i,:] = x.iloc[i,:]/x_mean[i]

    # x.iloc[i,:] = x.iloc[i,:]/x_mean[i]  #i或者i+1不太明晰

# 2、提取参考队列和比较队列

ck=x.iloc[0,:]

cp=x.iloc[1:,:]

# 比较队列与参考队列相减

t=pd.DataFrame()

for j in range(cp.index.size):

    temp=pd.Series(cp.iloc[j,:]-ck)

    t=t.append(temp,ignore_index=True)

    #求最大差和最小差

    mmax=t.abs().max().max()

    mmin=t.abs().min().min()

rho=0.5

#3、求关联系数

ksi=((mmin+rho*mmax)/(abs(t)+rho*mmax))

#4、求关联度

r=ksi.sum(axis=1)/ksi.columns.size

#5、关联度排序，得到结果r3>r2>r1

result=r.sort_values(ascending=False)

print(result)

'''

# 灰色关联法 是根据因素之间发展趋势的相似或相异程度，亦即“灰色关联度”，作为衡量因素间关联程度的一种方法。
# 在系统发展过程中，若两个因素变化的趋势具有一致性，即同步变化程度较高，即可谓二者关联程度较高；
# 反之，则较低。因此，灰色关联法，是根据因素之间发展趋势的相似或相异程度，亦即“灰色关联度”，
# 作为衡量因素间关联程度的一种方法。
#选择一个母序列(即评价标准)能反映系统行为特征的数据序列，类似于因变量Y，此处记为x0 req_merged，用每列的最优秀值
#导入相关库
import pandas as pd
import numpy as np
# project_id,date,forks,committer_id,commits,commit_comment,req_opened,req_closed,req_merged,other,issue,issue_comment,watchers
#导入数据
data_ori = pd.read_csv(r'E:\work\ai_work\oss_health\data\project_1486.csv')
data = data_ori.drop(columns=['project_id', 'date', 'committer_id'])
#提取变量名 x1 -- x7
label_need = data.keys()
#提取上面变量名下的数据
data1 = data[label_need].values
# 0.002~1区间归一化
[m,n]= data1.shape  # 得到行数和列数
data2=data1.astype('float')
data3=data2
ymin=0.002
ymax=1
for j in range(0,n):
    d_max = max(data2[:, j])
    d_min = min(data2[:, j])
    data3[:, j] = (ymax-ymin)*(data2[:, j]-d_min)/(d_max-d_min)+ymin

# 得到其他列和参考列相等的绝对值
for i in range(0, len(label_need)-1):
    data3[:, i] = np.abs(data3[:, i]-data3[:, 0])
'''
t=range(2007,2014)
plt.plot(t,data3[:,0],'*-',c='red')
for i in range(4):
    plt.plot(t,data3[:,2+i],'.-')
plt.xlabel('year')
plt.legend(['x1','x4','x5','x6','x7'])
plt.title('灰色关联分析')
'''
# 得到绝对值矩阵的全局最大值和最小值
# data4 = data3[:, 1:]
data4 = data_ori.drop(columns=['project_id', 'date', 'committer_id', 'req_merged'])
d_max = np.max(data2)
d_min = np.min(data2)
a = 0.5  # 定义分辨系数

# 计算灰色关联矩阵
data4 = (d_min + a*d_max)/(data4+a*d_max)
scores = np.mean(data4, axis=0)  # 可以当作权重吗
print('forks,commits,commit_comment,req_opened,req_closed,other,issue,issue_comment,watchers'
      '与 req_merged之间的灰色关联度分别为：')
print(scores)