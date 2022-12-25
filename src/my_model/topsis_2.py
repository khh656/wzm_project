# import pandas as pd
# import numpy as np
#
#
# def topsis(data, weight=None):
# 	# 归一化
# 	data = data / np.sqrt((data ** 2).sum())
#
# 	# 最优最劣方案
# 	Z = pd.DataFrame([data.min(), data.max()], index=['负理想解', '正理想解'])
#
# 	# 距离
# 	weight = entropyWeight(data) if weight is None else np.array(weight)
# 	Result = data.copy()
# 	Result['正理想解'] = np.sqrt(((data - Z.loc['正理想解']) ** 2 * weight).sum(axis=1))
# 	Result['负理想解'] = np.sqrt(((data - Z.loc['负理想解']) ** 2 * weight).sum(axis=1))
#
# 	# 综合得分指数
# 	Result['综合得分指数'] = Result['负理想解'] / (Result['负理想解'] + Result['正理想解'])
# 	Result['排序'] = Result.rank(ascending=False)['综合得分指数']
#
# 	return Result, Z, weight
#
#
#
# data = pd.DataFrame(
#         {'人均专著': [0.1, 0.2, 0.4, 0.9, 1.2], '生师比': [5, 6, 7, 10, 2], '科研经费': [5000, 6000, 7000, 10000, 400],
#          '逾期毕业率': [4.7, 5.6, 6.7, 2.3, 1.8]}, index=['院校' + i for i in list('ABCDE')])
#
# import numpy as np
#
#
# def entropyWeight(data):
# 	data = np.array(data)
# 	# 归一化
# 	P = data / data.sum(axis=0)
#
# 	# 计算熵值
# 	E = np.nansum(-P * np.log(P) / np.log(len(data)), axis=0)
#
# 	# 计算权系数
# 	return (1 - E) / (1 - E).sum()
#
# entropyWeight(data)
#
#
# import numpy as np
#
# RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51}
#
#
# def ahp(data):
# 	data = np.array(data)
# 	m = len(data)
#
# 	# 计算特征向量
# 	weight = (data / data.sum(axis=0)).sum(axis=1) / m
#
# 	# 计算特征值
# 	Lambda = sum((weight * data).sum(axis=1) / (m * weight))
#
# 	# 判断一致性
# 	CI = (Lambda - m) / (m - 1)
# 	CR = CI / RI[m]
#
# 	if CR < 0.1:
# 		print(f'最大特征值：lambda = {Lambda}')
# 		print(f'特征向量：weight = {weight}')
# 		print(f'\nCI = {round(CI,2)}, RI = {RI[m]} \nCR = CI/RI = {round(CR,2)} < 0.1，通过一致性检验')
# 		return weight
# 	else:
# 		print(f'\nCI = {round(CI,2)}, RI = {RI[m]} \nCR = CI/RI = {round(CR,2)} >= 0.1，不满足一致性')


import numpy as np
import pandas as pd


# TOPSIS方法函数
def Topsis(A1):
	W0 = [0.2, 0.3, 0.4, 0.1]  # 权重矩阵
	W = np.ones([A1.shape[1], A1.shape[1]], float)
	for i in range(len(W)):
		for j in range(len(W)):
			if i == j:
				W[i, j] = W0[j]
			else:
				W[i, j] = 0
	Z = np.ones([A1.shape[0], A1.shape[1]], float)
	Z = np.dot(A1, W)  # 加权矩阵

	# 计算正、负理想解
	Zmax = np.ones([1, A1.shape[1]], float)
	Zmin = np.ones([1, A1.shape[1]], float)
	for j in range(A1.shape[1]):
		if j == 3:
			Zmax[0, j] = min(Z[:, j])
			Zmin[0, j] = max(Z[:, j])
		else:
			Zmax[0, j] = max(Z[:, j])
			Zmin[0, j] = min(Z[:, j])

	# 计算各个方案的相对贴近度C
	C = []
	for i in range(A1.shape[0]):
		Smax = np.sqrt(np.sum(np.square(Z[i, :] - Zmax[0, :])))
		Smin = np.sqrt(np.sum(np.square(Z[i, :] - Zmin[0, :])))
		C.append(Smin / (Smax + Smin))
	C = pd.DataFrame(C, index=['院校' + i for i in list('12345')])
	return C


# 标准化处理
def standard(A):
	# 效益型指标
	A1 = np.ones([A.shape[0], A.shape[1]], float)
	for i in range(A.shape[1]):
		if i == 0 or i == 2:
			if max(A[:, i]) == min(A[:, i]):
				A1[:, i] = 1
			else:
				for j in range(A.shape[0]):
					A1[j, i] = (A[j, i] - min(A[:, i])) / (max(A[:, i]) - min(A[:, i]))

		# 成本型指标
		elif i == 3:
			if max(A[:, i]) == min(A[:, i]):
				A1[:, i] = 1
			else:
				for j in range(A.shape[0]):
					A1[j, i] = (max(A[:, i]) - A[j, i]) / (max(A[:, i]) - min(A[:, i]))

				# 区间型指标
		else:
			a, b, lb, ub = 5, 6, 2, 12
			for j in range(A.shape[0]):
				if lb <= A[j, i] < a:
					A1[j, i] = (A[j, i] - lb) / (a - lb)
				elif a <= A[j, i] < b:
					A1[j, i] = 1
				elif b <= A[j, i] <= ub:
					A1[j, i] = (ub - A[j, i]) / (ub - b)
				else:  # A[i,:]< lb or A[i,:]>ub
					A1[j, i] = 0
	return A1


# 读取初始矩阵并计算
def data(file_path):
	data = pd.read_excel(file_path).values
	A = data[:, 1:]
	A = np.array(A)
	# m,n=A.shape[0],A.shape[1] #m表示行数,n表示列数
	return A

if __name__ == '__main__':
	# 权重
	# A = data('研究生院评估数据.xlsx')
	data_ori = pd.read_csv(r'E:\work\ai_work\oss_health\data\project_1486.csv')
	A = data_ori.drop(columns=['project_id', 'date', 'committer_id', 'req_merged'])
	# target = data_ori['req_merged']
	A1 = standard(A)
	C = Topsis(A1)
	print(C)
