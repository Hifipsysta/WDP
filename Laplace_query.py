
import ot
import torch
import scipy.stats
import scipy.special
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import kl_divergence, renyi_divergence, wasserstein_distance

'''
m = nn.Conv1d(1, 33, 1, stride=2)
input = torch.randn(33,1,1)
output = m(input)
print(output.shape)
'''

dataset = pd.read_csv("ADULT/adult.csv",
    names=["Age", "Workclass", "fnlwgt", "Education", "Education-Num", "Martial Status",
        "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
        "Hours per week", "Country", "Target"],sep=r'\s*,\s*',na_values="?")

datacount = dataset["Age"].value_counts()


def Laplace_mechanism(scale, size, distribution_ = True):
	X = np.random.laplace(loc=1, scale=scale, size=size)	
	Y = np.random.laplace(loc=0, scale=scale, size=size)

	X = X + datacount
	Y = Y + datacount

	X_dist = scipy.special.softmax(X)
	Y_dist = scipy.special.softmax(Y)

	'''
	print('wasserstein_distance 1=', wasserstein_distance(X,Y))
	print('Renyi_Divergence=', renyi_divergence(1.2,X,Y))
	print('KL_divergence=', kl_divergence(X,Y))

	print('Scipy_KL=', scipy.stats.entropy(pk=X,qk=Y))
	print('wasserstein_distance 2=', wasserstein_distance(X,Y))
	'''
	if distribution_:
		X_out, Y_out = X_dist, Y_dist
	else:
		X_out, Y_out = X, Y
	return X_out, Y_out





def get_different_scale_order_renyi_wasser(order_start, order_end, order_num, scale_list, size):
	renyi_list = [[] for i in range(len(scale_list))]
	order_list = [[] for i in range(len(scale_list))]
	wasser_list = [[] for i in range(len(scale_list))]

	#plt.figure()

	for i in range(len(scale_list)):
		for order in np.linspace(order_start, order_end, order_num):
			order_list[i].append(order)

			X1,Y1 = Laplace_mechanism(scale=scale_list[i], size=size, distribution_=True) 
			X2,Y2 = Laplace_mechanism(scale=scale_list[i], size=size, distribution_=False)

			renyi_list[i].append(renyi_divergence(alpha=order, P=X1, Q=Y1).item())
			wasser_list[i].append(ot.wasserstein_1d(x_a=X2,x_b=Y2, p=order).item())

			#renyi_theory_list.append(RDP_Laplace_epsilon(alpha=order, scale=scale))
			#wasser_list.append(wasserstein_distance(X2,Y2).item())
			#KL1.append(scipy.stats.entropy(pk=X1,qk=Y1))

	return renyi_list, wasser_list, order_list


def plot_different_scale_order_renyi(renyi_list, order_list, scale_list):
	for i in range(len(scale_list)):
		plt.plot(
			order_list[i], renyi_list[i], 
			label='$\lambda$='+str(scale_list[i]), 
			alpha=0.6
			)
		#plt.plot(order_list, renyi_theory_list)

	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.xlabel('Order', fontsize=17)
	plt.ylabel('Privacy loss', fontsize=17)
	plt.legend(fontsize=15, loc=2)

	plt.savefig('result_dir/query/Adult_Laplace_RDP_diff_order.png', 
		bbox_inches='tight')
	plt.show()


def plot_different_scale_order_wasser(wasser_list, order_list, scale_list):
	for i in range(len(scale_list)):
		plt.plot(
			order_list[i], wasser_list[i], 
			label='$\lambda$='+str(scale_list[i]), 
			alpha=0.75
			)
		#plt.plot(order_list, renyi_theory_list)
	
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.xlabel('Order', fontsize=17)
	plt.ylabel('Privacy loss', fontsize=17)
	plt.legend(fontsize=15, loc=2)

	plt.savefig('result_dir/query/Adult_Laplace_WDP_diff_order.png',
		bbox_inches='tight')
	plt.show()


renyi_list, wasser_list, order_list = get_different_scale_order_renyi_wasser(
	order_start=1, 
	order_end=10, 
	order_num=1000, 
	scale_list=[1,2,4],
	size=len(list(datacount))
	)

plot_different_scale_order_renyi(renyi_list, order_list, scale_list=[1,2,4])
plot_different_scale_order_wasser(wasser_list, order_list, scale_list=[1,2,4])


def plot_different_scale_wasserstein(scale_start, scale_end, scale_num, size):
	wasser_list, scale_list = [],[]
	renyi_list, wasser2_list = [], []
	for scale in np.linspace(scale_start, scale_end, scale_num):
		scale_list.append(scale)
		X1,Y1 = Laplace_mechanism(scale=scale, size=size, distribution_=True) 
		X2,Y2 = Laplace_mechanism(scale=scale, size=size, distribution_=False)
		#wasser_list.append(wasserstein_distance(X2, Y2))
		renyi_list.append(renyi_divergence(alpha = 1,P = X1, Q = Y1))
		wasser2_list.append(ot.wasserstein_1d(x_a = X2, x_b = Y2, p = 1))
	#print(renyi_list)

	plt.figure()
	#plt.plot(scale_list, wasser_list, label='Wasserstein')
	plt.plot(scale_list, renyi_list, label='Renyi')
	plt.plot(scale_list, wasser2_list, label='Wasserstein')
	plt.xticks(fontsize=15)
	plt.yticks(fontsize=15)
	plt.xlabel('Scale', fontsize=17)
	plt.ylabel('Privacy loss', fontsize=17)
	plt.legend(fontsize=15, loc=2)
	plt.show()


plot_different_scale_wasserstein(scale_start = 1, scale_end = 5, scale_num = 100, size = len(list(datacount)))




