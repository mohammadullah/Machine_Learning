#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:15]
feature = ('TotPopulation', 'NumofBirth', 'Under5Death', 'GniCapital', 'LifeExpectency', 'LiteracyRate', 'SchoolEnrolment', 'BirthWeight')
#x = a1.normalize_data(x)

N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

train_err = np.zeros(8)
test_err = np.zeros(8)

# TO DO:: Complete the linear_regression and evaluate_regression functions of the assignment1.py

for n in range(8):

	(w_tr, tr_err) = a1.linear_regression(x=x_train[:,n], t=t_train, basis = 'polynomial', degree=3, dflag=1, w=0)
	te_err = a1.linear_regression(x=x_test[:,n], t=t_test, basis = 'polynomial', degree=3, dflag=0, w=w_tr)
	train_err[n] = tr_err
	test_err[n] = te_err


##tt = a1.linear_regression(x=x_train, t=x_test, basis = 'polynomial', reg_lambda=0, degree=1)
#(t_est, te_err) = a1.evaluate_regression()

index = np.arange(8)
bar_width = 0.35
# Produce a plot of results.
#plt.plot(train_err.keys(), train_err.values())
fig, ax = plt.subplots()
#ax = axs[0]
ax.bar(index, train_err, bar_width)
ax.bar(index + bar_width, test_err, bar_width)
#ax.plot(p_degree, train_err)
#ax.plot(p_degree, test_err)
ax.set_title('Features, no regularization, not normalized')
ax.legend(['Training error', 'Test error'])
ax.set_xlabel('Features')
ax.set_ylabel('RMS')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(feature, rotation = 'vertical')

plt.tight_layout()

plt.savefig("Feature_RMS_barplot.png", format="png", dpi=1000)
plt.show()

import visualize_1d as v1
