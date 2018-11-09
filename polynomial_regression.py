#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt

(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]


N_TRAIN = 100;
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
t_train = targets[0:N_TRAIN]
t_test = targets[N_TRAIN:]

train_err = np.zeros(6)
test_err = np.zeros(6)

# TO DO:: Complete the linear_regression and evaluate_regression functions of the assignment1.py

for n in range(1,7):

	(w_tr, tr_err) = a1.linear_regression(x=x_train, t=t_train, basis = 'polynomial', degree=n, dflag=1, w=0)
	te_err = a1.linear_regression(x=x_test, t=t_test, basis = 'polynomial', degree=n, dflag=0, w=w_tr)
	train_err[n-1] = tr_err
	test_err[n-1] = te_err

x = a1.normalize_data(x)
x_train = x[0:N_TRAIN,:]
x_test = x[N_TRAIN:,:]
ntrain_err = np.zeros(6)
ntest_err = np.zeros(6)


for n in range(1,7):

	(w_tr, tr_err) = a1.linear_regression(x=x_train, t=t_train, basis = 'polynomial', degree=n, dflag=1, w=0)
	te_err = a1.linear_regression(x=x_test, t=t_test, basis = 'polynomial', degree=n, dflag=0, w=w_tr)
	ntrain_err[n-1] = tr_err
	ntest_err[n-1] = te_err


p_degree = range(1,7)

fig, axs = plt.subplots(nrows=2, ncols=1)
ax = axs[0]
ax.plot(p_degree, train_err)
ax.plot(p_degree, test_err)
ax.set_title('Fit with polynomials, no regularization, not normalized')
ax.legend(['Training error', 'Test error'])
ax.set_xlabel('Polynomial degree')
ax.set_ylabel('RMS')

ax = axs[1]
ax.plot(p_degree, ntrain_err)
ax.plot(p_degree, ntest_err)
ax.set_title('Fit with polynomials, no regularization, normalized')
ax.legend(['Training error', 'Test error'])
ax.set_xlabel('Polynomial degree')
ax.set_ylabel('RMS')

plt.tight_layout()
plt.savefig("RMS_error.png", format="png", dpi=1000)
plt.show()
