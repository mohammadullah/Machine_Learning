#!/usr/bin/env python

import assignment1 as a1
import numpy as np
import matplotlib.pyplot as plt


(countries, features, values) = a1.load_unicef_data()

targets = values[:,1]
x = values[:,7:]
#x = a1.normalize_data(x)

N_TRAIN = 100;
# Select a single feature.
x_train = x[0:N_TRAIN,3:6]
t_train = targets[0:N_TRAIN]
x_test = x[N_TRAIN:,3:6]
t_test = targets[N_TRAIN:]

x_ev = np.zeros((500,3))
y_ev = np.zeros((500,3))
xte_ev = np.zeros((500,3))
yte_ev = np.zeros((500,3))
# TO DO:: Complete the linear_regression and evaluate_regression functions of the assignment1.py

#fig, axs = plt.subplots(nrows=3, ncols=1)

for n in range(3):
	(w_tr, tr_err) = a1.linear_regression(x=x_train[:,n], t=t_train, basis = 'polynomial', degree=3, dflag=1, w=0)
	x_ev[:,n] = np.linspace(np.asscalar(min(x_train[:,n])), np.asscalar(max(x_train[:,n])), num=500)
	xte_ev[:,n] = np.linspace(np.asscalar(min(x_test[:,n])), np.asscalar(max(x_test[:,n])), num=500)
	w_tr = np.asarray(w_tr)
	(y_ev[:,n], err)  = a1.evaluate_regression(x=x_ev[:,n], t=t_train, w=w_tr, basis='polynomial', degree=3)
	(yte_ev[:,n], err)  = a1.evaluate_regression(x=xte_ev[:,n], t=t_train, w=w_tr, basis='polynomial', degree=3)


fig, axs = plt.subplots(nrows=3, ncols=2)
ax = axs[0,0]
ax.plot(x_ev[:,0],y_ev[:,0],'r.-')
ax.plot(x_train[:,0],t_train,'bo')
ax.set_title('Polynomial fit on Training data')
ax.legend(['Trining Fit', 'Training points'])
ax.set_xlabel('GNI')
ax.set_ylabel('Mortality Rate')

ax = axs[1,0]
ax.plot(x_ev[:,1],y_ev[:,1],'r.-')
ax.plot(x_train[:,1],t_train,'bo')
ax.set_xlabel('Life Expectancy')
ax.set_ylabel('Mortality Rate')

ax = axs[2,0]
ax.plot(x_ev[:,2],y_ev[:,2],'r.-')
ax.plot(x_train[:,2],t_train,'bo')
ax.set_xlabel('Literacy')
ax.set_ylabel('Mortality Rate')

ax = axs[0,1]
ax.plot(xte_ev[:,0], yte_ev[:,0], 'm.-')
ax.plot(x_test[:,0], t_test, 'go')
ax.set_title('Polynomial fit on Test data')
ax.legend(['Fit', 'Test points'])
ax.set_xlabel('GNI')

ax = axs[1,1]
ax.plot(xte_ev[:,1], yte_ev[:,1], 'm.-')
ax.plot(x_test[:,1], t_test, 'go')
ax.set_xlabel('Life Expectancy')


ax = axs[2,1]
ax.plot(xte_ev[:,2], yte_ev[:,2], 'm.-')
ax.plot(x_test[:,2], t_test, 'go')
ax.set_xlabel('Literacy')

plt.tight_layout()
plt.savefig("Polynomial_fit.png", format="png", dpi=1000)
plt.show()
