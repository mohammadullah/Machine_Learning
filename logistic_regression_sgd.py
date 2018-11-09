#!/usr/bin/env python

# Run logistic regression training.

import numpy as np
import scipy.special as sps
import matplotlib.pyplot as plt
import assignment2 as a2


# Maximum number of iterations.  Continue until this limit, or when error change is below tol.
max_iter = 500
tol = 0.00001
np.random.seed(123)

# Load data.
data = np.genfromtxt('data.txt')

# Data matrix, with column of ones at end.
X = data[:,0:3]
# Target values, 0 for class 1, 1 for class 2.
t = data[:,3]
# For plotting data
class1 = np.where(t==0)
X1 = X[class1]
class2 = np.where(t==1)
X2 = X[class2]

lrates = np.array([0.5, 0.3, 0.1, 0.05, 0.01])
nrow = len(t)

#fig, axs = plt.subplots(nrows=5, ncols=1)
fig = plt.figure()
ax1 = fig.add_subplot(111)


for iter1 in range (0, 5):
  
  # Step size for gradient descent.
  eta = lrates[iter1]

  # Initialize w.
  w = np.array([0.1, 0, 0])

  # Error values over all iterations.
  e_all = []
  ep_all=[]
  flg = 0

  for iter in range (0,max_iter):
    

    for i in range(nrow):

      rand_row = np.random.randint(0, nrow)

      Xi = X[rand_row,:]
      ti = t[rand_row]

      y = sps.expit(np.dot(Xi,w))
  
      # e is the error, negative log-likelihood (Eqn 4.90)
      e = -(ti*np.log(y+0.00000001) + (1-ti)*np.log(1-y+0.00000001))

      # Add this error to the end of error vector.
      e_all.append(e)

      # Gradient of the error, using Eqn 4.91
      grad_e = np.multiply((y - ti), Xi)

      # Update w, *subtracting* a step in the error derivative since we're minimizing
      w = w - eta*grad_e

      # Stop iterating if error doesn't change more than tol.
      if i>0:
        if np.absolute(e-e_all[i-1]) < tol:
          flg = 1
          break
        
    ep_all.append(e)
    if flg == 1:
      break


  # Plot error over iterations

  #ax = axs[iter1]
  ax1.plot(ep_all, label = 'rate  ' + str(eta))
  ax1.set_title('Training logistic regression')
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Negative log likelihood')
  ax1.legend(loc = 'upper right')
  plt.tight_layout()

plt.show()
