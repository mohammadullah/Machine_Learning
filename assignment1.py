"""Basic code for assignment 1."""

import numpy as np
import pandas as pd
import scipy.stats as stats

def load_unicef_data():
    """Loads Unicef data from CSV file.

    Retrieves a matrix of all rows and columns from Unicef child mortality
    dataset.

    Args:
      none

    Returns:
      Country names, feature names, and matrix of values as a tuple (countries, features, values).

      countries: vector of N country names
      features: vector of F feature names
      values: matrix N-by-F
    """
    fname = 'SOWC_combined_simple.csv'

    # Uses pandas to help with string-NaN-numeric data.
    data = pd.read_csv(fname, na_values='_')
    # Strip countries title from feature names.
    features = data.axes[1][1:]
    # Separate country names from feature values.
    countries = data.values[:,0]
    values = data.values[:,1:]
    # Convert to numpy matrix for real.
    values = np.asmatrix(values,dtype='float64')

    # Modify NaN values (missing values).
    mean_vals = np.nanmean(values, axis=0)
    inds = np.where(np.isnan(values))
    values[inds] = np.take(mean_vals, inds[1])
    return (countries, features, values)


def normalize_data(x):
    """Normalize each column of x to have mean 0 and variance 1.
    Note that a better way to normalize the data is to whiten the data (decorrelate dimensions).  This can be done using PCA.

    Args:
      input matrix of data to be normalized

    Returns:
      normalized version of input matrix with each column with 0 mean and unit variance

    """
    mvec = x.mean(0)
    stdvec = x.std(axis=0)
    
    return (x - mvec)/stdvec
    


def linear_regression(x, t, basis, degree, dflag, w):
    """Perform linear regression on a training set with specified regularizer lambda and basis

    Args:
      x is training inputs
      t is training targets
      reg_lambda is lambda to use for regularization tradeoff hyperparameter
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)

    Returns:
      w vector of learned coefficients
      train_err RMS error on training set
      """

    # TO DO:: Complete the design_matrix function.
    phi = design_matrix(x, basis, degree, dflag)

    # TO DO:: Compute coefficients using phi matrix

    if (dflag == 1):
      w = np.matmul(np.linalg.pinv(phi),t)
    
    y_w = np.matmul(phi, w)
    #np.savetxt('w_text.txt', y_w)
    #np.savetxt('phi_text.txt', t)
    # Measure root mean squared error on training data.
    E = 0.5*np.sum(np.power(np.subtract(y_w,t),2))
    
    if dflag == 1:
      train_err = np.sqrt(2*E/100)
      return (w, train_err)
    else:
      test_err = np.sqrt(2*E/len(x))
      return test_err

def design_matrix(x, basis, degree, dflag):
    """ Compute a design matrix Phi from given input datapoints and basis.
	Args:
      x matrix of input datapoints
      basis string name of basis
    
    Returns:
      phi design matrix
    """
    y = x
    if dflag == 1:
      x0 = np.ones((100,1))
    else:
      x0 = np.ones((len(x),1))

    # TO DO:: Compute desing matrix for each of the basis functions
    if basis == 'polynomial':
        if degree == 1:
          phi = np.hstack((x0,y))
        else:
          for i in range(2,degree+1):
            temp_x = np.power(x, i)
            y = np.hstack((y, temp_x))
          phi = np.hstack((x0,y))

    elif basis == 'ReLU':
        phi = None
    else: 
        assert(False), 'Unknown basis %s' % basis
    
    ##np.savetxt('test_1.txt', phi, fmt='%.2f')

    return phi


def evaluate_regression(x, t, w, basis, degree):
    """Evaluate linear regression on a dataset.
  Args:
      x is evaluation (e.g. test) inputs
      w vector of learned coefficients
      basis is string, name of basis to use
      degree is degree of polynomial to use (only for polynomial basis)
      t is evaluation (e.g. test) targets

    Returns:
      t_est values of regression on inputs
      err RMS error on the input dataset 
      """
    # TO DO:: Compute t_est and err 

    t_est = w[0] + x*w[1] + np.power(x, 2)*w[2] + np.power(x, 3)*w[3]
    err = None

    return (t_est, err)
