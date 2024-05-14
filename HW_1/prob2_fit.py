import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
CSCI 635: Introduction to Machine Learning
Problem 2: Polynomial Regression &

@author/lecturer - Alexander G. Ororbia II

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
# @submitted by: Divyank Kulshrestha

# NOTE: you will need to tinker with the meta-parameters below yourself
#       (do not think of them as defaults by any means)
# meta-parameters for program
trial_name = 'p1_fit'  # will add a unique sub-string to output of this program
degree = 15  # p, order of model
beta = 0.0  # regularization coefficient
alpha = 0.05  # step size coefficient
eps = 0.00001  # controls convergence criterion
n_epoch = 10000  # number of epochs (full passes through the dataset)


# begin simulation

def regress(X, theta):
    b, w = theta
    return np.dot(X, w.T) + b


def gaussian_log_likelihood(mu, y):
    return np.sum((mu - y) ** 2)


def computeCost(X, y, theta, beta):  ## loss is now Gaussian Log likelihood
    size = X.shape[0]
    b, w = theta
    f = np.dot(X, w.T) + b
    return (gaussian_log_likelihood(f, y) + (beta * np.sum(w) ** 2)) / (2 * size)


def computeGrad(X, y, theta, beta):
    size = X.shape[0]
    b, w = theta
    f = np.dot(X, w.T) + b
    dL_db = np.sum(f - y) / size  # derivative w.r.t. model weights w
    dL_dw = (np.dot((f - y).T, X) / size) + ((beta * w) / size)  # derivative w.r.t model bias b
    nabla = (dL_db, dL_dw)  # nabla represents the full gradient
    return nabla


path = os.getcwd() + '/data/prob2.dat'
data = pd.read_csv(path, header=None, names=['X', 'Y'])
output = os.getcwd() + '/out/'

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:, 0:cols - 1]
y = data.iloc[:, cols - 1:cols]

# convert from data frames to numpy matrices
X = np.array(X.values)
y = np.array(y.values)

############################################################################
# apply feature map to input features x1
# WRITEME: write code to turn X_feat into a polynomial feature map (hint: you
#          could use a loop and array concatenation)
############################################################################
dss = X[:, -1]
for i in range(2, degree + 1):
    xnew = X[:, 0].T ** i
    xnew = xnew.reshape((X.shape[0], 1))
    X = np.concatenate((X, xnew), axis=1)
# convert to numpy arrays and initalize the parameter array theta
w = np.zeros((1, X.shape[1]))
b = np.array([0])
theta = (b, w)

L = computeCost(X, y, theta, beta)
halt = np.inf  # halting variable (you can use these to terminate the loop if you have converged)
i = 0
cost = []
cost.append(L)
while i < n_epoch and halt >= eps:
    dL_db, dL_dw = computeGrad(X, y, theta, beta)
    b = theta[0]
    w = theta[1]
    ############################################################################
    # update rules go here...
    # WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
    ############################################################################
    b = b.astype(float) - (alpha * dL_db)
    w = w.astype(float) - (alpha * dL_dw)
    theta = (b, w)
    L = computeCost(X, y, theta, beta)
    cost.append(L)
    # print(" {0} L = {1}".format(i, L))
    ############################################################################
    # WRITEME: write code to perform a check for convergence (or simply to halt early)
    ############################################################################
    if len(cost) >= 2:
        halt = cost[-2] - cost[-1]
    i += 1
# print parameter values found after the search
print("w = ", w)
print("b = ", b)

halt = cost[-2] - cost[-1]
if halt <= eps:
    print("Total number of epochs = ", n_epoch)
    print("Convergence at epoch = ", i - 1)

kludge = 0.25
# visualize the fit against the data
X_test = np.linspace(data.X.min(), data.X.max(), 100)
X_feat = np.expand_dims(X_test, axis=1)  # we need this otherwise, the dimension is missing (turns shape(value,) to shape(value,value))

############################################################################
# apply feature map to input features x1
# WRITEME: write code to turn X_feat into a polynomial feature map (hint: you
#          could use a loop and array concatenation)
############################################################################
for i in range(2, degree + 1):
    xnew = X_feat[:, 0].T ** i
    xnew = xnew.reshape((X_feat.shape[0], 1))
    X_feat = np.concatenate((X_feat, xnew), axis=1)

plt.plot(X_test, regress(X_feat, theta), label="Model")
plt.scatter(X[:, 0], y, edgecolor='g', s=20, label="Samples")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim((np.amin(X_test) - kludge, np.amax(X_test) + kludge))
plt.ylim((np.amin(y) - kludge, np.amax(y) + kludge))
plt.legend(loc="best")

############################################################################
# WRITEME: write your code here to save plot to disk (look up documentation or
#          the inter-webs for matplotlib)
############################################################################
plt.savefig(output + "prob2_regression_fit.png")
plt.show()

