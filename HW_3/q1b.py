
import matplotlib.pyplot as plt
import os
import pandas as pd
from helpers import *

beta = 0        # regularization coefficient
alpha = 0.001        # step size coefficient
n_epoch = 40000     # number of epochs (full passes through the dataset)
eps = 0.00000      # controls convergence criterion

path = os.getcwd() + '/data/spiral_train.dat'
data = pd.read_csv(path, header=None)
output = os.getcwd() + '/out/'

# set X and y
cols = data.shape[1]
X = data.iloc[:, 0:-1]
Y = data.iloc[:, -1]

y2 = Y.values
y2 = np.vstack(y2)
Y = pd.get_dummies(Y)
np.random.seed(1)

# convert to numpy arrays and initialize the parameter array theta
X = np.array(X.values)
Y = np.array(Y.values)
# w = np.zeros((X.shape[1],Y.shape[1]))

instances = X.shape[0]
attributes = X.shape[1]
hiddenNodes = 4
outputLabels = 3

w1 = np.random.rand(attributes, hiddenNodes)
b1 = np.random.randn(hiddenNodes)
w2 = np.random.rand(hiddenNodes, outputLabels)
b2 = np.random.randn(outputLabels)
theta2 = (b1, w1, b2, w2)

L = computeCost(X, Y, theta2, beta)
halt = np.inf   # halting variable (you can use these to terminate the loop if you have converged)

############################################################################
# Training the model
############################################################################
print("_____________________________________________________")
print("TRAINING THE MODEL... ")
i = 0
cost = []
cost.append(L)
while i < n_epoch:
    dL_db1, dL_dw1, dL_db2, dL_dw2 = computeGrad(X, Y, theta2, beta)
    b1 = theta2[0]
    w1 = theta2[1]
    b2 = theta2[2]
    w2 = theta2[3]
    ############################################################################
    # update rules go here...
    # WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
    ############################################################################
    b1 = b1 - (alpha * dL_db1)
    w1 = w1 - (alpha * dL_dw1)
    b2 = b2 - (alpha * dL_db2)
    w2 = w2 - (alpha * dL_dw2)
    theta2 = (b1, w1, b2, w2)
    L = computeCost(X, Y, theta2, beta)
    ############################################################################
    # WRITEME: write code to perform a check for convergence (or simply to halt early)
    ############################################################################
    cost.append(L)
    if len(cost) >= 2:
        halt = cost[-2] - cost[-1]
    # print(" {0} L = {1}".format(i, L))
    i += 1

# print("w = ", w)
# print("b = ", b)
halt = cost[-2] - cost[-1]
if halt <= eps:
    print("Total number of epochs = ", n_epoch)
    print("Convergence at epoch = ", i - 1)
############################################################################
predictions = predict(X, theta2)
# compute error (100 - accuracy)
# WRITEME: write your code here calculate your actual classification error (using the "predictions" variable)
y2 = np.argmax(Y, axis=1)
N = predictions.shape[0]
y = predictions
accuracy = (y2 == y).sum() / N
print('Accuracy = {0}%'.format(accuracy * 100.))
err = 1 - accuracy
print('Error = {0}%'.format(err * 100.))
############################################################################
h = 0.001
cmap = 'RdBu'
x_min = X[:, 0].min() - 100 * h
x_max = X[:, 0].max() + 100 * h
y_min = X[:, 1].min() - 100 * h
y_max = X[:, 1].max() + 100 * h
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# print(np.c_[xx.ravel(), yy.ravel()])
Z = predict(np.c_[xx.ravel(), yy.ravel()], theta2)
Z = Z.reshape(xx.shape)

############################################################################
plt.plot(range(len(cost)), cost)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.savefig(output + "q1b_loss_vs_epoch.png")
plt.show()

plt.figure(figsize=(7, 7))
plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.9)
plt.contour(xx, yy, Z, colors='k', linewidths=1)
plt.scatter(X[:,0], X[:,1], c=y, cmap=cmap, edgecolors='k');
plt.title("Decision Boundary")
plt.savefig(output + "q1b_decision_boundary.png")
plt.show()