
import matplotlib.pyplot as plt
import os
import pandas as pd
from helpers import *

beta = 0        # regularization coefficient
alpha = 0.1        # step size coefficient
n_epoch = 2000     # number of epochs (full passes through the dataset)
eps = 0.00000      # controls convergence criterion

path = os.getcwd() + '/data/xor.dat'
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
hiddenNodes = 3
outputLabels = 2

w1 = np.random.rand(attributes, hiddenNodes)
b1 = np.random.randn(hiddenNodes)
w2 = np.random.rand(hiddenNodes, outputLabels)
b2 = np.random.randn(outputLabels)
theta2 = (b1, w1, b2, w2)

L = computeCost(X, Y, theta2, beta)
halt = np.inf   # halting variable (you can use these to terminate the loop if you have converged)

############################################################################
# Gradient checking - Derivatives using limits proof
############################################################################
delta = 1e-4
w = np.zeros((X.shape[1], Y.shape[1]))
b = np.zeros((1, Y.shape[1]))
w1 = np.zeros((attributes, hiddenNodes))
b1 = np.zeros((1, hiddenNodes))
w2 = np.zeros((hiddenNodes, outputLabels))
b2 = np.zeros((1, outputLabels))

b1SecantDerivative = np.zeros(b1.shape)
w1SecantDerivative = np.zeros(w1.shape)
b2SecantDerivative = np.zeros(b2.shape)
w2SecantDerivative = np.zeros(w2.shape)
b1Initial = b1
w1Initial = w1
b2Initial = b2
w2Initial = w2

theta = (b1, w1, b2, w2)
dL_db1, dL_dw1, dL_db2, dL_dw2 = computeGrad(X, Y, theta, beta)

print("_____________________________________________________")
print("CHECKING BIASES")
for x, y in np.ndindex(b1.shape):
    b1 = b1Initial
    b1[x,y] = b1[x,y] - delta
    theta = (b1, w1, b2, w2)
    first = computeCost(X, Y, theta, beta)
    b1 = b1Initial
    b1[x,y] = b1[x,y] + delta
    theta = (b1, w1, b2, w2)
    second = computeCost(X, Y, theta, beta)
    final = (second - first) / (delta * 2)
    b1SecantDerivative[x,y] = final

print("Individual bias 1 hidden layer: ")
extra = abs(b1SecantDerivative - dL_db1) <= 1e-4
extra = extra.tolist()
for i in extra:
    for j in i:
        if j == True:
            print("CORRECT", end=" ")
        else:
            print(j, end=" ")
    print()

if (abs(b1SecantDerivative - dL_db1) <= 1e-4).all():
    print("Bias derivative for hidden layer CORRECT\n")

for x, y in np.ndindex(b2.shape):
    b2 = b2Initial
    b2[x,y] = b2[x,y] - delta
    theta = (b1, w1, b2, w2)
    first = computeCost(X, Y, theta, beta)
    b2 = b2Initial
    b2[x,y] = b2[x,y] + delta
    theta = (b1, w1, b2, w2)
    second = computeCost(X, Y, theta, beta)
    final = (second - first) / (delta * 2)
    b2SecantDerivative[x,y] = final

print("Individual bias 2 output layer: ")
extra = abs(b2SecantDerivative - dL_db2) <= 1e-4
extra = extra.tolist()
for i in extra:
    for j in i:
        if j == True:
            print("CORRECT", end=" ")
        else:
            print(j, end=" ")
    print()

if (abs(b1SecantDerivative - dL_db1) <= 1e-4).all():
    print("Bias derivative for output layer CORRECT")

print("_____________________________________________________")
print("CHECKING WEIGHTS")
for x, y in np.ndindex(w1.shape):
    w1 = w1Initial
    w1[x,y] = w1[x,y] - delta
    theta = (b1, w1, b2, w2)
    first = computeCost(X, Y, theta, beta)
    w1 = w1Initial
    w1[x,y] = w1[x,y] + delta
    theta = (b1, w1, b2, w2)
    second = computeCost(X, Y, theta, beta)
    final = (second-first) / (delta * 2)
    w1SecantDerivative[x,y] = final

print("Individual weights 1 hidden layer: ")
extra = abs(w1SecantDerivative - dL_dw1) <= 1e-4
extra = extra.tolist()
for i in extra:
    for j in i:
        if j == True:
            print("CORRECT", end=" ")
        else:
            print(j, end=" ")
    print()

if (abs(w1SecantDerivative - dL_dw1) <= 1e-4).all():
    print("\nWeights derivative for hidden layer CORRECT")

for x, y in np.ndindex(w2.shape):
    w2 = w2Initial
    w2[x,y] = w2[x,y] - delta
    theta = (b1, w1, b2, w2)
    first = computeCost(X, Y, theta, beta)
    w2 = w2Initial
    w2[x,y] = w2[x,y] + delta
    theta = (b1, w1, b2, w2)
    second = computeCost(X, Y, theta, beta)
    final = (second-first) / (delta * 2)
    w2SecantDerivative[x,y] = final

print("Individual weights 2 output layer: ")
extra = abs(w2SecantDerivative - dL_dw2) <= 1e-4
extra = extra.tolist()
for i in extra:
    for j in i:
        if j == True:
            print("CORRECT", end=" ")
        else:
            print(j, end=" ")
    print()

if (abs(w1SecantDerivative - dL_dw1) <= 1e-4).all():
    print("\nWeights derivative for output layer CORRECT")


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
plt.plot(range(len(cost)), cost)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.savefig(output + "q1a_loss_vs_epoch.png")
plt.show()

