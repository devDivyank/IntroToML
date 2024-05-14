
import matplotlib.pyplot as plt
import os
import pandas as pd
from helpers import *

beta = 1e-4         # regularization coefficient
alpha = 0.01        # step size coefficient
n_epoch = 10000     # number of epochs (full passes through the dataset)
eps = 0.00001      # controls convergence criterion

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
# w = np.zeros((X.shape[1],y2.shape[1]))
w = np.random.rand(X.shape[1], Y.shape[1])
b = np.zeros((1, Y.shape[1]))
theta2 = (b, w)

L = computeCost(X, Y, theta2, beta)
halt = np.inf   # halting variable (you can use these to terminate the loop if you have converged)
i = 0
cost = []
cost.append(L)
while (i < n_epoch and halt >= eps):
    dL_db, dL_dw = computeGrad(X, Y, theta2, beta)
    b = theta2[0]
    w = theta2[1]
    ############################################################################
    # update rules go here...
    # WRITEME: write your code here to perform a step of gradient descent & record anything else desired for later
    ############################################################################
    b = b - (alpha * dL_db)
    w = w - (alpha * dL_dw)
    theta2 = (b, w)
    L = computeCost(X, Y, theta2, beta)
    ############################################################################
    # WRITEME: write code to perform a check for convergence (or simply to halt early)
    ############################################################################
    cost.append(L)
    if len(cost) >= 2:
        halt = cost[-2] - cost[-1]
    i += 1

print("w = ", w)
print("b = ", b)
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
plt.title("Loss curve")
plt.savefig(output + "q1a_loss_vs_epoch.png")
plt.show()

############################################################################
# Gradient checking - Derivatives using limits proof
############################################################################
w = np.zeros((X.shape[1],Y.shape[1]))
b = np.zeros((1,Y.shape[1]))
wInitial = w
bInitial = b
theta = (b, w)
delta = 1e-4
dL_db, dL_dw = computeGrad(X, Y, theta, beta)

secantBias = np.zeros(b.shape)
secantWeights = np.zeros(w.shape)

print("_____________________________________________________")
print("CHECKING BIASES")
for x, y in np.ndindex(b.shape):
    b = bInitial
    b[x,y] = b[x,y] - delta
    theta = (b, w)
    first = computeCost(X, Y, theta, beta)
    b = bInitial
    b[x,y] = b[x,y] + delta
    theta = (b, w)
    second = computeCost(X, Y, theta, beta)
    final = (second-first) / (delta * 2)
    secantBias[x,y] = final

print("Individual biases: ")
extra = abs(secantBias - dL_db) <= 1e-4
extra = extra.tolist()
for i in extra:
    for j in i:
        if j == True:
            print("CORRECT", end=" ")
        else:
            print(j, end=" ")
    print()

if (abs(secantBias - dL_db) <= 1e-4).all():
    print("\nBias derivative is CORRECT")

print("_____________________________________________________")
print("CHECKING WEIGHTS")

for x, y in np.ndindex(w.shape):
    w = wInitial
    w[x,y] = w[x,y] - delta
    theta = (b, w)
    first = computeCost(X, Y, theta, beta)
    w = wInitial
    w[x,y] = w[x,y] + delta
    theta = (b, w)
    second = computeCost(X, Y, theta, beta)
    final = (second-first) / (delta * 2)
    secantWeights[x,y] = final

print("Individual Weights: ")

extra = abs(secantWeights - dL_dw) <= 1e-4
extra = extra.tolist()
for i in extra:
    for j in i:
        if j == True:
            print("CORRECT", end=" ")
        else:
            print(j, end=" ")
    print()

if (abs(secantWeights - dL_dw) <= 1e-4).all():
    print("\nWeights derivative is CORRECT")