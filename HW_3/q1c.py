
import matplotlib.pyplot as plt
import os
import pandas as pd
from helpers import *

beta = 1e-3  # regularization coefficient
alpha = 0.001  # step size coefficient
n_epoch = 1500  # number of epochs (full passes through the dataset)
eps = 0  # controls convergence criterion

# begin simulation

path = os.getcwd() + '/data/iris_train.dat'
trainingData = pd.read_csv(path, header=None)
path = os.getcwd() + '/data/iris_test.dat'
testingData = pd.read_csv(path, header=None)
output = os.getcwd() + '/out/'

cols = trainingData.shape[1]
trainX = trainingData.iloc[:, 0:-1]
trainY = trainingData.iloc[:, -1]

trainY2 = trainY.values
trainY2 = np.vstack(trainY2)
trainY = pd.get_dummies(trainY)
np.random.seed(1)

testX = testingData.iloc[:, 0:-1]
testY = testingData.iloc[:, -1]
testY = pd.get_dummies(testY)

# convert to numpy arrays and initialize the parameter array theta
trainX = np.array(trainX.values)
trainY = np.array(trainY.values)
testX = np.array(testX.values)
testY = np.array(testY.values)
# w = np.zeros((trainX.shape[1],trainY.shape[1]))

instances = trainX.shape[0]
attributes = trainX.shape[1]
hiddenNodes = 4
outputLabels = 3

w1 = np.random.rand(attributes, hiddenNodes)
b1 = np.random.randn(hiddenNodes)
w2 = np.random.rand(hiddenNodes, outputLabels)
b2 = np.random.randn(outputLabels)
theta2 = (b1, w1, b2, w2)

L = computeCost(trainX, trainY, theta2, beta)
halt = np.inf  # halting variable (you can use these to terminate the loop if you have converged)

############################################################################
# Training the model
############################################################################
print("_____________________________________________________")
print("TRAINING THE MODEL... ")
i = 0
trainCost = []
testCost = []
trainCost.append(L)
while (i < n_epoch):
    batches = create_mini_batch(trainX, trainY, 32)
    counter = 1
    for x, y in batches:
        dL_db1, dL_dw1, dL_db2, dL_dw2 = computeGrad(trainX, trainY, theta2, beta)
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
        L = computeCost(trainX, trainY, theta2, beta)
        testL = computeCost(testX, testY, theta2, beta)
        ############################################################################
        # WRITEME: write code to perform a check for convergence (or simply to halt early)
        ############################################################################
        trainCost.append(L)
        testCost.append(testL)  # TODO
        if len(trainCost) >= 2:
            halt = trainCost[-2] - trainCost[-1]
        print("Batch {0} -> Loss = {1}".format(counter, L))
        counter += 1
    print("Epoch {0} -> Loss = {1}".format(i, L))
    i += 1

# print("\nw = ", w)
# print("b = ", b)
halt = trainCost[-2] - trainCost[-1]
if halt <= eps:
    print("Total number of epochs = ", n_epoch)
    print("Convergence at epoch = ", i - 1)

############################################################################
predictions = predict(trainX, theta2)
testPredictions = predict(testX, theta2)
# compute error (100 - accuracy)
# WRITEME: write your code here calculate your actual classification error (using the "predictions" variable)
trainY2 = np.argmax(trainY, axis=1)
N = predictions.shape[0]
y = predictions
print("_____________________________________________________")
print("TRAINING ACCURACY")
accuracy = (trainY2 == y).sum() / N
print('Accuracy = {0}%'.format(accuracy * 100.))
err = 1 - accuracy
print('Error = {0}%'.format(err * 100.))

testY2 = np.argmax(testY, axis=1)
testN = testPredictions.shape[0]
testy = testPredictions
print("_____________________________________________________")
print("TESTING ACCURACY")
testAccuracy = (testY2 == testy).sum() / testN
print('Accuracy = {0}%'.format(testAccuracy * 100.))
err = 1 - testAccuracy
print('Error = {0}%'.format(err * 100.))

############################################################################
plt.plot(range(len(trainCost)), trainCost)
plt.plot(range(len(testCost)), testCost)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss curve")
plt.savefig(output + "q1c_loss_vs_epoch_superimposed.png")
plt.show()
