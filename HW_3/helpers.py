import numpy as np


def softmax(z):
    denom = np.sum(np.exp(z), axis=1)
    return np.exp(z) / np.reshape(denom, (denom.shape[0], 1))


def predict(X, theta):
    p, q = regress(X, theta)
    return np.argmax(p, axis=1)


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def sigmoidDerivative(X):
    return sigmoid(X) * (1 - sigmoid(X))


def dRelu(X):
    X[X <= 0] = 0
    X[X > 0] = 1
    return X


def Relu(Z):
    return np.maximum(0, Z)


def regress(X, theta):
    b1, w1, b2, w2 = theta
    z1 = np.dot(X, w1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = softmax(z2)

    return a2, a1


def logLikelihood(p, y):
    size = y.shape[0]
    return -1 * np.sum(np.sum(np.multiply(y, np.log(p)))) / size


def computeCost(X, y, theta, beta):
    p, q = regress(X, theta)
    b1, w1, b2, w2 = theta
    loss = logLikelihood(p, y)
    reg = beta * (np.sum(w1**2) + np.sum(w2**2)) / 2
    return loss + reg


def computeGrad(X, y, theta, beta):
    b1, w1, b2, w2 = theta
    p, q = regress(X, theta)
    z1 = np.dot(X, w1) + b1
    db2 = np.sum(p - y, axis=0)
    dw2 = np.dot(q.T, p - y) + (beta * w2)
    dh1 = np.dot(p - y, w2.T)
    dz1 = sigmoidDerivative(z1)
    db1 = np.sum(dh1 * dz1, axis=0)
    dw1 = np.dot(X.T, dz1 * dh1) + (beta * w1)
    nabla = (db1, dw1, db2, dw2)  # nabla represents the full gradient
    return nabla


def create_mini_batch(X, Y, batchSize):
    xx = X.shape[1]
    yy = Y.shape[1]
    data = np.hstack((X, Y))
    np.random.shuffle(data)
    X = data[:, 0:xx]
    Y = data[:, xx:]

    batches = []
    batchcount = X.shape[0] // batchSize
    for i in range(batchcount):
        a = X[batchSize * i:batchSize * i + batchSize, :]
        b = Y[batchSize * i:batchSize * i + batchSize]
        batches.append((a, b))
    a = X[batchSize * (i + 1):, :]
    b = Y[batchSize * (i + 1):]
    batches.append((a, b))
    return batches
