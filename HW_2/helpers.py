import numpy as np


def softmax(z):
    denom = np.sum(np.exp(z), axis=1)
    return np.exp(z) / np.reshape(denom, (denom.shape[0], 1))


def predict(X, theta):
    p = regress(X, theta)
    return np.argmax(p, axis=1)


def regress(X, theta):
    b, w = theta
    return softmax(np.dot(X, w) + b)


def log_likelihood(p, y):
    size = y.shape[0]
    return -1 * np.sum(np.sum(np.multiply(y, np.log(p)))) / size


def computeCost(X, y, theta, beta):
    p = regress(X, theta)
    b, w = theta
    loss = log_likelihood(p, y)
    reg = (beta * np.sum(w * w)) / 2
    return loss + reg


def computeGrad(X, y, theta, beta):
    p = regress(X, theta)
    b, w = theta
    size = X.shape[0]
    dL_dfy = p - y  # derivative w.r.t. to model output units (fy)
    dL_db = np.sum((dL_dfy), axis=0) / size  # derivative w.r.t. model b
    dL_dw = np.dot(X.T, (dL_dfy / size)) + (beta * w)  # derivative w.r.t model w
    nabla = (dL_db, dL_dw)  # nabla represents the full gradient
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
