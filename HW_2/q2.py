import os

import numpy as np
import pandas as pd
from itertools import chain, combinations


class NaiveBayes():
    """
    Naive Bayes Classifier
    """
    def __init__(self, alpha=1, flag=1):
        self.alpha = alpha
        self.rows = 0
        self.mean = []
        self.var = []
        self.count = None
        self.classes = None
        self.classesCount = None
        self.categoricalFeatures = []
        self.numericalFeatures = []
        self.prior = None

    def gaussianDensity(self, class_idx, col1, x):
        '''
        calculates probability using Gaussian density function
        assumption: probability of specific target value is normally distributed
        '''
        mean = self.mean.at[class_idx, col1]
        var = self.var.at[class_idx, col1]
        num = np.exp((-1 / 2) * ((x - mean) ** 2) / (2 * var))
        denom = np.sqrt(2 * np.pi * var)
        return num / denom

    def calculateNumericalMeanAndVariance(self, features, target):
        self.mean = features.groupby(target).mean()
        self.var = features.groupby(target).var()
        self.numericalFeatures = self.numericalFeatures + list(features.columns)
        return self.mean, self.var

    def calculateCategoricalCount(self, features, target):
        col = features.columns
        features = pd.get_dummies(features[col])
        self.count = features.groupby(target).sum()
        self.categoricalFeatures = self.categoricalFeatures + list(features.columns)
        return self.count

    def calculatePrior(self, features, target):
        self.classesCount = features.groupby([target]).count().iloc[:, 0]
        self.prior = self.classesCount / features.shape[0]

    def train(self, data, target):
        self.classes = np.unique(data[target])
        self.calculatePrior(data, target)
        categoricalFeatures = data.select_dtypes(include=[object])
        numericalFeatures = data.select_dtypes(exclude=[object])
        categoricalFeatures[target] = data[target]
        numericalFeatures[target] = data[target]

        self.rows = data.shape[0]
        if len(numericalFeatures) > 0:
            self.calculateNumericalMeanAndVariance(numericalFeatures, target)
        if len(categoricalFeatures) > 0:
            self.calculateCategoricalCount(categoricalFeatures, target)

    def test(self, data):
        probability = []
        for i, k in pd.get_dummies(data).iterrows():
            elemProbability = {}
            for j in k.index:
                if j in self.categoricalFeatures:
                    if k[j] == 1:
                        for c in self.classes:
                            # prior = self.prior[c]
                            classCount = self.classesCount[c]
                            # print(j)
                            likelihoodCount = self.count.at[c, j]
                            likelihood = (likelihoodCount / classCount)
                            elemProbability[c] = likelihood * elemProbability.get(c, 1)
                elif j in self.numericalFeatures:
                    for c in self.classes:
                        likelihood = self.gaussianDensity(c, j, k[j])
                        elemProbability[c] = elemProbability.get(c, 1) * likelihood
            for c in self.classes:
                prior = self.prior[c]
                elemProbability[c] = elemProbability.get(c, 1) * prior
            probability.append(elemProbability)
        return probability


def trainOnFullData(data):
    """
        Trains the model on whole dataset
        :param data: dataframe

        :return: Naive-Bayes classifier
    """
    columns = ['in html', ' has emoji', ' sent to list', ' from .com', ' has my name',
               ' has sig', ' # sentences', ' # words']
    nb = NaiveBayes()
    columnsInData = data.select_dtypes(include=[bool]).columns

    for c in columnsInData:
        if c != ' is spam':
            data[c] = data[c].map({True: 'True', False: 'False'})

    nb.train(data[columns + [' is spam']], ' is spam')
    x = nb.test(data[columns])
    pred = []
    for i in x:
        pred.append(max(i, key=i.get))
    y2 = np.array(pred)
    y = data[' is spam'].values
    accuracy = (y2 == y).sum() / y2.shape[0]
    print("Accuracy of training data: ", accuracy)
    return nb


def classify(data, nb, columns):
    if not columns:
        columns = ['in html', ' has emoji', ' sent to list', ' from .com', ' has my name',
                   ' has sig', ' # sentences', ' # words']
    col = data.select_dtypes(include=[bool]).columns
    for i in col:
        if i != ' is spam':
            data[i] = data[i].map({True: 'True', False: 'False'})

    x = nb.test(data[columns])
    pred = []
    for i in x:
        pred.append(max(i, key=i.get))
    y2 = np.array(pred)
    y = data[' is spam'].values
    accuracy = (y2 == y).sum() / y2.shape[0]
    return y2, accuracy


def trainOnSubset(data, minFeatures):
    """
    finds optimal features by creating NB models and compare accuracy
    :param data: dataframe
    :param minFeatures:  minimum number of feature in the model

    :return: best accuracy, best feature list and best model
    """
    columns = ['in html', ' has emoji', ' sent to list', ' from .com', ' has my name',
              ' has sig', ' # sentences', ' # words']
    x = chain.from_iterable(combinations(columns, r) for r in range(minFeatures, len(columns) + 1))
    maxAccuracy = 0
    bestColumns = []
    bestClassifier = None
    for i in x:
        columns = list(i)
        nb = NaiveBayes()
        col = data.select_dtypes(include=[bool]).columns
        for i in col:
            if i != ' is spam':
                data[i] = data[i].map({True: 'True', False: 'False'})

        nb.train(data[columns + [' is spam']], ' is spam')
        testData = pd.read_csv(os.getcwd() + '/data/q3b.csv')
        pred, accuracy = classify(testData, nb, columns)
        if accuracy > maxAccuracy:
            maxAccuracy = accuracy
            bestClassifier = nb
            bestColumns = columns
    return maxAccuracy, bestColumns, bestClassifier


def saveParameters(n):

    output = os.getcwd() + '/out/'
    f = n.count
    x = n.classesCount
    likelihood = f.divide(x, axis='index')
    mean = n.mean
    var = n.var
    likelihood.to_csv(output + "q2_categorical_maximum_Likelihood.csv")
    mean.to_csv(output + "q2_numerical_mean.csv")
    var.to_csv(output + "q2_numerical_variance.csv")


print("FULL DATA")
data = pd.read_csv(os.getcwd() + '/data/q3.csv')
nb = trainOnFullData(data)
print("_____________________________________________________")
print("TEST DATA")
data = pd.read_csv(os.getcwd() + '/data/q3b.csv')
predictions, accuracy = classify(data, nb, None)
print("Accuracy (all columns used): ", accuracy)
print("Error: ", 1 - accuracy)
saveParameters(nb)

print("_____________________________________________________")
print("SUBSET DATA")
data = pd.read_csv(os.getcwd() + '/data/q3.csv')
a, b, c = trainOnSubset(data, 6)
print("Best Subset: ", b)
print("Best Accuracy: ", a)
print("Error: ", 1 - a)
