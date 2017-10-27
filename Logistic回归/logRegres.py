#!usr/bin/python
#-*- coding: UTF-8 -*-

from numpy import *

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# I: Logistic regression & Sigmoid function

def loadDataSet():
    """"create the dataset & labellist"""
    dataMat = []; labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])# the two features
        labelMat.append(int(lineArr[2]))# the class label
    return dataMat, labelMat


def sigmoid(inX):
    """Sigmoid function"""
    return 1.0 / (1 + exp(-inX))


def gradAscent(dataMatIn, classLabels):
    """use gradient ascent(梯度上升) to calculate the Regression Coefficients(回归系数)"""
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    # use the grad to calculate the best coefficients
    m, n = shape(dataMatrix)
    alpha = 0.001# the length of every step
    weights = ones((n, 1))# refers to the regression coefficients
    maxCycles = 500
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)# multiply each line with 'weights'
        error = (labelMat -h)# the 'error' vector
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# II: Plot the dots & line

def plotBestFit(weights):
    """plot the best line to split these dots"""
    import matplotlib.pyplot as plt
    dataMat, labelMat = loadDataSet()
    dataArr = array(dataMat)
    # get location of dots
    n = shape(dataArr)[0]# total of dataset
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    # plot all the dots & line
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# III: Improve the core algorithm

def stocGradAscent0(dataMatrix, classLabels):
    """stochastic(随机) gradient ascent algorithm"""
    m, n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights


def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    """(improved) -> stochastic gradient ascent"""
    m, n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Example: forecast the death rate of the sick horses

def classifyVector(inX, weights):
    """the feature vector & weights -> multiply -> sum -> input sigmoid()"""
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5: return 1.0
    else: return 0.0


def colicTest():
    """Use the Logistic to analyse colic horse"""
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    # Training
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():# for each trianing data
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):# there are 20 features in each data
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))# the last row is class label
    trainWeights = stocGradAscent1(array(trainingSet), trainingLabels, 500)
    # Testing
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():# for each testing data
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr), trainWeights) != int(currLine[21])):
            errorCount += 1
    errorRate = (float(errorCount) / numTestVec)
    print "the error rate of this teset is: %f" % errorRate
    return errorRate


def multiTest():
    """test for 10 times to calculate the average rate"""
    numTests = 10; errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" % (numTests, errorSum / float(numTests))