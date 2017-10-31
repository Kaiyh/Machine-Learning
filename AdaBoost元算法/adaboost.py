#!usr/bin/python
#-*- coding: UTF-8 -*-

from numpy import *

def loadSimpData():
    datMat = matrix([[1. , 2.1],
                     [2. , 1.1],
                     [1.3, 1. ],
                     [1. , 1. ],
                     [2. , 1. ]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# I 构造单层决策树

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """根据给定的特征列dimen和符号threshIneq，通过阈值thresVal比较对数据进行分类"""
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr, classLabels, D):
    """构造单层决策树"""
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0                          # 特征的所有可能取值
    bestStump = {}                           # 存储给定权重向量D所得到的最佳单层决策树
    bestClasEst = mat(zeros((m, 1)))         # 最佳单层决策树下对每个数据点的预测类别
    minError = inf
    # 嵌套的3个for循环
    for i in range(n):                                    # 1: 遍历数据集中的每个特征
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps)+1):              # 2: 遍历每个步长
            for inequal in ['lt', 'gt']:                  # 3: 遍历两个符号：小于/大于
                threshVal = (rangeMin + float(j) * stepSize)                       # 计算进行分类的阈值
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)   # 进行分类，得到预测结果
                # 计算错误率
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                # print "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" % (i, threshVal, inequal, weightedError)
                # 更新
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst     # 返回单层决策树、对应的错误率、对应的每个数据点的预测类别


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# II 训练数据集

def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    """根据数据集训练模型"""
    weakClassArr = []                        # 多个弱分类器组成的数组（其中每个元素为字典类型）
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)                # 每个数据点的权重
    aggClassEst = mat(zeros((m, 1)))         # 列向量，记录每个数据点的预测类别
    for i in range(numIt):
        # 创建单层决策树
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print "D:", D.T
        # 根据错误率error计算当前分类器的权重alpha
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print "classEst: ", classEst.T
        # 根据alpha值更新每个数据点的权重向量D
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D / D.sum()
        # 计算错误率，累加
        aggClassEst += alpha * classEst
        print "aggClassEst: ", aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum() / m
        print "total error: ", errorRate, "\n"
        if errorRate == 0.0: break
    return weakClassArr


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# III 分类

def adaClassify(datToClass, classifierArr):
    """分类函数，对每个若分类器的结果加权求和作为分类结果"""
    dataMatrix = mat(datToClass)             # 测试数据矩阵
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))         # 每个数据点的预测类别 
    for i in range(len(classifierArr)):      # 对每个弱分类器
        # 调用StumpClassify函数，得到每个数据点在该分类器作用下的预测类别
        classEst = stumpClassify(dataMatrix,\
                    classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print aggClassEst
    return sign(aggClassEst)                 # 返回符号值（-1 or 1）作为分类结果


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# IV 利用AdaBoost预测患有疝病马的存活率

def loadDataSet(fileName):
    """读取患有疝病马的数据集"""
    numFeat = len(open(fileName).readline().split('\t'))      # 读取一行，得到总列数
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():                 # 对每条数据
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)                 # 得到数据集（矩阵）
        labelMat.append(float(curLine[-1]))     # 得到类别标签（向量）
    return dataMat, labelMat


def testAdaBoost(trainFile, testFile):
    """预测病马死亡率（训练+分类）"""
    # 训练数据集
    dataArr, labelArr = loadDataSet(trainFile)
    classifierArr = adaBoostTrainDS(dataArr, labelArr, 10)
    # 测试数据，进行分类
    testArr, testLabelArr = loadDataSet(testFile)
    prediction = adaClassify(testArr, classifierArr)
    # 比对、计算错误率
    errArr = mat(ones((67, 1)))
    errTol = errArr[prediction != mat(testLabelArr).T].sum()
    print "the total error is %d, the error rate is %f" % (errTol, errTol / 67.0)