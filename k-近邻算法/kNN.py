import operator
from numpy import *
from os import listdir


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# kNN algorithm

def createDataSet():
	group = array([[1.0,1.1], [1.0,1.0], [0,0], [0,0.1]])
	labels = ['A', 'A', 'B', 'B']
	return group, labels


def classify0(inX, dataSet, labels, k):
	"""kNN algorithm"""
	dataSetSize = dataSet.shape[0]# the first dimension of 'dataSet'
	diffMat = tile(inX, (dataSetSize, 1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)# sum the second dimension of the matrix
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort()# sort and get the increasing index
	
	classCount = {}
	for i in range(k):# the first k vectors(the minimum distance)
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1# count the number of every label
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]# return the max label
	
	
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Example I: use kNN algoritm to analyse the dating website

def file2matrix(filename):
	"""transfer the file to dataSet"""
	fr = open(filename)
	arrayOLines = fr.readlines()# read all lines from the file
	numberOfLines = len(arrayOLines)# the number of lines(one line as a vector)
	returnMat = zeros((numberOfLines, 3))# array: n x 3
	index = 0# point to the first dimension of the 'returnMat'
	classLabelVector = []
	for line in arrayOLines:
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index, :] = listFromLine[0:3]# the three attributes
		index += 1
		classLabelVector.append(int(listFromLine[-1]))# the object label
	return returnMat, classLabelVector
	
	
def autoNorm(dataSet):
	"""normalization the attributes"""
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	
	normDataSet = zeros(shape(dataSet))
	m = dataSet.shape[0]# the first dimension of 'dataSet' as the number of points
	# substract the minVals and divide the ranges (the operation of normalization)
	normDataSet = dataSet - tile(minVals, (m,1))
	normDataSet = normDataSet / tile(ranges, (m,1))
	return normDataSet, ranges, minVals


def datingClassTest():
	"""testing algorithm of dating website"""
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')# read the dataSet
	normMat, ranges, minVals = autoNorm(datingDataMat)# normalization the dataSet
	m = normMat.shape[0]
	hoRatio = 0.10
	numTestVecs = int(m*hoRatio)# use the first 1/10 datas as the test datas
	errorCount = 0.0
	for i in range(numTestVecs):
		# call the core function: classify0(inX, dataSet, labels, k)
		classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], datingLabels[numTestVecs:m], 3)
		print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
		if (classifierResult != datingLabels[i]): errorCount += 1.0
	print "the total error rate is: %f" % (errorCount/float(numTestVecs))


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Example II: use kNN algorithm to recognize the handwriting

def img2vector(filename):
	"""convert the digital image to vector"""
	returnVect = zeros((1, 1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0, 32*i+j] = int(lineStr[j])
	return returnVect


def handwritingClassTest():
	"""testing algorithm of handwriting"""
	# Train the examples
	hwLabels = []
	trainingFileList = listdir('trainingDigits')
	m = len(trainingFileList)# look the number of files as the vectors' number
	trainingMat = zeros((m, 1024))# the array: m x 1024
	for i in range(m):
		# get the fileStr and classNumber
		fileNameStr = trainingFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		# the list 'hwLabels' store the classNumber as 'labels'
		hwLabels.append(classNumStr)
		# the list 'trainingMat' as the 'dataSet'
		trainingMat[i, :] = img2vector('trainingDigits/%s' % fileNameStr)
	
	# Test the dataSet
	testFileList = listdir('testDigits')
	mTest = len(testFileList)
	errorCount = 0.0
	for i in range(mTest):
		# read one file to get information
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])# the real number
		# the testing vector
		vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
		# call the core function: classify0(inX, dataSet, labels, k)
		classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
		print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
		if (classifierResult != classNumStr): errorCount += 1.0
	print "\nthe total number of errors is: %d" % errorCount
	print "\nthe total rate is: %f" % (errorCount/float(mTest))


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
