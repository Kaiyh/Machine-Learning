#!/usr/bin/python
# -*- coding: UTF-8 -*-

from numpy import *
from math import log
import operator


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def createDataSet():
	"""create the dataSet"""
	dataSet = [[1,1,'yes'], [1,1,'yes'], [1,0,'no'], [0,1,'no'], [0,1,'no']]
	labels = ['no surfacing', 'flippers']
	return dataSet, labels


def calcShannonEnt(dataSet):
	"""calculate the Entropy of dataSet"""
	numEntries = len(dataSet)
	labelCounts = {}# the dictionary of labels
	for featVec in dataSet:
		currentLabel = featVec[-1]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
	shannonEnt = 0.0# the Entropy
	for key in labelCounts:
		prob = float(labelCounts[key]) / numEntries
		shannonEnt -= prob * log(prob, 2)
	return shannonEnt


def splitDataSet(dataSet, axis, value):
	"""split the dataSet"""
	retDataSet = []
	for featVec in dataSet:
		if featVec[axis] == value:
			reducedFeatVec = featVec[:axis]# the chip
			reducedFeatVec.extend(featVec[axis+1:])# extend the left
			retDataSet.append(reducedFeatVec)
	return retDataSet


def chooseBestFeatureToSplit(dataSet):
	"""choose the best feature"""
	numFeatures = len(dataSet[0]) - 1# the number of the features
	baseEntropy = calcShannonEnt(dataSet)# get the Entropy of dataSet
	bestInfoGain = 0.0# the information gain
	bestFeature = -1
	# enumerate all the features
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]# get the value list of this feature
		uniqueVals = set(featList)
		# calculate the new Entropy by spliting the dataSet
		newEntropy = 0.0
		for value in uniqueVals:# enumerate all the values of the value list(set)
			subDataSet  = splitDataSet(dataSet, i, value)
			prob = len(subDataSet) / float(len(dataSet))
			newEntropy += prob * calcShannonEnt(subDataSet)
		# the increament of Entropy
		infoGain = baseEntropy - newEntropy
		if (infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature
	

def majorityCnt(classList):
	"""find the majority"""
	# count the number of every class
	classCount = {}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote] = 0
		classCount[vote] += 1
	# sort the 'classCount'
	sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]


def createTree(dataSet, labels):
	"""(core algorithm) -> create the Tree"""
	classList = [example[-1] for example in dataSet]
	if classList.count(classList[0] == len(classList)):
		return classList[0]
	if len(dataSet[0]) == 1:
		return majorityCnt(classList)
	# recursively build the tree
	bestFeat = chooseBestFeatureToSplit(dataSet)
	# print bestFeat
	bestFeatLabel = labels[bestFeat]
	del(labels[bestFeat])
	myTree = {bestFeatLabel:{}}
	# get all the values releated to 'bestFeat' as a list
	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)
	for value in uniqueVals:
		subLabels = labels[:]
		myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
	return myTree


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
