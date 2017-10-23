#!/usr/bin/python
#-*- coding: UTF-8 -*-

import trees
import treePlotter as tp

def execute():
    """use Decision Tree to address problem 'which lense?' """
    fr = open('lenses.txt')# read data set
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]# data set
    lensesLabels = ['age', 'prescirpt', 'astigmatic', 'tearRate']# labels
    lensesTree = trees.createTree(lenses, lensesLabels)# build the Decision Tree
    print lensesTree
    tp.createPlot(lensesTree)
