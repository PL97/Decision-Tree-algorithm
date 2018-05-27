from numpy import *
from math import log
from collections import defaultdict
import operator

def calShannonEnt(dataSet):
	dataSize = dataSet.shape[0]
	labelCount = defaultdict(lambda: 0)
	for temp in dataSet:
		label = temp[-1]
		labelCount[label] += 1;
	shannonEnt = 0.0
	for key in labelCount:
		probility = float(labelCount[key])/dataSize
		shannonEnt -= probility*log(probility, 2)
	return shannonEnt

#测试数据集
def createDataSet():
	dataSet = array([[1, 1, 'yes'], [1, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [0, 1, 'no']])
	labels = ['no surfacing', 'flippers']
	return dataSet, labels

#分裂数据集
def splitDataSet(dataSet, axis):
	labels = set(dataSet[:, axis])
	split = defaultdict(lambda:[])
	listDataSet = dataSet.tolist()
	for temp in listDataSet:
		newData = temp[0:axis]
		newData.extend(temp[axis+1:])
		split[temp[axis]].append(newData)
	return split

#选择最佳划分方式
def chooseBestAttr(dataSet):
	baseEnt = calShannonEnt(dataSet)
	dataSize = dataSet.shape[0]
	featureSize = len(dataSet[0])-1
	bestEnt = -1
	bestSub = -1
	for i in range(featureSize):
		split = splitDataSet(dataSet, i)
		features = set(dataSet[:, i])
		subEnt = 0.0
		for fea in features:
			probility = float(len(split[fea]))/dataSize
			subEnt += probility*calShannonEnt(array(split[fea]))
		if bestEnt < baseEnt - subEnt:
			bestEnt = subEnt
			bestSub = i
	return bestSub

def majorityLabel(dataSet):
	labels = dataSet[:,-1]
	print(labels)
	labelCount = defaultdict(lambda:0)
	for temp in labels:
		labelCount[temp] += 1;
	sortedLabelCount = sorted(labelCount.items(), key = operator.itemgetter(1), reverse=True)
	return sortedLabelCount[0][0]

def createTree(dataSet, labels):
	leftLabels = set(dataSet[:, -1])
	#如果所有待分类的数据标签一致时，直接返回标签
	if len(leftLabels) == 1: 
		return dataSet[0, -1]
	#当所有的属性都已经被使用
	if len(labels) == 0:
		return majorityLabel(dataSet)
	#需要进行计算获取最佳分类策略
	bestSub = chooseBestAttr(dataSet)
	split = splitDataSet(dataSet, bestSub)
	DTree = {labels[bestSub]:{}}
	newlabel = labels[0:bestSub]
	newlabel.extend(labels[bestSub+1:])
	for temp in split:
		DTree[labels[bestSub]][temp] = createTree(array(split[temp]), newlabel)
	return DTree


#主函数
dataSet, labels = createDataSet()
DTree = createTree(dataSet, labels)
print(DTree)