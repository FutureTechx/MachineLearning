import numpy as np
# 导入运算模块
import operator
import matplotlib
import matplotlib.pyplot as plt


def createDataSet():
    group = np.array([[1, 1.1], [1, 1], [0, 0], [0, 0.1]])
    labels = ["A", "A", "B", "B"]
    return group, labels


def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndistIncicies = distances.argsort()
    classCount = {}

    for i in range(k):
        voteIlabel = labels[sortedDistIndistIncicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) +1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


group, lables = createDataSet()

predict = classify([0, 0], group, lables, 3)
print(predict)


def file2matrix(filename):
    fr = open(filename)
    arrayOlines = fr.readlines()
    numberOfLines = len(arrayOlines)
    returnMatt = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOlines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMatt[index, :] = listFromLine[0:3]
        classLabelVector.append((int(listFromLine[-1])))
        index += 1
    return returnMatt, classLabelVector


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))

    return normDataSet, ranges, minVals


datingDataMat, datingLabels = file2matrix("datingTestSet2.txt")
print("before Normalize ............")
print(datingDataMat)
print(datingLabels)

print("after Normalize ............")
normMat, ranges, minVals = autoNorm(datingDataMat)
print(normMat,ranges,minVals)
plt.scatter(datingDataMat[:, 1], datingDataMat[:, 2], 15.0 * np.array(datingLabels), 15.0 * np.array(datingLabels))
plt.xlabel("play game time %")
plt.ylabel("eat ice cream rate by pre week %")
plt.show()

def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix("datingTestSet2.txt")
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs = int (m*hoRatio)
    errorCount = 0.0

    for i in range(numTestVecs):
        classifierResult = classify(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifier came back with :%d,the real answer is : %d" % (classifierResult,datingLabels[i]))
        if(classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("the total error rate is %f" % (errorCount/float(numTestVecs)))

datingClassTest()

'''
tile(a,x):   x是控制a重复几次的，结果是一个一维数组
tile(a,(x,y))：   结果是一个二维矩阵，其中行数为x，列数是一维数组a的长度和y的乘积
tile(a,(x,y,z)):   结果是一个三维矩阵，其中矩阵的行数为x，矩阵的列数为y，而z表示矩阵每个单元格里a重复的次数。(三维矩阵可以看成一个二维矩阵，每个矩阵的单元格里存者一个一维矩阵a)

argsort 返回排序后的下标，是一个数组，比如：

x = numpy.array([1.48,1.41,0.0,0.1])
print x.argsort()

>[2 3 1 0]
2 is the index of 0.0.
3 is the index of 0.1.
1 is the index of 1.41.
0 is the index of 1.48.

dict.get(key[, value]) 
get() Parameters
The get() method takes maximum of two parameters:
key - key to be searched in the dictionary
value (optional) - Value to be returned if the key is not found. The default value is None.
'''
