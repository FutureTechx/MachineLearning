# MachineLearning
Python Machine learning 

## 第一章 机器学习的基础
### 1 理解数学公式与Numpy矩阵计算
#### 1.1 创建全1矩阵和全0矩阵
``` python
import numpy as np
# crate zero and one 矩阵
myZero = np.zeros([3,5])
print(myZero)
myOnes = np.ones([3,5])
print(myOnes)
```
#### 1.2 创建随机矩阵
``` python
myRand=np.random.rand(3,4)
print(myRand)
```
#### 1.3 创建单位矩阵
``` python
myEye = np.eye(3) # 3*3单位矩阵
print(myEye)
```
** 输出结果：
``` cmd
jianchaos-MacBook-Pro:01chapter t_maj$ python MachineLearning_Numpy.py
[[0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0.]]
[[1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1.]
 [1. 1. 1. 1. 1.]]
[[0.10870504 0.22385547 0.15411805 0.59378845]
 [0.37870102 0.69480163 0.22523871 0.00347421]
 [0.70544795 0.42247914 0.51460716 0.3109724 ]]
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
 ```
#### 2 矩阵的元素运算
#### 2.1 矩阵的加减 和 数乘 和数学中的一样，两个矩阵直接加减，和一个数相乘

``` python
myOnes = ones([3,3]
myEye = eye(3)
print(myOnes+myEye,myOnes-myEye)

mymatrix = mat([[1,2,3],[4,5,6]],[7,8,9])
a = 10
print(mymatrix*a)
```
#### 2.2 矩阵所有元素求和
``` python
mymatrix = mat([[1,2,3],[4,5,6]],[7,8,9])
print (sum(mymatrix)) # 45
```
#### 2.3 矩阵各元素的积
矩阵的点乘同相应的元素的相乘，但矩阵的维度不同的时候，会根据一定的广播规则将维数扩充到一致的形式。
multiply(mymatrix1,mymatrix2)
#### 2.4 矩阵各元素的n次幂，n=2
power(mymatrix,2)

#### 3 矩阵的乘法：矩阵相乘,转置 
mymatrix*mymatrix2
mymatrix.T 或者mymatrix.transpose()
#### 4 矩阵的其他操作：行列数，切片，复制，比较
``` python
mymatrix = mat([[1,2,3],[4,5,6]],[7,8,9])
[m,n]=shape(mymatrix) #维数
myscl1 = mymatrix[0] # 行切片
myscl2 = mymatrix.T[0] # 列切片
mycpmat = mymatrix.copy() #矩阵的复制
print(mymatrix < mymatrix.T) #矩阵的比较，返回一个bool形的矩阵
```

