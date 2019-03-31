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
### 3 linalg线性代数库
Numpy的linalg库可以满足大多数的线性代数运算。
1）矩阵的行列式
linalg.det(A) #A为n阶矩阵
2）矩阵的逆
linalg.inv(A)
3）矩阵的对称（对称矩阵的转置等于其本身）
4）矩阵的秩
矩阵中的最大的不相关的向量的个数，就叫秩。
linalg.matrix_rand(A) # 矩阵的秩
5）可逆矩阵求解线性方程组
 numpy.linalg中的函数solve可以求解形如 Ax = b 的线性方程组，其中 A 为矩阵，b 为一维或二维的数组，x 是未知变量
 linalg.solve(A,T(b))

 more inf : https://www.cnblogs.com/xieshengsen/p/6836430.html

### 4 各种距离的意义与python实现
距离公式列表和代码：（https://blog.csdn.net/thesnowboy_2/article/details/64921185）
1. 欧氏距离
> python 实现：
> sqrt((vector1-vector2)*(vector1-vector2).T)
2. 曼哈顿距离
sub(abs(vector1-vector2))
3. 切比雪夫距离
abs(vector1-vector2).max()
4. 闵可夫斯基距离
5. 标准化欧氏距离
6. 马氏距离
7. 夹角余弦
几何中的夹角余弦可用来衡量两个向量方向的差异。机器学习中借用这一概念来衡量样本向量之间的差异。夹角余弦越大，表示两个向量的夹角越小，反正依然成立。当两个向量的方向重合时候，夹角余弦取得最大值1，当两个方向完全相反的时候。夹角余弦取最小值-1，[-1,1]
cosV12 = dot(vector1,vector2)/(linalg.norm(vector1)*linalg.norm(vector2))
8. 汉明距离
汉明距离：当两个等长度字符串s1和s2之间的汉明距离定义为将其中一个变为另一个所需要的最小替换次数。例如”1111“ 与 ”1001“之间的汉明距离为2
应用：信息编码，为了增强容错性，应该使得编码间最小汉明距离尽可能大
matV = mat([[1,1,0,1,0],[0,1,1,1,0]])
smstr = nonzero(matV[0]-matV[1]);
print(shape(smstr[0])[1])
9. 杰卡德距离 & 杰卡德相似系数
两个集合A和B的交集元素在A，B的并集中所占的比例，称为两个集合的杰拉德相似系数。
杰卡德相似系数是衡量两个集合的相似度的一种指标。
