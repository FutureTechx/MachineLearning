## 一 创建工作区
#### 1 查看是否装了pip:
pip3 --version
```
futuretechx@ubuntu:~/Documents$ pip3 --version

Command 'pip3' not found, but can be installed with:

sudo apt install python3-pip

futuretechx@ubuntu:~/Documents$ sudo apt install python3-pip
Reading package lists... Done
Building dependency tree 
```
#### 2 没有安装使用apt安装
sudo apt install python3-pip   
#### 3 创建一个虚拟环境
1. 安装virtualenv
```
futuretechx@ubuntu:~/Documents$ pip3 install --user --upgrade virtualenv
Collecting virtualenv
  Downloading https://files.pythonhosted.org/packages/4f/ba/6f9315180501d5ac3e707f19fcb1764c26cc6a9a31af05778f7c2383eadb/virtualenv-16.5.0-py2.py3-none-any.whl (2.0MB)
    100% |████████████████████████████████| 2.0MB 525kB/s 
Installing collected packages: virtualenv
Successfully installed virtualenv-16.5.0
```
2. 创建env python环境
- cd $ML_PATH
- virtualenv env (这种对我不work)

```
futuretechx@ubuntu:~/ml$ virtualenv env

Command 'virtualenv' not found, but can be installed with:

sudo apt install virtualenv

futuretechx@ubuntu:~/ml$ sudo apt install virtualenv
[sudo] password for futuretechx: 
Reading package lists... Done

```
- sudo apt install virtualenv
- virtualenv env
```
futuretechx@ubuntu:~/ml$ virtualenv env
The path python2 (from --python=python2) does not exist
```
- virtualenv env --python=python3
- source env/bin/activate

#### 4 安装模块
- pip3 install --upgrade jupyter matplotlib numpy pandas scipy scikit-learn 
#### 5 检验安装
```python
python3 -c "import jupyter,matplotlib,numpy,pandas,scipy,sklearn"
```
#### 6启动Jupyter
jupyter notebook

## 二 下载数据
下载数据的函数：
```python
import os
import tarfile
from six.moves import urllib
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_RUL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_RUL,housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path=os.path.join(housing_path,"housing.tgz")
    urllib.request.urlretrieve(housing_url,tgz_path)
    housing_tgz=tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
fetch_housing_data()
```
执行完之后在/ml/datasets/housing下面会有两个文件：
![a23c74aed7d38122620ec4dd57b7af16.png](en-resource://database/1418:1)

- 1 urllib模块提供的urlretrieve()函数。urlretrieve()方法直接将远程数据下载到本地。
urlretrieve(url, filename=None, reporthook=None, data=None)

参数filename指定了保存本地路径（如果参数未指定，urllib会生成一个临时文件保存数据。）
参数reporthook是一个回调函数，当连接上服务器、以及相应的数据块传输完毕时会触发该回调，我们可以利用这个回调函数来显示当前的下载进度。
参数data指post导服务器的数据，该方法返回一个包含两个元素的(filename, headers) 元组，filename 表示保存到本地的路径，header表示服务器的响应头
如果想显示进度，可以实现类似如下的一个回调函数：
```python
def cbk(a,b,c):  
    '''''回调函数 
    @a:已经下载的数据块 
    @b:数据块的大小 
    @c:远程文件的大小 
    '''  
    per=100.0*a*b/c  
    if per>100:  
        per=100  
    print '%.2f%%' % per  
```
- 2 tarfile
既然有压缩模块zipfile，那有一个归档模块tarfile也是很自然的。tarfile模块用于解包和打包文件，包括被gzip，bz2或lzma压缩后的打包文件。如果是.zip类型的文件，建议使用zipfile模块，更高级的功能请使用shutil模块。
tarfile.open(name=None, mode='r', fileobj=None, bufsize=10240, \*\*kwargs)
返回一个TarFile类型的对象。本质上就是打开一个文件对象。
name是文件名或路径。
bufsize用于指定数据块的大小，默认为20\*512字节。
mode是打开模式，一个类似filemode[:compression]格式的字符串，可以有下表所示的组合，默认为“r”。
- 3 解包指定文件
TarFile.extractfile(member)
## 三 pandas加载数据
```python
import pandas as pd
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path=os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)

housing = load_housing_data()
housing.head()
```
##### 1 housing.head() 显示数据前五行
![743f83e28d1e912989eaa65f2c844985.png](en-resource://database/1434:1)
##### 2 通过info方法可以快速获取数据集的简单描述，特别是总行数，每个属性的类型和非空值的数量。
![bc5ea33faa85104e21aae4424935d9e6.png](en-resource://database/1436:1)
##### 3 value_counts()方法可以查看有多少种类存在
![2f0ff318b05d2cb7fdabd9b3bdc4f7c7.png](en-resource://database/1438:1)
##### 4 describe()方法可以显示数值属性的摘要
![75bc50a450659d9289816b63b5ec072f.png](en-resource://database/1440:1)
##### 5 绘制每个数值属性的直方图
```python
%matplotlib inline
import matplotlib.pyplot as plt
housing.hist(bins=50,figsize=(20,15))
plt.show()
```
![a8dcddad358a38fc3bfb8c0692adee1f.png](en-resource://database/1442:1)

```python
import numpy as np
def split_train_test(data,test_ratio):
    shuffled_indices=np.random.permutation(len(data))
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]

train_set,test_set=split_train_test(housing,0.2)
print(str(len(train_set)) + " train " + str(len(test_set)) + " test")
```
![94a59f00a534c51f17b0be846f6e2dad.png](en-resource://database/1444:1)

Scikit-Learn 提供了一些函数，可以通过多种方式将数据集分割成多个子集，最简单的函数是train_test_split.
```python
from sklearn.model_selection import train_test_split
train_set,test_set = train_test_split(housing,test_size=0.2,random_state=42)
```
分层抽样：将人口划分为均匀的子集，每个子集被称为一层，然后从没层抽取正确的实例数量，以确保测试集合代表总的人口比例。
将中位数除以1.5一限制收入类别的数量，然后使用ceil进行取整。最后将所有大于5的类别合并为类别5
```python
housing["income_cat"]=np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"]<5,5.0,inplace=True)

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing["income_cat"]):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]

housing["income_cat"].value_counts()/len(housing)
```
```cmd
3.0    0.350581
2.0    0.318847
4.0    0.176308
5.0    0.114438
1.0    0.039826
Name: income_cat, dtype: float64
```
Notes:
1) np.ceil(ndarray)
计算大于等于改值的最小整数

>>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
>>> np.ceil(a)
array([-1., -1., -0.,  1.,  2.,  2.,  2.])

2） np.floor(a) : 计算各元素的 floor值（ceiling向上取整，floor向下取整） 
3）numpy.where() 有两种用法：
1. np.where(condition, x, y)
满足条件(condition)，输出x，不满足输出y。
2. np.where(condition)
只有条件 (condition)，没有x和y，则输出满足条件 (即非0) 元素的坐标 (等价于numpy.nonzero)。这里的坐标以tuple的形式给出，通常原数组有多少维，输出的tuple中就包含几个数组，分别对应符合条件元素的各维坐标。
