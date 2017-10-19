## 使用NumPy

import numpy as np

## 数组

1、使用array()方法

array((1.2,2,3,4))

2、使用arange()方法

```python
>>> arr = np.arange(10)
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

>>> arr2 = np.arange(2, 5)
array([2, 3, 4])
```

3、使用numpy.linspace方法

```python
np.linspace(1,10,20) #在从1到10中产生20个数
```

4、使用numpy.zeros，numpy.ones，numpy.eye等方法可以构造特定的矩阵

```python
>>> print np.zeros((3,4))
[[ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]
 [ 0.  0.  0.  0.]]

>>> print np.ones((3,4))
[[ 1.  1.  1.  1.]
 [ 1.  1.  1.  1.]
 [ 1.  1.  1.  1.]]

>>> print np.eye(3)
[[ 1.  0.  0.]
 [ 0.  1.  0.]
 [ 0.  0.  1.]]
```

5、以list或tuple变量为参数产生一维数组

```python
np.array([1,2,3,4])
[1 2 3 4]
>>> print np.array((1.2,2,3,4))
[1.2 2 3 4]
>>> print type(np.array((1.2,2,3,4)))  
<type 'numpy.ndarray'>
```

6、以list或tuple变量为元素产生二维数组或者多维数组

```python
x = np.array(((1,2,3),(4,5,6)))  
>>> x  
array([[1, 2, 3],  
       [4, 5, 6]])  
>>> y = np.array([[1,2,3],[4,5,6]])  
>>> y  
array([[1, 2, 3],  
       [4, 5, 6]])
```

## 数组的常用属性

```python
a = np.zeros((2, 2, 2))  
>>> print a.ndim   #数组的维数
3

>>> print a.shape  #数组每一维的大小
(2, 2, 2)

>>> print a.size   #数组的元素数
8

>>> print a.dtype  #元素类型
float64

>>> print a.itemsize  #每个元素所占的字节数
8
```

## 数组常用方法

sum()方法

```python
>>> x
array([[[ 0,  1,  2],
        [ 3,  4,  5],
        [ 6,  7,  8]],
       [[ 9, 10, 11],
        [12, 13, 14],
        [15, 16, 17]],
       [[18, 19, 20],
        [21, 22, 23],
        [24, 25, 26]]])

>>> x.sum(axis=0) # sum方法
array([[27, 30, 33],
       [36, 39, 42],
       [45, 48, 51]])

# for sum, axis is the first keyword, so we may omit it, specifying only its value
>>> x.sum(0), x.sum(1), x.sum(2)
(array([[27, 30, 33],
        [36, 39, 42],
        [45, 48, 51]]),
 array([[ 9, 12, 15],
        [36, 39, 42],
        [63, 66, 69]]),
 array([[ 3, 12, 21],
        [30, 39, 48],
        [57, 66, 75]]))

>>> np.sum([[0, 1], [0, 5]])  
6  
>>> np.sum([[0, 1], [0, 5]], axis=0)
array([0, 6])
>>> np.sum([[0, 1], [0, 5]], axis=1)
array([1, 5])
``` 

## 一维数组 & 索引与切片

index 和slicing: 第一数值代表数组横坐标，第二个为纵坐标

```
>>> x[1, 2]
6
>>> y=x[:, 1]
>>> y
array([2, 5])
```

当改变y时，x也跟着改变，如下所示：(_y和x指向是同一块内存空间值_)

```
>>> y[0] = 10
>>> y
array([10,  5])
>>> x
array([[ 1, 10,  3],
       [ 4,  5,  6]])
```

当将一个标量赋值给切片时，该值会自动传播整个切片区域，这个跟列表最重要本质区别，数组切片是原始数组的视图，视图上任何修改直接反映到源数据上面。Numpy 设计是为了处理大数据，如果切片采用数据复制话会产生极大的性能和内存消耗问题。

```
>>> arr = np.arange(10)
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

>>> arr[3:6]  
array([3, 4, 5])  

>>> arr[3:6] = 12
array([ 0,  1,  2, 12, 12, 12,  6,  7,  8,  9])
```

得到数组(或数组切片)的副本

arr[3:6].copy()

## 多维数组 & 索引、切片

多维数组

```python
arr2d = np.arange(1, 10).reshape(3, 3)  

>>> arr2d
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])

>>> arr2d[2]
array([7, 8, 9])

>>> arr2d[0][2]
3

>>> arr2d[0,2]
3
```

布尔型索引

```python
names = np.array(['Bob','joe','Bob','will'])  
>>> names == 'Bob'
array([ True, False,  True, False], dtype=bool)

data = np.array([2, -1, 3, 5, -5])
>>> data[data < 0] = 0
array([2, 0, 3, 5, 0])  
```

## 数组文件读取

在numpy中已经有成熟函数封装好了可以使用

将数组以二进制形式格式保存到磁盘，np.save 、np.load 函数是读写磁盘的两个主要函数

```python
arr = np.arange(10)
np.save('some_array', arr)

>>> np.load('some_array.npy')
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
```

存取文本文件：

文本中存放是聚类需要数据，直接可以方便读取到numpy array中，省去一行行读文件繁琐。np.savetxt 执行相反的操作

```python
arr = np.loadtxt('dataMatrix.txt', delimiter=' ')
arr
...
```

## numpy数据类型设定与转换

numpy ndarray数据类型可以通过参数dtype 设定，而且可以使用astype转换类型，在处理文件时候这个会很实用，注意astype 调用会返回一个新的数组，也就是原始数据的一份拷贝。

    numeric_strings2 = np.array(['1.23','2.34','3.45'],dtype=np.string_)  
      
    numeric_strings2  
    Out[32]:   
    array(['1.23', '2.34', '3.45'],   
          dtype='|S4')  
      
    numeric_strings2.astype(float)  
    Out[33]: array([ 1.23,  2.34,  3.45])  

