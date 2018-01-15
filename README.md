## SMO-SVM-Python
### Implementation of SMO-SVM based on Python code. author：yaliangzhang.

中文
---
这是用Python代码写的基于SVM算法，可用于二分类和多分类。

# 数据
irisdata.txt 包含所有的鸢尾花数据（三种类别的鸢尾花数据）。
我利用其中前两种鸢尾花的数据用于二分类测试，两种鸢尾花数据各取一半作为训练集，另一半为测试集。
训练集和测试集文件名分别为：iris_train_set_bi.txt 和iris_test_set_bi.txt
多分类测试用三种类别的鸢尾花数据，同样的，三种鸢尾花数据每种各取一半作为训练集，另一半为测试集。
训练集和测试集文件名分别为：iris_train_set_multi.txt 和iris_test_set_multi.txt

# 代码部分
这是John C. Platt用于训练支持向量机（SVM）的顺序最小优化（SMO）的Python实现。该程序基于Platt（1998）中的伪代码。
代码部分参考了John C  Platt 的文献：Fast Training of Support Vector Machines using Sequential Minimal Optimization(1998)。
以及一个基于C++版本写的SVM代码，这个C++版本代码也是基于上面这个文献写的
前面的大部分部分代码都是基于C++版本改写成Python版本的（有改动），后面的多分类代码是由自己写的
相应的原文文献可以从网上下载。

# 使用
这个包内有许多函数：
['MainRoutine',
 'SVM_parameter',
 '__builtins__',
 '__doc__',
 '__file__',
 '__name__',
 '__package__',
 'accuracy',
 'dot_product_func',
 'examineExample',
 'kernel_func',
 'learned_func',
 'math',
 'random',
 're',
 'support_vector',
 'takeStep',
 'test_SVM',
 'train_SVM']
 
但这个包的使用只需要掌握其中的两到三个函数就可以了：
train_SVM()、test_SVM、support_vector()
如果你对支持向量不感兴趣，那你只需要掌握前面两个函数。
我会在tutorial中写下如何利用我给的代码对我的相应的数据进行二分类和多分类。


English
---
This is an SVM-based algorithm written in Python code that can be used for binary-classification and multi-classification.

# data
irisdata.txt contains all the iris data (three categories of iris data).
I used the data from the first two Irises for the dichotomous test, half of the Iris data for each training set, and the other half as the test set.
Training and test set file names were: iris_train_set_bi.txt and iris_test_set_bi.txt
multi-classification test with three types of iris data, the same, three kinds of iris each take half each as a training set, the other half for the test set.
The training and test set file names are: iris_train_set_multi.txt and iris_test_set_multi.txt

# Code section
This is a Python implementation of John C. Platt's sequential minimal optimization(SMO) for train a support vector machine(SVM).This program is based the pseudocode in Platt(1998).
The code section refers to John C Platt's article: Fast Training of Support Vector Machines using Sequential Minimal Optimization (1998), and a SVM code based on the C ++ version, which is also based on the above document
Most of the previous part of the code is based on the C + + version rewritten into Python version (with changes), behind the multi-classification code is written by myself.
The corresponding original literature can be downloaded from the Internet.

# Use
There are many functions inside this package：
['MainRoutine',
 'SVM_parameter',
 '__builtins__',
 '__doc__',
 '__file__',
 '__name__',
 '__package__',
 'accuracy',
 'dot_product_func',
 'examineExample',
 'kernel_func',
 'learned_func',
 'math',
 'random',
 're',
 'support_vector',
 'takeStep',
 'test_SVM',
 'train_SVM']

But the use of this package only need to master one or two of the three functions on it:
train_SVM (), test_SVM, support_vector ()
If you are not interested in support vectors, then you only need to know the first two functions.
I will write in the tutorial how to use the code I give my corresponding data to do binary-classification and multi-classification.




