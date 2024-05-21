[toc]

# run notebook

``` bash
# 启用 python虚拟机
source .venv/bin/activate
# 安装项目依赖
pip3 install -r requirements.txt 
# 安装 jupyter
pip3 install jupyter
# run notebook, Go to the main page of Notebook
jupyter notebook

# 访问 http://localhost:8888/tree
```


# notebook 目录

```
|- regression 回归
| |- linear                               线性回归/单项式
| | |- univariate_LinearRegression.ipynb    一元
| | |- scikit-learn_univariate_LinearRegression.ipynb
| | |- multiple_LinearRegression            多元
| |- nonlinear                            非线性回归/多项式
| | |- 2degree_polynomial_fit_data.ipynb    2次多项式拟合
| | |- Ndegree_polynomial_fit_data.ipynb    n次多项式拟合
| | |- scikit-learn_polynomial_fit_data.ipynb
| | |- polynomial_regression.ipynb          多项式回归
|- classify   分类
| |- logistic_regression.ipynb            线性可分-逻辑回归
| |- scikit-learn_logistic_regression.ipynb.ipynb
| |- K-NearestNeighbor.ipynb              线性不可分-K近邻
| |- NaiveBayes.ipynb                     概率预测-朴素贝叶斯
| |- case-NaiveBayes_spamfiltering.ipynb  朴素贝叶斯-伯努利模型
| |- NaiveBayes_Gaussian.ipynb            朴素贝叶斯-高斯分布
Support Vector Machine， 支持向量机

```

## pandas
```
import pandas as pd
```

```
df = pd.read_csv(url, storage_options=storage_options, header=0)

```

## numpy
```
import numpy as np
```
- array
```
np.array()  # 生成 N维数组对象 ndarray
ndarray1 = np.array([1,2,3])
ndarray2 = np.array([[1,  2],  [3,  4]]) 

ndarray.shape  # 输出数组形状
ndarray1.shape # (3,)
ndarray2.shape # (2,2)

ndarray.reshape(newshape, order) # 改变数组形状而不更改其数据
  newshape :int或int的多元组
```

```
np.linspace(start, stop, num) //生成一个等间距的数组
- start：指定序列的起始值。
- stop：指定序列的结束值。
- num：指定生成的序列的长度。
```

```
np.square(x:np.linspace, y) //计算每个元素的平方
```

```
np.poly1d(p: array) # 生成多项式
 p: 多项式系数

np.poly1d([1, 2, 3])
1 x + 2 x + 3
```

```
np.dot(x, p) 计算两个数组的点积
```
## scipy.optimize.leastsq 最小二乘法求系数
```
from scipy.optimize import leastsq

parameters = leastsq(err_func, p_init, args=(np.array(x), np.array(y)))
  err_func  残差函数
  p_init:array  预估最小系数
  args(x,y)

parameters[0] 最佳拟合参数

```

## matplotlib 可视化
```
from matplotlib import pyplot as plt

plt.scatter(x:array, y:array) # 绘制散点图
plt.plot(x, y "r") # 绘制线


# 绘制子图
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

axes[0, 0].scatter(x, y)
axes[0, 0].set_title("子图1")

axes[0, 1].scatter(x, y)
axes[0, 1].set_title("子图2")

axes[1, 0].scatter(x, y)
axes[1, 0].set_title("子图3")
```


## scikit-learn

### linear_model 线性模型
- LinearRegression()
```
from sklearn.linear_model import LinearRegression

model = LinearRegression()  # 定义线性回归模型

model.fit(X, y)  # 训练
 # X 训练数据,array-like, shape(n_samples, n_features)
 # y 目标值,array-like, shape (n_samples,) or shape(n_samples, n_targets)

# Attributes
model.intercept_  # 模型截距
model.coef_  # 模型拟合参数

model.predict([[x]]) # 预测
```

### PolynomialFeatures 多项式
```
from sklearn.preprocessing import PolynomialFeatures

poly_features = PolynomialFeatures(degree: int, include_bias=False)
  degree: n次多项式
poly_features.fit_transform(X列向量)
```

###  回归预测结果指标
平均绝对误差 MAE 和 均方误差 MSE 求解方法

```
from sklearn.metrics import mean_absolute_error, mean_squared_error

mae_ = mean_absolute_error(y_test, preds)
mse_ = mean_squared_error(y_test, preds)

print("scikit-learn MAE: ", mae_)
print("scikit-learn MSE: ", mse_)
```

