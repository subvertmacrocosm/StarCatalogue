# 自定义函数，curve_fit支持自定义函数的形式进行拟合，这里定义的是指数函数的形式
# 包括自变量x和a，b，c三个参数
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

def func(x, a, b, c):
    return a * np.exp(-b * x) + c


# 产生数据
xdata = np.linspace(0, 4, 50)  # x从0到4取50个点
y = func(xdata, 2.5, 1.3, 0.5)  # 在x取xdata，a，b，c分别取2.5, 1.3, 0.5条件下，运用自定义函数计算y的值

# 在y上产生一些扰动模拟真实数据
np.random.seed(1729)
# 产生均值为0，标准差为1，维度为xdata大小的正态分布随机抽样0.2倍的扰动
y_noise = 0.2 * np.random.normal(size=xdata.size)
ydata = y + y_noise
plt.plot(xdata, ydata, 'b-', label='data')

# 利用“真实”数据进行曲线拟合
popt, pcov = curve_fit(func, xdata, ydata)  # 拟合方程，参数包括func，xdata，ydata，
# 有popt和pcov两个个参数，其中popt参数为a，b，c，pcov为拟合参数的协方差

# plot出拟合曲线，其中的y使用拟合方程和xdata求出
plt.plot(xdata, func(xdata, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

#     如果参数本身有范围，则可以设置参数的范围，如 0 <= a <= 3,
#     0 <= b <= 1 and 0 <= c <= 0.5:
popt, pcov = curve_fit(func, xdata, ydata, bounds=(0, [3., 1., 0.5]))  # bounds为限定a，b，c参数的范围

plt.plot(xdata, func(xdata, *popt), 'g--',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt))

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()