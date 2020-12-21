#  https://www.cnblogs.com/harrylyx/p/11852149.html
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
"""
filename = 'E:/senior/traffic prediction/data/12-12-1.xlsx'
data = pd.read_excel(filename, index_col=u'date',nrows = 40 )  # 指定列为索引
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
data1= pd.read_excel(filename, index_col=u'date',nrows = 45 )  # 指定列为索引
"""

filename = 'E:/senior/traffic prediction/data/12-12-3.xlsx'
data = pd.read_excel(filename, nrows = 40 )  # 指定列为索引
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
data1= pd.read_excel(filename, nrows = 45 )  # 指定列为索引




"""
data.plot()
plt.title('Time Series')
plt.show()
"""
"""
data = [253993,275396.2,315229.5,356949.6,400158.2,442431.7,495102.9,570164.8,640993.1,704250.4,767455.4,781807.8,776332.3,794161.7,834177.7,931651.5,1028390,1114914]
plt.plot(data)
plt.show()
"""
"""
#  简单指数平滑(SES)
# Simple Exponential Smoothing
fit1 = SimpleExpSmoothing(data).fit(smoothing_level=0.2,optimized=False)
# plot
l1, = plt.plot(list(fit1.fittedvalues) + list(fit1.forecast(5)), marker='o')


fit2 = SimpleExpSmoothing(data).fit(smoothing_level=0.6,optimized=False)
# plot
l2, = plt.plot(list(fit2.fittedvalues) + list(fit2.forecast(5)), marker='o')


fit3 = SimpleExpSmoothing(data).fit()
# plot
l3, = plt.plot(list(fit3.fittedvalues) + list(fit3.forecast(5)), marker='o')

l4, = plt.plot(data1, marker='o')
plt.legend(handles = [l1, l2, l3, l4], labels = ['a=0.2', 'a=0.6', 'auto', 'data'], loc = 'best', prop={'size': 7})
plt.show()
"""
#二次指数平滑

# Holt’s Method
fit1 = Holt(data).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
l1, = plt.plot(list(fit1.fittedvalues) + list(fit1.forecast(5)), marker='^')

fit2 = Holt(data, exponential=True).fit(smoothing_level=0.8, smoothing_slope=0.2, optimized=False)
l2, = plt.plot(list(fit2.fittedvalues) + list(fit2.forecast(5)), marker='.')

fit3 = Holt(data, damped=True).fit(smoothing_level=0.8, smoothing_slope=0.2)
l3, = plt.plot(list(fit3.fittedvalues) + list(fit3.forecast(5)), marker='.')

l4, = plt.plot(data1, marker='.')
plt.legend(handles = [l1, l2, l3, l4], labels = ["Holt's linear trend", "Exponential trend", "Additive damped trend", 'data'], loc = 'best', prop={'size': 7})
plt.show()
print(data)
print(fit3.forecast(5))
"""
#H-W

fit1 = ExponentialSmoothing(data, seasonal_periods=4, trend='add', seasonal='add').fit(use_boxcox=True)
fit2 = ExponentialSmoothing(data, seasonal_periods=4, trend='add', seasonal='mul').fit(use_boxcox=True)
fit3 = ExponentialSmoothing(data, seasonal_periods=4, trend='add', seasonal='add', damped=True).fit(use_boxcox=True)
fit4 = ExponentialSmoothing(data, seasonal_periods=4, trend='add', seasonal='mul', damped=True).fit(use_boxcox=True)

l1, = plt.plot(list(fit1.fittedvalues) + list(fit1.forecast(5)), marker='^')
l2, = plt.plot(list(fit2.fittedvalues) + list(fit2.forecast(5)), marker='*')
l3, = plt.plot(list(fit3.fittedvalues) + list(fit3.forecast(5)), marker='.')
l4, = plt.plot(list(fit4.fittedvalues) + list(fit4.forecast(5)), marker='.')

print(fit4.forecast(5))

l5, = plt.plot(data1, marker='.')
plt.legend(handles = [l1, l2, l3, l4, l5], labels = ["aa", "am", "aa damped", "am damped","data"], loc = 'best', prop={'size': 7})

plt.show()
"""