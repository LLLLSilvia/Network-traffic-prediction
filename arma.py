#  200比较合适
# 30  最合适
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARMA
import warnings
warnings.filterwarnings("ignore")
filename = 'E:/senior/traffic prediction/data/12-12-1.xlsx'
forrecastnum = 5
data = pd.read_excel(filename, index_col=u'date', nrows = 200 )  # 指定列为索引
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
data.plot()
plt.title('Time Series')
plt.show()
plot_acf(data)  # 绘制自相关图
plt.show()
plot_pacf(data).show()  # 偏自相关图
plt.show()
print(u'原始序列的ADF检验结果为：',ADF(data[u'dst']))  # 显示是平稳序列
"""
#  一阶差分
D_data=data.diff(periods=1).dropna()
D_data.columns=[u'销量差分']
D_data.plot()  # 时序图
plt.show()
plot_acf(D_data).show()  # 自相关图
plt.show()
plot_pacf(D_data).show()  # 偏自相关图
print(u'1阶差分序列的ADF检验结果为：',ADF(D_data[u'销量差分']))
print(u'差分序列的白噪声检验结果为：',acorr_ljungbox(D_data,lags=1))
"""
#  print(u'序列的白噪声检验结果为：',acorr_ljungbox(data, lags = None))

data[u'dst'] = data[u'dst'].astype(float)
pmax=int(len(data)/50)
qmax=int(len(data)/50)
bic_matrix = []
for p in range(pmax+1):
    tmp=[]
    for q in range(qmax+1):
        try:
            tmp.append(ARMA(data,(p,q)).fit().bic)
        except:
            tmp.append(None)
    bic_matrix.append(tmp)
bic_matrix=pd.DataFrame(bic_matrix)  # 从中可以找出最小值
#  print(bic_matrix)
p, q = bic_matrix.stack().idxmin()
print(u'bic最小的P值和q值为：%s、%s'%(p,q))

model = ARMA(data,(p,q)).fit()
model.summary2()  # 给出一份模型报告
forecast = model.forecast(5)  # 作为期5天的预测，返回预测结果、标准误差、置信区间
print(forecast)
