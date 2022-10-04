
# from plotly.offline import init_notebook_mode, iplot
import plotly.offline as py
py.init_notebook_mode(connected=True)         # initiate notebook for offline plot
import plotly.io as pio

import plotly.express as px
from scipy import stats
from data import *
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import math
import matplotlib.pyplot as plt
import plotly.graph_objects as go



def add_data(data):
    if math.isnan(data) or data > 3:
        return 1
    return data
def reverse(data):
    if data < 0:
        return (data * -1)
    return data

data = pd.read_csv("dataset/Turbine_Data_Text.csv",index_col=0,low_memory=False)
data['ActivePower'] = data['ActivePower'].apply(reverse)

data['ActivePower'] = data['ActivePower'].fillna(data['ActivePower'].mean())
data['GearboxBearingTemperature'] = data['GearboxBearingTemperature'].fillna(data['GearboxBearingTemperature'].mean())
data['GeneratorWinding1Temperature'] = data['GeneratorWinding1Temperature'].fillna(data['GeneratorWinding1Temperature'].mean())
data['MainBoxTemperature'] = data['MainBoxTemperature'].fillna(data['MainBoxTemperature'].mean())
data['WindSpeed'] = data['WindSpeed'].fillna(data['WindSpeed'].mean())
# data['TurbineStatus'] = data['TurbineStatus'].apply(add_data)

continuous_columns = ['ActivePower', 'GearboxBearingTemperature', 'GeneratorWinding1Temperature', 'MainBoxTemperature', 'WindSpeed']
discrete_columns = ['HourOfDay']#, 'Day', 'Month']
target_columns = ['ActivePower']
g = get_batches(data,1000, 1, continuous_columns, discrete_columns,
                target_columns, batch_size = 1, gpu = False, normalise = False)

in_seq_continuous, in_seq_discrete, out_seq, target_seq  = next(g)
c = in_seq_continuous[:,:,0].squeeze()

c = pd.DataFrame(c)[0]
# print(adfuller(c)[1])
# print(c)
c = pd.DataFrame(stats.boxcox(c)[0])[0]

c = c.pct_change()

#c =(c - c.mean()) / c.std()


c = c.cumsum()
c = c.drop(c.index[0])

# print(adfuller(c)[1])
#print(c[0])

#c = stats.boxcox(c)[0]
#修改
# fig = px.line(c)
# fig.show()

#print(in_seq_discrete.scatter_(1, in_seq_discrete.long(), 1))


#print("Dickey–Fuller test {}".format())

#print(in_seq_discrete)
#print(in_seq_continuous[:,0])
fig = go.Figure(data=[go.Candlestick(open=in_seq_continuous[:,:,0].squeeze(),
                high=in_seq_continuous[:,:,1].squeeze(),
                low=in_seq_continuous[:,:,2].squeeze(),
                close=in_seq_continuous[:,:,3].squeeze())])


fig2 = px.line(target_seq)


fig.show()
# py.iplot(fig)
# plt.show()
# fig2.show()