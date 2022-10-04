import torch
import torch.nn as nn
from Network import *
from data import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_finance import candlestick_ohlc

def add_data(data):
    if math.isnan(data) or data > 3:
        return 1
    return data
def reverse(data):
    if data < 0:
        return (data * -1)
    return data

data = pd.read_csv("dataset/Turbine_Data_Text.csv",index_col=0,low_memory=False)
# data = pd.read_csv("dataset/text.csv",index_col=0,low_memory=False)
# data['ActivePower'] = data['ActivePower'].apply(reverse)
# data  = data.fillna(0).astype(float)

data['ActivePower'] = data['ActivePower'].fillna(data['ActivePower'].mean())
data['GearboxBearingTemperature'] = data['GearboxBearingTemperature'].fillna(data['GearboxBearingTemperature'].mean())
data['GeneratorWinding1Temperature'] = data['GeneratorWinding1Temperature'].fillna(data['GeneratorWinding1Temperature'].mean())
data['MainBoxTemperature'] = data['MainBoxTemperature'].fillna(data['MainBoxTemperature'].mean())
data['WindSpeed'] = data['WindSpeed'].fillna(data['WindSpeed'].mean())
data['TurbineStatus'] = data['TurbineStatus'].apply(add_data)

continuous_columns = ['ActivePower', 'GearboxBearingTemperature', 'GeneratorWinding1Temperature', 'WindSpeed']
# discrete_columns = ['TimeOfHour']#, 'Day', 'Month']
discrete_columns = ['DayOfMonth']#, 'Month']
target_columns = ['ActivePower']


print("Loading : ")
# btc_data = load_data(['2018', '2019'], 'BTCUSDT', continuous_columns, '5m')
# btc_test_data = load_data(['2020'], 'BTCUSDT', continuous_columns, interval = '5m')
btc_data = data['2019-10-21 17:20:00+00:00':'2020-02-16 00:00:00+00:00']
btc_test_data = data['2020-02-16 00:00:10+00:00':'2020-03-30 23:50:00+00:00']


#input data shape
n_variables_past_continuous = 4
n_variables_future_continuous = 0
n_variables_past_discrete = [24]#144]#, 31, 12]
n_variables_future_discrete = [24]#144]#, 31, 12]

#hyperparams
batch_size = 12
test_batch_size = 12
dim_model = 120
n_lstm_layers = 4
n_attention_layers = 1
n_heads = 6
learning_rate = 0.0001

load_model = False
path = "model/model_1158.pt"


quantiles = torch.tensor([0.1, 0.5, 0.9]).float().type(torch.cuda.FloatTensor)
# quantiles = torch.tensor([0.1, 0.5, 0.9]).float().type(torch.FloatTensor)

#initialise
t = TFN(n_variables_past_continuous, n_variables_future_continuous,
            n_variables_past_discrete, n_variables_future_discrete, dim_model,
            n_quantiles = quantiles.shape[0], dropout_r = 0.2,
            n_attention_layers = n_attention_layers,n_lstm_layers = n_lstm_layers, n_heads = n_heads).cuda()
optimizer = torch.optim.Adam(t.parameters(), lr=learning_rate)

#try to load from checkpoint
if load_model:
    checkpoint = torch.load(path)
    t = checkpoint['model_state']
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    losses = checkpoint['losses']
    test_losses = checkpoint['test_losses']
    print("Loaded model from checkpoint")
else:
    losses = []
    test_losses = []
    print("No checkpoint loaded, initialising model")


#losses = []


fig = plt.figure()
ax = fig.add_subplot(411)
ax1 = fig.add_subplot(412)
ax2 = fig.add_subplot(413)
ax3 = fig.add_subplot(414)
plt.ion()

fig.show()
fig.canvas.draw()

#历史记录
past_seq_len = 360
#预测长度
future_seq_len = 144

btc_gen = get_batches(btc_data, past_seq_len,
                      future_seq_len, continuous_columns, discrete_columns,
                      target_columns, batch_size=batch_size)

test_btc_gen = get_batches(btc_test_data, past_seq_len,
                           future_seq_len, continuous_columns, discrete_columns,
                           target_columns, batch_size=batch_size, norm=btc_data)

best_text_loss = -1
t.train()
for e in range(5000):
    # forward pass
    optimizer.zero_grad()
    loss, net_out, vs_weights, given_data = forward_pass(t, btc_gen, batch_size, quantiles)
    net_out = net_out.cpu().detach()[0]

    t.eval()
    test_loss, _, _, _ = forward_pass(t, test_btc_gen, test_batch_size, quantiles)
    test_losses.append(test_loss.cpu().detach().numpy())
    t.train()

    print('epoch------',e,"--------",'train_loss：',loss,'text_loss：', test_loss)
    # backwards pass
    losses.append(loss.cpu().detach().numpy())
    loss.backward()
    optimizer.step()

    # loss graphs
    fig.tight_layout(pad=0.1)
    ax.clear()
    ax.title.set_text("Training loss")
    ax.plot(losses)

    ax1.clear()
    ax1.title.set_text("Test loss")
    ax1.plot(test_losses)

    # compare network out put and data
    ax2.clear()
    ax2.title.set_text("Network output comparison")
    # c = given_data[0][0].cpu()
    # a = torch.arange(-past_seq_len, 0).unsqueeze(-1).unsqueeze(-1).float()
    # c = torch.cat((a, c), dim=1)
    # candlestick_ohlc(ax2, c.squeeze(), colorup="green", colordown="red")

    ax2.plot(net_out[:, 0], color="red")
    ax2.plot(net_out[:, 1], color="blue")
    ax2.plot(net_out[:, 2], color="red")
    ax2.plot(given_data[3].cpu().detach().numpy()[0], label="target", color="orange")

    # visualise variable selection weights

    vs_weights = torch.mean(torch.mean(vs_weights, dim=0), dim=0).squeeze()
    vs_weights = vs_weights.cpu().detach().numpy()
    ax3.clear()
    ax3.title.set_text("Variable Selection Weights")
    plt.xticks(rotation=-30)
    x = ['AP', 'GBT', 'GWT','WS','HourOfDay']  # , 'Eth_Open', 'Eth_High', 'Eth_Low', 'Eth_Close', 'Hour']#, 'Day', 'Month']
    ax3.bar(x=x, height=vs_weights)
    fig.canvas.draw()
    # 显示loss最少的图片
    if(best_text_loss > test_loss or e == 0):
        fig.show()
    # 保存最少的loss模型
    if (best_text_loss > test_loss or e == 0):
        best_text_loss = test_loss
        torch.save({'model_state': t,
                    'optimizer_state': optimizer.state_dict(),
                    'losses': losses, 'test_losses': test_losses}, "model/model_{}.pt".format(e))
    del loss
    del net_out
    del vs_weights
    del given_data


# Draw test cases
fig = plt.figure()
axes = []
batch_size_ = 4

for i in range(batch_size_):
    axes.append(fig.add_subplot(411 + i))

test_btc_gen = get_batches(btc_test_data, past_seq_len,
                           future_seq_len, continuous_columns, discrete_columns,
                           target_columns, batch_size=batch_size_, norm=btc_data)

loss, net_out, vs_weights, given_data = forward_pass(t, test_btc_gen, batch_size_, quantiles)
net_out = net_out.cpu().detach()
t.eval()
for idx, a in enumerate(axes):
    a.clear()

    # c = given_data[0][idx].cpu()
    #
    # b = torch.arange(-past_seq_len, 0).unsqueeze(-1).unsqueeze(-1).float()
    # c = torch.cat((b, c), dim=1)
    # candlestick_ohlc(a, c.squeeze(), colorup="green", colordown="red")

    a.plot(net_out[idx][:, 0], color="red")
    a.plot(net_out[idx][:, 1], color="blue")
    a.plot(net_out[idx][:, 2], color="red")
    a.plot(given_data[3].cpu().detach().numpy()[idx], label="target", color="orange")

t.train()
plt.ion()

fig.show()
# fig.canvas.draw()

print("train_mean_loss", np.array(losses).mean())
print("text_mean_loss", np.array(test_losses).mean())
print(np.array(test_losses).min())


