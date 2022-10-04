import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from math import sqrt
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

series = pd.read_csv("../dataset/Turbine_Data.csv")
# xuan ze xiang tong de shi jian buchang
df_seleced = series[['Unnamed: 0', 'ActivePower','GearboxBearingTemperature','GeneratorWinding1Temperature','MainBoxTemperature','WindSpeed']]

#print(df_seleced.tail())
input_df = df_seleced[['ActivePower','GearboxBearingTemperature','GeneratorWinding1Temperature','MainBoxTemperature','WindSpeed']]

rng = pd.date_range('2017-12-31',periods = 118224, freq='10T')
time_df = pd.DataFrame(rng)

input_df = input_df.fillna(0).astype(float)
input_df = pd.concat((time_df, input_df), axis=1)

input_df = input_df.set_index(0)

input_df = input_df.loc['2019-10-21':]
print(input_df.isna().sum())

#自定义损失函数




# ping jia zhi biao
def forecast_accuracy(forecast, actual):
    forecast = np.array(forecast)
    actual = np.array(actual)
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None],
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None],
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    return({'mape':mape, 'me':me, 'mae': mae,
            'mpe': mpe, 'rmse':rmse,
            'corr':corr, 'minmax':minmax})


def create_X_Y(ts: np.array, lag=1, n_ahead=1, target_index=0) -> tuple:
    """
    A method to create X and Y matrix from a time series array for the training of
    deep learning models
    """
    # Extracting the number of features that are passed from the array
    n_features = ts.shape[1]

    # Creating placeholder lists
    X, Y = [], []

    if len(ts) - lag <= 0:
        X.append(ts)
    else:
        for i in range(len(ts) - lag - n_ahead):
            Y.append(ts[(i + lag):(i + lag + n_ahead), target_index])
            X.append(ts[i:(i + lag)])

    X, Y = np.array(X), np.array(Y)

    # Reshaping the X array to an LSTM input shape
    X = np.reshape(X, (X.shape[0], lag, n_features))

    return X, Y

# Number of lags (steps back in 10min intervals) to use for models
lag = 360
# Steps in future to forecast (steps in 10min intervals)
n_ahead = 144
# ratio of observations for training from total series
train_share = 0.8
# training epochs
epochs = 20
# Batch size , which is the number of samples of lags
batch_size = 512
# Learning rate
lr = 0.001
# The features for the modeling
features_final = ['ActivePower','GearboxBearingTemperature','GeneratorWinding1Temperature','MainBoxTemperature','WindSpeed']

# Subseting only the needed columns
ts = input_df[features_final]


#Scaling data between 0 and 1
scaler = MinMaxScaler()
scaler.fit(ts)
ts_scaled = scaler.transform(ts)
X, Y = create_X_Y(ts_scaled, lag=lag, n_ahead=n_ahead)
Xtrain, Ytrain = X[0:int(X.shape[0] * train_share)], Y[0:int(X.shape[0] * train_share)]
Xtest, Ytest = X[int(X.shape[0] * train_share):], Y[int(X.shape[0] * train_share):]

# text = scaler.inverse_transform(ts_scaled)
# textdataX, textdataY = create_X_Y(text,lag=lag, n_ahead = n_ahead)
# testX , textY = textdataX[int(textdataX.shape[0] * train_share):], textdataY[int(textdataX.shape[0] * train_share):]



#Neural Network Model configuration, this is a Vanilla LSTM model
model = tf.keras.Sequential()
#model.add(tf.keras.layers.LSTM(16, activation='relu', return_sequences=False))
model.add(tf.compat.v1.keras.layers.CuDNNLSTM(32, return_sequences=False))
#model.add(tf.keras.layers.CuDNNLSTM(32, return_sequences=False)) #you can try to use the 10x faster GPU accelerated CuDNNLSTM instaed of the Vanilla LSTM above, but do not forget to set up the notebook accelerator to "GPU"
model.add(tf.keras.layers.Dense(144))
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001), loss='mae', metrics='mae')


if os.path.exists('./my_model.ckpt.index'):
    print('--------load the model-----------')
    model.load_weights('./my_model.ckpt')

#set up early stop function to stop training when val_loss difference is higher than 0.001
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, min_delta=0.001)
model_check = tf.keras.callbacks.ModelCheckpoint(filepath='./my_model.ckpt', monitor='val_loss', save_weights_only=True, save_best_only=True)
#If the model does not converge accurately, you need check if it is a input data quality issue, introduce a dropout layer, or you can try adjusting the number of hidden nodes
history = model.fit(Xtrain, Ytrain,epochs=epochs, validation_data=(Xtest, Ytest), shuffle=False, callbacks=[early_stopping,model_check])

predicted_data = model.predict(Xtest)


Ytest1 = Ytest.flatten()[50000:50500]

predicted_data1 = predicted_data.flatten()[50000:50500]
# forecast_accuracy(predicted_data, Ytest)


plt.title('text')
plt.plot(Ytest1, color='red', label='Actual Power')
plt.plot(predicted_data1, color='blue', label='Predicted Power')
# plt.plot(textdata, color='red', label='Actual Power')
# plt.plot(predicted_data, color='blue', label='Predicted Power')
plt.legend()
plt.show()


