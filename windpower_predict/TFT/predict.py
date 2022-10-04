import torch
import torch.nn as nn
from Network import *
from data import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from mpl_finance import candlestick_ohlc


t = TFN(n_variables_past_continuous, n_variables_future_continuous,
            n_variables_past_discrete, n_variables_future_discrete, dim_model,
            n_quantiles = quantiles.shape[0], dropout_r = 0.2,
            n_attention_layers = n_attention_layers,n_lstm_layers = n_lstm_layers, n_heads = n_heads).cuda()

optimizer = torch.optim.Adam(t.parameters(), lr=learning_rate)

