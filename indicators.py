import numpy as np
import pandas as pd

depth = 1

def moving_average(x, n):
    x_new = np.zeros_like(x)
    kernel = np.ones(n)/n
    for i in range(n-1):
        x_new[i] = np.mean(x[:i+1])
    x_new[n-1:] = np.convolve(x,kernel,mode='valid')
    return x_new

def moving_std(x, n):
    rolling = pd.Series(x).rolling(n, min_periods=1)
    std = rolling.std().values
    std[0] = 0
    return std

def moving_absmax(x, n):
    rolling = pd.Series(x).rolling(n)
    return np.max([rolling.max().values, -rolling.min().values], axis=0)

def macd(x, n1=12, n2=26, norm=False):
    if norm:
        return moving_average(x,n1) / moving_average(x,n2) - 1
    else:
        return moving_average(x,n1) - moving_average(x,n2)

def macd_oscillator(x, n1=12, n2=26, n3=9, norm=False):
    macd_tmp = macd(x, n1, n2, norm=norm)
    return macd_tmp - moving_average(macd_tmp, n=n3)

def minmax(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

def maxnorm(x):
    return x/np.max(np.abs(x))

def moving_minmax(x, n):
    rolling = pd.Series(x).rolling(n)
    x_min, x_max = rolling.min().values, rolling.max().values
    for i in range(n-1):
        x_min[i] = x[:i+1].min()
        x_max[i] = x[:i+1].max()
    return x_min, x_max

def sto_k(close, low, high, n=15):
    lowest, _ = moving_minmax(low, n)
    _, highest = moving_minmax(high, n)
    return (close - lowest) / np.clip(highest - lowest, depth, None) * 100

def sto_slow(close, low, high, n1=15, n2=5, n3=3):
    fast_K = sto_k(close, low, high, n1)
    slow_K = moving_average(fast_K, n2)
    slow_D = moving_average(slow_K, n3)
    return slow_K, slow_D

def get_rsi(x, n):
    m_Df = pd.DataFrame(x)
    U = np.where(m_Df.diff(1) > 0, m_Df.diff(1), 0)
    D = np.where(m_Df.diff(1) < 0, m_Df.diff(1) *(-1), 0)
    AU = pd.DataFrame(U).rolling( window=n, min_periods=1).mean()
    AD = pd.DataFrame(D).rolling( window=n, min_periods=1).mean()
    RSI = AU.div(np.clip(AD + AU, depth, None)) *100
    RSI[0][0] = 50
    return RSI[0].values

def bollinger(x, n=20, s=2):
    ma = moving_average(x, n)
    ms = moving_std(x, n)
    return ma + s * ms, ma - s * ms

def get_reverse(original, start, max_length):
    gm = np.exp(np.mean(np.log(original['close'][start : start + max_length])))
    new = original.copy()
    new['open'] = depth * np.round(gm ** 2 / original['open'] / depth)
    new['low'] = depth * np.round(gm ** 2 / original['high'] / depth)
    new['high'] = depth * np.round(gm ** 2 / original['low'] / depth)
    new['close'] = depth * np.round(gm ** 2 / original['close'] / depth)
    return new

def get_mixup(x1, x2, depth=depth, beta=True):
    if beta:
        f = np.random.beta(0.5, 0.5)
    else:
        f = np.random.uniform()
    price_index = ['open', 'low', 'high', 'close']
    xnew = x1.copy()
    xnew[:] = x1.values ** f * x2.values ** (1 - f)
    xnew[price_index] = depth * np.round(xnew[price_index] / depth)
    return xnew

def add_random_offset(data, level=0.05, depth=depth):
    pmin = min(data['low'])
    offset = level * pmin * np.random.uniform(-1, 1)
    offset = depth * np.round(offset / depth)
    price_index = ['open', 'low', 'high', 'close']
    data[price_index] += offset
    return data

