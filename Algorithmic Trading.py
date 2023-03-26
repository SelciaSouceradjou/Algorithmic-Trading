#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

df = pd.read_csv('TSLA.csv')

df ["SMA20"] = df['Close'].rolling(20).mean()
df ["SMA50"] = df['Close'].rolling(50).mean()


df["Signal"] = 0.0
df["Signal"] = np.where(df['SMA20'] > df['SMA50'], 1.0, 0.0)

df["Position"] = df['Signal'].diff()



plt.figure(figsize = (20,10))

df['Close'].plot(color = 'k', label= 'Close')
df['SMA20'].plot(color = 'b',label = 'SMA20')
df['SMA50'].plot(color = 'g', label = 'SMA50')

plt.plot(df[df['Position'] == 1].index,
         df['SMA20'][df['Position'] == 1],
         '^', markersize = 15, color = 'g', label = 'buy')

plt.plot(df[df['Position'] == -1].index,
         df['SMA20'][df['Position'] == -1],
         'v', markersize = 15, color = 'r', label = 'sell')
plt.ylabel('USD', fontsize = 15 )
plt.xlabel('Date', fontsize = 15 )

plt.xticks(rotation=45)

plt.title('AMD', fontsize = 20)
plt.legend()
plt.grid()
plt.show()

df


# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime



df = pd.read_csv('TSLA.csv')

# Calculate the moving average, upper band, and lower band
df['SMA20'] = df['Close'].rolling(window=20).mean()
df['Upper'] = df['SMA20'] + 2 * df['Close'].rolling(window=20).std()
df['Lower'] = df['SMA20'] - 2 * df['Close'].rolling(window=20).std()

# Define a function to generate signals
def generate_signals(df):
    buy_signal = []
    sell_signal = []
    for i in range(len(df)):
        if df['Close'][i] < df['Lower'][i] and df['Close'][i-1] > df['Lower'][i-1]:
            buy_signal.append(df['Close'][i])
            sell_signal.append(np.nan)
        elif df['Close'][i] > df['Upper'][i] and df['Close'][i-1] < df['Upper'][i-1]:
            buy_signal.append(np.nan)
            sell_signal.append(df['Close'][i])
        else:
            buy_signal.append(np.nan)
            sell_signal.append(np.nan)
    return (buy_signal, sell_signal)

# Generate buy and sell signals
buy_signal, sell_signal = generate_signals(df)

# Plot the price, upper band, lower band, buy signals, and sell signals
plt.figure(figsize=(20,10))
plt.plot(df['Close'], label='Price')
plt.plot(df['SMA20'], label='SMA20')
plt.plot(df['Upper'], label='Upper Band')
plt.plot(df['Lower'], label='Lower Band')
plt.scatter(df.index, buy_signal, label='Buy', marker='^', color='green')
plt.scatter(df.index, sell_signal, label='Sell', marker='v', color='red')
plt.legend()
plt.show()

