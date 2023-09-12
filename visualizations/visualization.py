#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from yahoofinancials import YahooFinancials
get_ipython().run_line_magic('matplotlib', 'inline')


# # Download Data

# In[2]:


start_date = '2017-01-01'
end_date = '2017-12-31'
stock_code = 'NVDA'


# In[3]:


stock_data = YahooFinancials(stock_code).get_historical_price_data(start_date, end_date, 'daily')
price_data = stock_data[stock_code]['prices']


# In[4]:


columns = ['formatted_date', 'open', 'high', 'low', 'close', 'adjclose', 'volume']
new_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
# order dataframe columns
df = pd.DataFrame(data=price_data)[columns]
# rename dataframe columns
df = df.rename(index=str, columns=dict(zip(columns, new_columns)))


# In[5]:


df.head(10)


# In[6]:


# save to 'data' directory
df.to_csv('../data/{}_{}.csv'.format(stock_code, start_date[:4]))


# Alternatively, we can go to https://finance.yahoo.com/ to look up a stock and download its historical data by setting the stock's time period and frequency.

# # Visualize Data

# In[7]:


df = pd.read_csv('../data/^GSPC_2016.csv')
df.head(10)


# In[8]:


df.shape


# In[9]:


plt.figure(figsize=(15, 5), dpi=100)
plt.plot(df['Date'], df['Close'], color='black', label='S&P 500 2016')
plt.xticks(np.linspace(0, len(df), 10))
plt.legend()
plt.grid()


# In[10]:


fig = plt.figure(figsize=(15, 5), dpi=100)
ax = fig.add_subplot(1,1,1)
ax.plot(df['Date'], df['Close'], color='black', label='S&P 500 2016')
ax.set_xticks(np.linspace(0, len(df), 10))
ax.legend()
ax.grid()


# In[ ]:




