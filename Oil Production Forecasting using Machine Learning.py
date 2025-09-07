#!/usr/bin/env python
# coding: utf-8

# ## Oil Production Forecasting
# 
# > How to Pose a Time Series Forecasting problem as a Supervised learning Algorithm!

# In[812]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[813]:


df = pd.read_csv('Volve_P-12_DatesCorrected.csv',
                index_col=0, parse_dates=True)


# In[814]:


df_model_preparation = df[df.index.year <=2014]
df_model_testing = df[df.index.year>=2015]


# In[815]:


df_model_preparation.head()


# In[816]:


df_original = df_model_preparation.copy()


# In[817]:


df = df_original.loc[:,['BORE_OIL_VOL']]


# In[818]:


df


# In[819]:


df.plot(figsize=(12,5))


# In[820]:


df.shift(1)


# In[821]:


window= 60


# In[822]:


def TS_to_Xy(df,n_lags=3, window=window):
    
    shifted_ys = []
    
    for i in range(1,n_lags+1):
        df[f'y_{i}'] = 0
        df[f'y_{i}'] = df.shift(i+window)
    
    
    
    return df


# In[823]:


df_Xy = TS_to_Xy(df.rolling('30D').mean(),1)


# In[824]:


df_Xy.head()


# In[825]:


df_Xy = df_Xy.fillna(method='bfill')


# In[826]:


df_Xy['y_1'].plot(figsize=(12,5))


# In[827]:


from sklearn.model_selection import train_test_split


# In[828]:


X = df_Xy[[col for col in df_Xy.columns if col.startswith('y')]]
y = df_Xy['BORE_OIL_VOL']


# In[829]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42,shuffle=False)


# In[830]:


from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error


# In[831]:


rf = RandomForestRegressor()


# In[832]:


rf.fit(X_train,y_train)


# In[833]:


yp_train = rf.predict(X_train)


# In[834]:


plt.figure(figsize=(12,4))

plt.plot(X_train.index, y_train)
plt.plot(X_train.index, yp_train)

plt.title(f'RMSE = {np.sqrt(mean_squared_error(y_train,yp_train))}', size=20)


# In[835]:


ti = 800;  tf = 1500 

plt.figure(figsize=(15,5))

plt.scatter(df_Xy.index[ti:tf] , df_Xy['BORE_OIL_VOL'][ti:tf], marker='o', label='Actual Oil Rates', s=200)
plt.plot(df_Xy.index[ti:tf] , rf.predict(df_Xy[['BORE_OIL_VOL']][ti-window:tf-window]),color='k', label='Predicted Rates')


# plt.plot(df_Xy.index[ti:tf] , df_Xy['BORE_OIL_VOL'][ti:tf], marker='x', label='Actual Oil Rates')

plt.grid()
plt.legend()


# In[836]:


yp_test = rf.predict(X_test)


# In[837]:


plt.figure(figsize=(15,5))
plt.scatter(X_test.index, y_test)
plt.plot(X_test.index, yp_test, color='k')


# In[838]:


plt.figure(figsize=(15,5))






plt.scatter(X_train.index, y_train, label='Actual Oil Rates (Training)')
plt.plot(X_train.index, yp_train, color='k', label='Predicted Oil Rates (Training)')



plt.axvline(X_test.index[0], color='k')

plt.scatter(X_test.index, y_test, color='green', label='Actual Oil Rates (Validation)')
plt.plot(X_test.index, yp_test, color='brown', label='Predicted Oil Rates (Validation)')




plt.grid()
plt.legend()


# In[839]:


X_test


# ## Forecasting

# In[840]:


last_input_rates = X_test.iloc[-window:,0:]

next_forecasts = np.array(rf.predict(last_input_rates))


# In[841]:


# len(next_forecasts)
df_model_testing.index[:window]


# In[842]:


df_model_testing


# In[843]:


plt.scatter(df_model_testing.index[:window], df_model_testing['BORE_OIL_VOL'][:window], label='Actual Future Rates')
plt.plot(df_model_testing.index[:window], next_forecasts.reshape(window,1), label='Model Forecasted Rates', color='k')


# In[848]:


plt.figure(figsize=(15,5))






plt.scatter(X_train.index, y_train, label='Actual Oil Rates (Training)')
plt.plot(X_train.index, yp_train, color='k', label='Predicted Oil Rates (Training)')



plt.axvline(X_test.index[0], color='k')

# plt.scatter(X_test.index, y_test, color='green', label='Actual Oil Rates (Validation)')
plt.plot(X_test.index, yp_test, label='Predicted Oil Rates (Validation)', color='green')

plt.axvline(df_model_testing.index[0])

# plt.scatter(df_model_testing.index[:window], df_model_testing['BORE_OIL_VOL'][:window], label='Actual Future Rates')
plt.plot(df_model_testing.index[:window], next_forecasts, label='Model Forecasted Rates', color='brown')






plt.ylabel('Oil Production rates, STB/D', size=20);
plt.xlabel('Date', size=20)
plt.grid()
plt.legend()


# In[ ]:




