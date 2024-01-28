# -*- coding: utf-8 -*-
"""
Created on Sat May 15 20:34:05 2021

@author: Thiago
"""
#================================================
# PACKAGES
#================================================

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import os

#================================================
# FUNCTIONS
#================================================
# MAPE FUNCTION
def mape_vectorized(true, pred): 
    mask = true != 0
    return (np.fabs(true - pred)/true)[mask].mean()*100

#==================================================================
# series_to_supervised FUNCTION - author: Jason Brownlee
#================================================

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
  # dataset varaibles quantity
  n_vars = 1 if type(data) is list else data.shape[1]
  # dataframe transformation
  df = pd.DataFrame(data)
  cols, names = list(), list()

  # input sequence (t-n, ... t-1). lags before time t
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

  # forecast sequence (t, t+1, ... t+n). lags after time t
    
  for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
      names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
      names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
  # put it all together
  agg = pd.concat(cols, axis=1)
  agg.columns = names
  # drop rows with NaN values
  if dropnan:
    agg.dropna(inplace=True)
  return agg

#Variables for dataset creation with grid search results
MODEL_ID = []
PARAMS = []
VARS = []
LAGS = []
RMSE_one = []
MAPE_one = []
NAIVE_rmse = []
NAIVE_mape = []

#================================================
# LOAD DATA
#================================================
os.getcwd()
os.chdir('D:/CLISAU/RAW/DO/OBITOS_ATUALIZADO/processed')
#content/drive/Shared drives/Clima&Saúde/Dados/Dados_Saude/Obitos_SSC/data/processed/2001-2018_obitos_clima_diario.csv
#Deaths and environment
df = pd.read_csv('2001-2018_obitos_clima_diario.csv')
df['DATE'] = pd.to_datetime(df['DATE'],dayfirst=True)
df = df.set_index('DATE')

#================================================
#GRID 01
#================================================
variables = ['CO_MEAN','MP10_MEAN','TMIN_IAC']

#================================================
# VARIABLE SELECTION
#================================================
# Variables
endog = df.loc[:]['all']
exog = df.loc[:][variables]
endog = endog.asfreq('D')
exog = exog.asfreq('D')
#nobs = endog.shape[0]

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))

#fill exogenous nan with mean (for each exogenous variable)
exog = exog.fillna(exog.mean())

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))
exog.head(2)

#================================================
# Train/Test split 80-20
#================================================
n_train = int(0.8*len(endog))

endog_train = endog.iloc[:n_train].asfreq('D')
endog_test = endog.iloc[n_train:].asfreq('D')

exog_train = exog.iloc[:n_train].asfreq('D')
exog_test = exog.iloc[n_train:].asfreq('D')

lags = list(range(1,16))
for n_hours in lags:
    LAGS.append(n_hours)
    VARS.append(variables)
    #================================================
    # Normalization
    #================================================
    train = pd.merge(endog_train,exog_train,on='DATE')
    test = pd.merge(endog_test,exog_test,on='DATE')
    test.head(3)
    train.head(3)
    
    '''
    if exog does contain endog, mode = 1
    if exog doesn't contain endog, mode = 0
    '''
    mode = 0
    
    if mode == 1:
      flag = 0
    else:
      flag = 1
    
    # load dataset
    values = train.values
    # ensure all data is float
    np.set_printoptions(suppress=True) #supress scientific notation
    values = values.astype('float32')
    
    # normalize train features
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(values)
    # normalize test based on train scaler
    test_scaled = scaler.transform(test.values)
    # specify the number of lag (hours in this case, since each line equals a hour)
    #n_features is the number of the exogenous variables
    
    n_features = len(exog.columns)+mode
    
    #================================================
    # Supervised Learning
    #================================================
    # frame as supervised learning
    
    #Here I used scaled[:,1:] because I exlcuded the y (deaths all) variable from the function below.
    reframed = series_to_supervised(train_scaled[:,flag:], n_hours, 0)
    reframed_test = series_to_supervised(test_scaled[:,flag:], n_hours, 0)
    print(reframed.shape) #nº observation x ((nºfeatures + 1) * nºlags)
    
    reframed.head()
    
    #================================================
    # Reframed as 3d data
    #================================================
    # split into train and test sets
    values = reframed.values
    #Train data = 1 year of data
    '''
    To speed up the training of the model for this demonstration, we will only fit 
    the model on the first year of data (365 days * 24hours (1 day)), then evaluate it on the remaining 4 years 
    of data. If you have time, consider exploring the inverted version of 
    this test harness.
    '''
    #n_train_hours = 365 * 24
    #train = values[:n_train_hours, :]
    #test = values[n_train_hours:, :]
    
    # split into input and outputs
    
    n_obs = n_hours * n_features
    #Train and test
    
    train_X, train_y = values, train_scaled[n_hours:,0]
    test_X, test_y = reframed_test.values, test_scaled[n_hours:,0]
    print(train_X.shape, len(train_X), train_y.shape)
    # reshape input (train_X and test_X) to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
    #================================================
    # Create LSTM model for GridSearch
    #================================================
    optimizer = ['adam']
    batch_size = [300]
    epochs = [300]
    init_mode = ['glorot_uniform']
    layers = ['deep']
    callback = ['ES_30']
    param_grid = dict(batch_size=batch_size, epochs=epochs,optimizer=optimizer,init_mode=init_mode,layers=layers,callback=callback)
    
    model_id = 'deep'
    MODEL_ID.append(model_id)
    PARAMS.append(param_grid)
    # create model
    model01 = Sequential()
    model01.add(LSTM(16, return_sequences=True,input_shape=(train_X.shape[1], train_X.shape[2])))
    model01.add(LSTM(32, return_sequences=True))
    model01.add(LSTM(64, return_sequences=True))
    model01.add(LSTM(32, return_sequences=True))
    model01.add(LSTM(16))
    model01.add(Dense(1))
    
    optimizer = tf.keras.optimizers.Adam()
    model01.compile(loss='mse', optimizer=optimizer)
    
    # fit network
    from keras.callbacks import EarlyStopping
    stop = EarlyStopping(monitor='val_loss', patience=30,verbose=0,restore_best_weights=True)
    callbacks_list = [stop]
    
    history = model01.fit(train_X, train_y, epochs=300, batch_size=300, validation_data=(test_X, test_y),callbacks = callbacks_list, verbose=2, shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
     
    # make a prediction
    yhat = model01.predict(test_X)
    
    #================================================
    # Invert Scaler
    #================================================
    #Reshape test_X to get 7 random columns
    test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
    
    #gambiarra
    n_features = n_features-mode
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, -n_features:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    '''
    Q: why do we use -11 in 'test_X[:, -11:]' especially?
    A: We are only interested in inverting the target, but the transform requires 
    the same columns when inverting as when transforming. Therefore we are adding 
    the target with other input vars for the inverse operation.
    '''
    # invert scaling for actual
    
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, -n_features:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    mape = mape_vectorized(inv_y,inv_yhat)
    print('Test MAPE: %.3f' % mape)
    
    #NAIVE - MEAN VALUE: evaluate if I just predict the average of the timeseries
    naive = pd.Series(np.repeat(df['all'][n_train+n_hours:].mean(),len(df['all'][n_train+n_hours:])))
    naive.index = df['all'][n_train+n_hours:].index
    
    naive_mse = mean_squared_error(df['all'][n_train+n_hours:],naive)
    print('\nNAIVE RMSE: ',round(np.sqrt(naive_mse),3))
    naive_mape = mape_vectorized(df['all'][n_train+n_hours:],naive)
    print('NAIVE MAPE: ',round(naive_mape,2),'%')
    
    RMSE_one.append(round(rmse,3))
    MAPE_one.append(round(mape,2))
    NAIVE_rmse.append(round(np.sqrt(naive_mse),2))
    NAIVE_mape.append(round(naive_mape,2))
    
    #================================================
    # Plot Prediction
    #================================================
    # plot time series
    plt.figure(figsize=[20,3])
    #test.iloc[n_hours:].index
    plt.plot(test.iloc[n_hours:].index, inv_y[:], label='Observed',alpha=0.4,color='gray')
    plt.plot(test.iloc[n_hours:].index, inv_yhat[:], label='Predicted',color='dodgerblue')
    #plt.title('Prediction vs Observed\nLast 365 days (~1 year)')
    plt.xlabel('Time',fontsize=15)
    plt.ylabel('Deaths',fontsize=15)
    plt.legend(fontsize=15,loc='upper left')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
    
report_LSTM_deep01 = pd.DataFrame()
report_LSTM_deep01['MODEL'] = MODEL_ID
report_LSTM_deep01['PARAM'] = PARAMS
report_LSTM_deep01['VARS'] = VARS
report_LSTM_deep01['LAGS'] = LAGS
report_LSTM_deep01['RMSE_ONE'] = RMSE_one
report_LSTM_deep01['MAPE_ONE'] = MAPE_one
report_LSTM_deep01['NAIVE_RMSE'] = NAIVE_rmse
report_LSTM_deep01['NAIVE_MAPE'] = NAIVE_mape

os.chdir('C:/Users/trbella/Desktop/Thiago/Python/LSTM_GRID_LAGS/results/daily')
report_LSTM_deep01.to_excel('report_LSTM_deep01.xlsx')
    
#================================
# Clean the envinronment
#================================
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

import gc
gc.collect()
    
    
#=========================
#  PART 02
#=========================
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import os

#================================================
# FUNCTIONS
#================================================
# MAPE FUNCTION
def mape_vectorized(true, pred): 
    mask = true != 0
    return (np.fabs(true - pred)/true)[mask].mean()*100

#==================================================================
# series_to_supervised FUNCTION - author: Jason Brownlee
#================================================

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
  # dataset varaibles quantity
  n_vars = 1 if type(data) is list else data.shape[1]
  # dataframe transformation
  df = pd.DataFrame(data)
  cols, names = list(), list()

  # input sequence (t-n, ... t-1). lags before time t
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

  # forecast sequence (t, t+1, ... t+n). lags after time t
    
  for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
      names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
      names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
  # put it all together
  agg = pd.concat(cols, axis=1)
  agg.columns = names
  # drop rows with NaN values
  if dropnan:
    agg.dropna(inplace=True)
  return agg

#Variables for dataset creation with grid search results
MODEL_ID = []
PARAMS = []
VARS = []
LAGS = []
RMSE_one = []
MAPE_one = []
NAIVE_rmse = []
NAIVE_mape = []

#================================================
# LOAD DATA
#================================================
os.getcwd()
os.chdir('D:/CLISAU/RAW/DO/OBITOS_ATUALIZADO/processed')
#content/drive/Shared drives/Clima&Saúde/Dados/Dados_Saude/Obitos_SSC/data/processed/2001-2018_obitos_clima_diario.csv
#Deaths and environment
df = pd.read_csv('2001-2018_obitos_clima_diario.csv')
df['DATE'] = pd.to_datetime(df['DATE'],dayfirst=True)
df = df.set_index('DATE')

#================================================
#GRID 02
#================================================
variables = ['CO_MEAN','TMIN_IAC']

#================================================
# VARIABLE SELECTION
#================================================
# Variables
endog = df.loc[:]['all']
exog = df.loc[:][variables]
endog = endog.asfreq('D')
exog = exog.asfreq('D')
#nobs = endog.shape[0]

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))

#fill exogenous nan with mean (for each exogenous variable)
exog = exog.fillna(exog.mean())

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))
exog.head(2)

#================================================
# Train/Test split 80-20
#================================================
n_train = int(0.8*len(endog))

endog_train = endog.iloc[:n_train].asfreq('D')
endog_test = endog.iloc[n_train:].asfreq('D')

exog_train = exog.iloc[:n_train].asfreq('D')
exog_test = exog.iloc[n_train:].asfreq('D')

lags = list(range(1,16))
for n_hours in lags:
    LAGS.append(n_hours)
    VARS.append(variables)
    #================================================
    # Normalization
    #================================================
    train = pd.merge(endog_train,exog_train,on='DATE')
    test = pd.merge(endog_test,exog_test,on='DATE')
    test.head(3)
    train.head(3)
    
    '''
    if exog does contain endog, mode = 1
    if exog doesn't contain endog, mode = 0
    '''
    mode = 0
    
    if mode == 1:
      flag = 0
    else:
      flag = 1
    
    # load dataset
    values = train.values
    # ensure all data is float
    np.set_printoptions(suppress=True) #supress scientific notation
    values = values.astype('float32')
    
    # normalize train features
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(values)
    # normalize test based on train scaler
    test_scaled = scaler.transform(test.values)
    # specify the number of lag (hours in this case, since each line equals a hour)
    #n_features is the number of the exogenous variables
    
    n_features = len(exog.columns)+mode
    
    #================================================
    # Supervised Learning
    #================================================
    # frame as supervised learning
    
    #Here I used scaled[:,1:] because I exlcuded the y (deaths all) variable from the function below.
    reframed = series_to_supervised(train_scaled[:,flag:], n_hours, 0)
    reframed_test = series_to_supervised(test_scaled[:,flag:], n_hours, 0)
    print(reframed.shape) #nº observation x ((nºfeatures + 1) * nºlags)
    
    reframed.head()
    
    #================================================
    # Reframed as 3d data
    #================================================
    # split into train and test sets
    values = reframed.values
    #Train data = 1 year of data
    '''
    To speed up the training of the model for this demonstration, we will only fit 
    the model on the first year of data (365 days * 24hours (1 day)), then evaluate it on the remaining 4 years 
    of data. If you have time, consider exploring the inverted version of 
    this test harness.
    '''
    #n_train_hours = 365 * 24
    #train = values[:n_train_hours, :]
    #test = values[n_train_hours:, :]
    
    # split into input and outputs
    
    n_obs = n_hours * n_features
    #Train and test
    
    train_X, train_y = values, train_scaled[n_hours:,0]
    test_X, test_y = reframed_test.values, test_scaled[n_hours:,0]
    print(train_X.shape, len(train_X), train_y.shape)
    # reshape input (train_X and test_X) to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
    #================================================
    # Create LSTM model for GridSearch
    #================================================
    optimizer = ['adam']
    batch_size = [300]
    epochs = [300]
    init_mode = ['glorot_uniform']
    layers = ['deep']
    callback = ['ES_30']
    param_grid = dict(batch_size=batch_size, epochs=epochs,optimizer=optimizer,init_mode=init_mode,layers=layers,callback=callback)
    
    model_id = 'deep'
    MODEL_ID.append(model_id)
    PARAMS.append(param_grid)
    # create model
    model01 = Sequential()
    model01.add(LSTM(16, return_sequences=True,input_shape=(train_X.shape[1], train_X.shape[2])))
    model01.add(LSTM(32, return_sequences=True))
    model01.add(LSTM(64, return_sequences=True))
    model01.add(LSTM(32, return_sequences=True))
    model01.add(LSTM(16))
    model01.add(Dense(1))
    
    optimizer = tf.keras.optimizers.Adam()
    model01.compile(loss='mse', optimizer=optimizer)
    
    # fit network
    from keras.callbacks import EarlyStopping
    stop = EarlyStopping(monitor='val_loss', patience=30,verbose=0,restore_best_weights=True)
    callbacks_list = [stop]
    
    history = model01.fit(train_X, train_y, epochs=300, batch_size=300, validation_data=(test_X, test_y),callbacks = callbacks_list, verbose=2, shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
     
    # make a prediction
    yhat = model01.predict(test_X)
    
    #================================================
    # Invert Scaler
    #================================================
    #Reshape test_X to get 7 random columns
    test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
    
    #gambiarra
    n_features = n_features-mode
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, -n_features:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    '''
    Q: why do we use -11 in 'test_X[:, -11:]' especially?
    A: We are only interested in inverting the target, but the transform requires 
    the same columns when inverting as when transforming. Therefore we are adding 
    the target with other input vars for the inverse operation.
    '''
    # invert scaling for actual
    
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, -n_features:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    mape = mape_vectorized(inv_y,inv_yhat)
    print('Test MAPE: %.3f' % mape)
    
    #NAIVE - MEAN VALUE: evaluate if I just predict the average of the timeseries
    naive = pd.Series(np.repeat(df['all'][n_train+n_hours:].mean(),len(df['all'][n_train+n_hours:])))
    naive.index = df['all'][n_train+n_hours:].index
    
    naive_mse = mean_squared_error(df['all'][n_train+n_hours:],naive)
    print('\nNAIVE RMSE: ',round(np.sqrt(naive_mse),3))
    naive_mape = mape_vectorized(df['all'][n_train+n_hours:],naive)
    print('NAIVE MAPE: ',round(naive_mape,2),'%')
    
    RMSE_one.append(round(rmse,3))
    MAPE_one.append(round(mape,2))
    NAIVE_rmse.append(round(np.sqrt(naive_mse),2))
    NAIVE_mape.append(round(naive_mape,2))
    
    #================================================
    # Plot Prediction
    #================================================
    # plot time series
    plt.figure(figsize=[20,3])
    #test.iloc[n_hours:].index
    plt.plot(test.iloc[n_hours:].index, inv_y[:], label='Observed',alpha=0.4,color='gray')
    plt.plot(test.iloc[n_hours:].index, inv_yhat[:], label='Predicted',color='dodgerblue')
    #plt.title('Prediction vs Observed\nLast 365 days (~1 year)')
    plt.xlabel('Time',fontsize=15)
    plt.ylabel('Deaths',fontsize=15)
    plt.legend(fontsize=15,loc='upper left')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
    
report_LSTM_deep02 = pd.DataFrame()
report_LSTM_deep02['MODEL'] = MODEL_ID
report_LSTM_deep02['PARAM'] = PARAMS
report_LSTM_deep02['VARS'] = VARS
report_LSTM_deep02['LAGS'] = LAGS
report_LSTM_deep02['RMSE_ONE'] = RMSE_one
report_LSTM_deep02['MAPE_ONE'] = MAPE_one
report_LSTM_deep02['NAIVE_RMSE'] = NAIVE_rmse
report_LSTM_deep02['NAIVE_MAPE'] = NAIVE_mape

os.chdir('C:/Users/trbella/Desktop/Thiago/Python/LSTM_GRID_LAGS/results/daily')
report_LSTM_deep02.to_excel('report_LSTM_deep02.xlsx')
    
#================================
# Clean the envinronment
#================================
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

import gc
gc.collect()
    
    
#=====================    
# PARTE 03
#=====================
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import os

#================================================
# FUNCTIONS
#================================================
# MAPE FUNCTION
def mape_vectorized(true, pred): 
    mask = true != 0
    return (np.fabs(true - pred)/true)[mask].mean()*100

#==================================================================
# series_to_supervised FUNCTION - author: Jason Brownlee
#================================================

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
  # dataset varaibles quantity
  n_vars = 1 if type(data) is list else data.shape[1]
  # dataframe transformation
  df = pd.DataFrame(data)
  cols, names = list(), list()

  # input sequence (t-n, ... t-1). lags before time t
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

  # forecast sequence (t, t+1, ... t+n). lags after time t
    
  for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
      names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
      names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
  # put it all together
  agg = pd.concat(cols, axis=1)
  agg.columns = names
  # drop rows with NaN values
  if dropnan:
    agg.dropna(inplace=True)
  return agg

#Variables for dataset creation with grid search results
MODEL_ID = []
PARAMS = []
VARS = []
LAGS = []
RMSE_one = []
MAPE_one = []
NAIVE_rmse = []
NAIVE_mape = []

#================================================
# LOAD DATA
#================================================
os.getcwd()
os.chdir('D:/CLISAU/RAW/DO/OBITOS_ATUALIZADO/processed')
#content/drive/Shared drives/Clima&Saúde/Dados/Dados_Saude/Obitos_SSC/data/processed/2001-2018_obitos_clima_diario.csv
#Deaths and environment
df = pd.read_csv('2001-2018_obitos_clima_diario.csv')
df['DATE'] = pd.to_datetime(df['DATE'],dayfirst=True)
df = df.set_index('DATE')

#================================================
#GRID 03
#================================================
variables = ['MP10_MEAN','TMIN_IAC']

#================================================
# VARIABLE SELECTION
#================================================
# Variables
endog = df.loc[:]['all']
exog = df.loc[:][variables]
endog = endog.asfreq('D')
exog = exog.asfreq('D')
#nobs = endog.shape[0]

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))

#fill exogenous nan with mean (for each exogenous variable)
exog = exog.fillna(exog.mean())

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))
exog.head(2)

#================================================
# Train/Test split 80-20
#================================================
n_train = int(0.8*len(endog))

endog_train = endog.iloc[:n_train].asfreq('D')
endog_test = endog.iloc[n_train:].asfreq('D')

exog_train = exog.iloc[:n_train].asfreq('D')
exog_test = exog.iloc[n_train:].asfreq('D')

lags = list(range(1,16))
for n_hours in lags:
    LAGS.append(n_hours)
    VARS.append(variables)
    #================================================
    # Normalization
    #================================================
    train = pd.merge(endog_train,exog_train,on='DATE')
    test = pd.merge(endog_test,exog_test,on='DATE')
    test.head(3)
    train.head(3)
    
    '''
    if exog does contain endog, mode = 1
    if exog doesn't contain endog, mode = 0
    '''
    mode = 0
    
    if mode == 1:
      flag = 0
    else:
      flag = 1
    
    # load dataset
    values = train.values
    # ensure all data is float
    np.set_printoptions(suppress=True) #supress scientific notation
    values = values.astype('float32')
    
    # normalize train features
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(values)
    # normalize test based on train scaler
    test_scaled = scaler.transform(test.values)
    # specify the number of lag (hours in this case, since each line equals a hour)
    #n_features is the number of the exogenous variables
    
    n_features = len(exog.columns)+mode
    
    #================================================
    # Supervised Learning
    #================================================
    # frame as supervised learning
    
    #Here I used scaled[:,1:] because I exlcuded the y (deaths all) variable from the function below.
    reframed = series_to_supervised(train_scaled[:,flag:], n_hours, 0)
    reframed_test = series_to_supervised(test_scaled[:,flag:], n_hours, 0)
    print(reframed.shape) #nº observation x ((nºfeatures + 1) * nºlags)
    
    reframed.head()
    
    #================================================
    # Reframed as 3d data
    #================================================
    # split into train and test sets
    values = reframed.values
    #Train data = 1 year of data
    '''
    To speed up the training of the model for this demonstration, we will only fit 
    the model on the first year of data (365 days * 24hours (1 day)), then evaluate it on the remaining 4 years 
    of data. If you have time, consider exploring the inverted version of 
    this test harness.
    '''
    #n_train_hours = 365 * 24
    #train = values[:n_train_hours, :]
    #test = values[n_train_hours:, :]
    
    # split into input and outputs
    
    n_obs = n_hours * n_features
    #Train and test
    
    train_X, train_y = values, train_scaled[n_hours:,0]
    test_X, test_y = reframed_test.values, test_scaled[n_hours:,0]
    print(train_X.shape, len(train_X), train_y.shape)
    # reshape input (train_X and test_X) to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
    #================================================
    # Create LSTM model for GridSearch
    #================================================
    optimizer = ['adam']
    batch_size = [300]
    epochs = [300]
    init_mode = ['glorot_uniform']
    layers = ['deep']
    callback = ['ES_30']
    param_grid = dict(batch_size=batch_size, epochs=epochs,optimizer=optimizer,init_mode=init_mode,layers=layers,callback=callback)
    
    model_id = 'deep'
    MODEL_ID.append(model_id)
    PARAMS.append(param_grid)
    # create model
    model01 = Sequential()
    model01.add(LSTM(16, return_sequences=True,input_shape=(train_X.shape[1], train_X.shape[2])))
    model01.add(LSTM(32, return_sequences=True))
    model01.add(LSTM(64, return_sequences=True))
    model01.add(LSTM(32, return_sequences=True))
    model01.add(LSTM(16))
    model01.add(Dense(1))
    
    optimizer = tf.keras.optimizers.Adam()
    model01.compile(loss='mse', optimizer=optimizer)
    
    # fit network
    from keras.callbacks import EarlyStopping
    stop = EarlyStopping(monitor='val_loss', patience=30,verbose=0,restore_best_weights=True)
    callbacks_list = [stop]
    
    history = model01.fit(train_X, train_y, epochs=300, batch_size=300, validation_data=(test_X, test_y),callbacks = callbacks_list, verbose=2, shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
     
    # make a prediction
    yhat = model01.predict(test_X)
    
    #================================================
    # Invert Scaler
    #================================================
    #Reshape test_X to get 7 random columns
    test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
    
    #gambiarra
    n_features = n_features-mode
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, -n_features:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    '''
    Q: why do we use -11 in 'test_X[:, -11:]' especially?
    A: We are only interested in inverting the target, but the transform requires 
    the same columns when inverting as when transforming. Therefore we are adding 
    the target with other input vars for the inverse operation.
    '''
    # invert scaling for actual
    
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, -n_features:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    mape = mape_vectorized(inv_y,inv_yhat)
    print('Test MAPE: %.3f' % mape)
    
    #NAIVE - MEAN VALUE: evaluate if I just predict the average of the timeseries
    naive = pd.Series(np.repeat(df['all'][n_train+n_hours:].mean(),len(df['all'][n_train+n_hours:])))
    naive.index = df['all'][n_train+n_hours:].index
    
    naive_mse = mean_squared_error(df['all'][n_train+n_hours:],naive)
    print('\nNAIVE RMSE: ',round(np.sqrt(naive_mse),3))
    naive_mape = mape_vectorized(df['all'][n_train+n_hours:],naive)
    print('NAIVE MAPE: ',round(naive_mape,2),'%')
    
    RMSE_one.append(round(rmse,3))
    MAPE_one.append(round(mape,2))
    NAIVE_rmse.append(round(np.sqrt(naive_mse),2))
    NAIVE_mape.append(round(naive_mape,2))
    
    #================================================
    # Plot Prediction
    #================================================
    # plot time series
    plt.figure(figsize=[20,3])
    #test.iloc[n_hours:].index
    plt.plot(test.iloc[n_hours:].index, inv_y[:], label='Observed',alpha=0.4,color='gray')
    plt.plot(test.iloc[n_hours:].index, inv_yhat[:], label='Predicted',color='dodgerblue')
    #plt.title('Prediction vs Observed\nLast 365 days (~1 year)')
    plt.xlabel('Time',fontsize=15)
    plt.ylabel('Deaths',fontsize=15)
    plt.legend(fontsize=15,loc='upper left')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
    
report_LSTM_deep03 = pd.DataFrame()
report_LSTM_deep03['MODEL'] = MODEL_ID
report_LSTM_deep03['PARAM'] = PARAMS
report_LSTM_deep03['VARS'] = VARS
report_LSTM_deep03['LAGS'] = LAGS
report_LSTM_deep03['RMSE_ONE'] = RMSE_one
report_LSTM_deep03['MAPE_ONE'] = MAPE_one
report_LSTM_deep03['NAIVE_RMSE'] = NAIVE_rmse
report_LSTM_deep03['NAIVE_MAPE'] = NAIVE_mape

os.chdir('C:/Users/trbella/Desktop/Thiago/Python/LSTM_GRID_LAGS/results/daily')
report_LSTM_deep03.to_excel('report_LSTM_deep03.xlsx')
    
#================================
# Clean the envinronment
#================================
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

import gc
gc.collect()
    
    
#=======================    
# PARTE 04
#=======================
    
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import os

#================================================
# FUNCTIONS
#================================================
# MAPE FUNCTION
def mape_vectorized(true, pred): 
    mask = true != 0
    return (np.fabs(true - pred)/true)[mask].mean()*100

#==================================================================
# series_to_supervised FUNCTION - author: Jason Brownlee
#================================================

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
  # dataset varaibles quantity
  n_vars = 1 if type(data) is list else data.shape[1]
  # dataframe transformation
  df = pd.DataFrame(data)
  cols, names = list(), list()

  # input sequence (t-n, ... t-1). lags before time t
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

  # forecast sequence (t, t+1, ... t+n). lags after time t
    
  for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
      names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
      names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
  # put it all together
  agg = pd.concat(cols, axis=1)
  agg.columns = names
  # drop rows with NaN values
  if dropnan:
    agg.dropna(inplace=True)
  return agg

#Variables for dataset creation with grid search results
MODEL_ID = []
PARAMS = []
VARS = []
LAGS = []
RMSE_one = []
MAPE_one = []
NAIVE_rmse = []
NAIVE_mape = []

#================================================
# LOAD DATA
#================================================
os.getcwd()
os.chdir('D:/CLISAU/RAW/DO/OBITOS_ATUALIZADO/processed')
#content/drive/Shared drives/Clima&Saúde/Dados/Dados_Saude/Obitos_SSC/data/processed/2001-2018_obitos_clima_diario.csv
#Deaths and environment
df = pd.read_csv('2001-2018_obitos_clima_diario.csv')
df['DATE'] = pd.to_datetime(df['DATE'],dayfirst=True)
df = df.set_index('DATE')

#================================================
#GRID 04
#================================================
variables = ['MP10_MEAN','CO_MEAN']

#================================================
# VARIABLE SELECTION
#================================================
# Variables
endog = df.loc[:]['all']
exog = df.loc[:][variables]
endog = endog.asfreq('D')
exog = exog.asfreq('D')
#nobs = endog.shape[0]

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))

#fill exogenous nan with mean (for each exogenous variable)
exog = exog.fillna(exog.mean())

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))
exog.head(2)

#================================================
# Train/Test split 80-20
#================================================
n_train = int(0.8*len(endog))

endog_train = endog.iloc[:n_train].asfreq('D')
endog_test = endog.iloc[n_train:].asfreq('D')

exog_train = exog.iloc[:n_train].asfreq('D')
exog_test = exog.iloc[n_train:].asfreq('D')

lags = list(range(1,16))
for n_hours in lags:
    LAGS.append(n_hours)
    VARS.append(variables)
    #================================================
    # Normalization
    #================================================
    train = pd.merge(endog_train,exog_train,on='DATE')
    test = pd.merge(endog_test,exog_test,on='DATE')
    test.head(3)
    train.head(3)
    
    '''
    if exog does contain endog, mode = 1
    if exog doesn't contain endog, mode = 0
    '''
    mode = 0
    
    if mode == 1:
      flag = 0
    else:
      flag = 1
    
    # load dataset
    values = train.values
    # ensure all data is float
    np.set_printoptions(suppress=True) #supress scientific notation
    values = values.astype('float32')
    
    # normalize train features
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(values)
    # normalize test based on train scaler
    test_scaled = scaler.transform(test.values)
    # specify the number of lag (hours in this case, since each line equals a hour)
    #n_features is the number of the exogenous variables
    
    n_features = len(exog.columns)+mode
    
    #================================================
    # Supervised Learning
    #================================================
    # frame as supervised learning
    
    #Here I used scaled[:,1:] because I exlcuded the y (deaths all) variable from the function below.
    reframed = series_to_supervised(train_scaled[:,flag:], n_hours, 0)
    reframed_test = series_to_supervised(test_scaled[:,flag:], n_hours, 0)
    print(reframed.shape) #nº observation x ((nºfeatures + 1) * nºlags)
    
    reframed.head()
    
    #================================================
    # Reframed as 3d data
    #================================================
    # split into train and test sets
    values = reframed.values
    #Train data = 1 year of data
    '''
    To speed up the training of the model for this demonstration, we will only fit 
    the model on the first year of data (365 days * 24hours (1 day)), then evaluate it on the remaining 4 years 
    of data. If you have time, consider exploring the inverted version of 
    this test harness.
    '''
    #n_train_hours = 365 * 24
    #train = values[:n_train_hours, :]
    #test = values[n_train_hours:, :]
    
    # split into input and outputs
    
    n_obs = n_hours * n_features
    #Train and test
    
    train_X, train_y = values, train_scaled[n_hours:,0]
    test_X, test_y = reframed_test.values, test_scaled[n_hours:,0]
    print(train_X.shape, len(train_X), train_y.shape)
    # reshape input (train_X and test_X) to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
    #================================================
    # Create LSTM model for GridSearch
    #================================================
    optimizer = ['adam']
    batch_size = [300]
    epochs = [300]
    init_mode = ['glorot_uniform']
    layers = ['deep']
    callback = ['ES_30']
    param_grid = dict(batch_size=batch_size, epochs=epochs,optimizer=optimizer,init_mode=init_mode,layers=layers,callback=callback)
    
    model_id = 'deep'
    MODEL_ID.append(model_id)
    PARAMS.append(param_grid)
    # create model
    model01 = Sequential()
    model01.add(LSTM(16, return_sequences=True,input_shape=(train_X.shape[1], train_X.shape[2])))
    model01.add(LSTM(32, return_sequences=True))
    model01.add(LSTM(64, return_sequences=True))
    model01.add(LSTM(32, return_sequences=True))
    model01.add(LSTM(16))
    model01.add(Dense(1))
    
    optimizer = tf.keras.optimizers.Adam()
    model01.compile(loss='mse', optimizer=optimizer)
    
    # fit network
    from keras.callbacks import EarlyStopping
    stop = EarlyStopping(monitor='val_loss', patience=30,verbose=0,restore_best_weights=True)
    callbacks_list = [stop]
    
    history = model01.fit(train_X, train_y, epochs=300, batch_size=300, validation_data=(test_X, test_y),callbacks = callbacks_list, verbose=2, shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
     
    # make a prediction
    yhat = model01.predict(test_X)
    
    #================================================
    # Invert Scaler
    #================================================
    #Reshape test_X to get 7 random columns
    test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
    
    #gambiarra
    n_features = n_features-mode
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, -n_features:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    '''
    Q: why do we use -11 in 'test_X[:, -11:]' especially?
    A: We are only interested in inverting the target, but the transform requires 
    the same columns when inverting as when transforming. Therefore we are adding 
    the target with other input vars for the inverse operation.
    '''
    # invert scaling for actual
    
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, -n_features:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    mape = mape_vectorized(inv_y,inv_yhat)
    print('Test MAPE: %.3f' % mape)
    
    #NAIVE - MEAN VALUE: evaluate if I just predict the average of the timeseries
    naive = pd.Series(np.repeat(df['all'][n_train+n_hours:].mean(),len(df['all'][n_train+n_hours:])))
    naive.index = df['all'][n_train+n_hours:].index
    
    naive_mse = mean_squared_error(df['all'][n_train+n_hours:],naive)
    print('\nNAIVE RMSE: ',round(np.sqrt(naive_mse),3))
    naive_mape = mape_vectorized(df['all'][n_train+n_hours:],naive)
    print('NAIVE MAPE: ',round(naive_mape,2),'%')
    
    RMSE_one.append(round(rmse,3))
    MAPE_one.append(round(mape,2))
    NAIVE_rmse.append(round(np.sqrt(naive_mse),2))
    NAIVE_mape.append(round(naive_mape,2))
    
    #================================================
    # Plot Prediction
    #================================================
    # plot time series
    plt.figure(figsize=[20,3])
    #test.iloc[n_hours:].index
    plt.plot(test.iloc[n_hours:].index, inv_y[:], label='Observed',alpha=0.4,color='gray')
    plt.plot(test.iloc[n_hours:].index, inv_yhat[:], label='Predicted',color='dodgerblue')
    #plt.title('Prediction vs Observed\nLast 365 days (~1 year)')
    plt.xlabel('Time',fontsize=15)
    plt.ylabel('Deaths',fontsize=15)
    plt.legend(fontsize=15,loc='upper left')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
    
report_LSTM_deep04 = pd.DataFrame()
report_LSTM_deep04['MODEL'] = MODEL_ID
report_LSTM_deep04['PARAM'] = PARAMS
report_LSTM_deep04['VARS'] = VARS
report_LSTM_deep04['LAGS'] = LAGS
report_LSTM_deep04['RMSE_ONE'] = RMSE_one
report_LSTM_deep04['MAPE_ONE'] = MAPE_one
report_LSTM_deep04['NAIVE_RMSE'] = NAIVE_rmse
report_LSTM_deep04['NAIVE_MAPE'] = NAIVE_mape

os.chdir('C:/Users/trbella/Desktop/Thiago/Python/LSTM_GRID_LAGS/results/daily')
report_LSTM_deep04.to_excel('report_LSTM_deep04.xlsx')
    
#================================
# Clean the envinronment
#================================
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

import gc
gc.collect() 
    

#=================
# PARTE 05
#=================
   
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import os

#================================================
# FUNCTIONS
#================================================
# MAPE FUNCTION
def mape_vectorized(true, pred): 
    mask = true != 0
    return (np.fabs(true - pred)/true)[mask].mean()*100

#==================================================================
# series_to_supervised FUNCTION - author: Jason Brownlee
#================================================

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
  # dataset varaibles quantity
  n_vars = 1 if type(data) is list else data.shape[1]
  # dataframe transformation
  df = pd.DataFrame(data)
  cols, names = list(), list()

  # input sequence (t-n, ... t-1). lags before time t
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

  # forecast sequence (t, t+1, ... t+n). lags after time t
    
  for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
      names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
      names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
  # put it all together
  agg = pd.concat(cols, axis=1)
  agg.columns = names
  # drop rows with NaN values
  if dropnan:
    agg.dropna(inplace=True)
  return agg

#Variables for dataset creation with grid search results
MODEL_ID = []
PARAMS = []
VARS = []
LAGS = []
RMSE_one = []
MAPE_one = []
NAIVE_rmse = []
NAIVE_mape = []

#================================================
# LOAD DATA
#================================================
os.getcwd()
os.chdir('D:/CLISAU/RAW/DO/OBITOS_ATUALIZADO/processed')
#content/drive/Shared drives/Clima&Saúde/Dados/Dados_Saude/Obitos_SSC/data/processed/2001-2018_obitos_clima_diario.csv
#Deaths and environment
df = pd.read_csv('2001-2018_obitos_clima_diario.csv')
df['DATE'] = pd.to_datetime(df['DATE'],dayfirst=True)
df = df.set_index('DATE')

#================================================
#GRID 05
#================================================
variables = ['TMIN_IAC']

#================================================
# VARIABLE SELECTION
#================================================
# Variables
endog = df.loc[:]['all']
exog = df.loc[:][variables]
endog = endog.asfreq('D')
exog = exog.asfreq('D')
#nobs = endog.shape[0]

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))

#fill exogenous nan with mean (for each exogenous variable)
exog = exog.fillna(exog.mean())

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))
exog.head(2)

#================================================
# Train/Test split 80-20
#================================================
n_train = int(0.8*len(endog))

endog_train = endog.iloc[:n_train].asfreq('D')
endog_test = endog.iloc[n_train:].asfreq('D')

exog_train = exog.iloc[:n_train].asfreq('D')
exog_test = exog.iloc[n_train:].asfreq('D')

lags = list(range(1,16))
for n_hours in lags:
    LAGS.append(n_hours)
    VARS.append(variables)
    #================================================
    # Normalization
    #================================================
    train = pd.merge(endog_train,exog_train,on='DATE')
    test = pd.merge(endog_test,exog_test,on='DATE')
    test.head(3)
    train.head(3)
    
    '''
    if exog does contain endog, mode = 1
    if exog doesn't contain endog, mode = 0
    '''
    mode = 0
    
    if mode == 1:
      flag = 0
    else:
      flag = 1
    
    # load dataset
    values = train.values
    # ensure all data is float
    np.set_printoptions(suppress=True) #supress scientific notation
    values = values.astype('float32')
    
    # normalize train features
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(values)
    # normalize test based on train scaler
    test_scaled = scaler.transform(test.values)
    # specify the number of lag (hours in this case, since each line equals a hour)
    #n_features is the number of the exogenous variables
    
    n_features = len(exog.columns)+mode
    
    #================================================
    # Supervised Learning
    #================================================
    # frame as supervised learning
    
    #Here I used scaled[:,1:] because I exlcuded the y (deaths all) variable from the function below.
    reframed = series_to_supervised(train_scaled[:,flag:], n_hours, 0)
    reframed_test = series_to_supervised(test_scaled[:,flag:], n_hours, 0)
    print(reframed.shape) #nº observation x ((nºfeatures + 1) * nºlags)
    
    reframed.head()
    
    #================================================
    # Reframed as 3d data
    #================================================
    # split into train and test sets
    values = reframed.values
    #Train data = 1 year of data
    '''
    To speed up the training of the model for this demonstration, we will only fit 
    the model on the first year of data (365 days * 24hours (1 day)), then evaluate it on the remaining 4 years 
    of data. If you have time, consider exploring the inverted version of 
    this test harness.
    '''
    #n_train_hours = 365 * 24
    #train = values[:n_train_hours, :]
    #test = values[n_train_hours:, :]
    
    # split into input and outputs
    
    n_obs = n_hours * n_features
    #Train and test
    
    train_X, train_y = values, train_scaled[n_hours:,0]
    test_X, test_y = reframed_test.values, test_scaled[n_hours:,0]
    print(train_X.shape, len(train_X), train_y.shape)
    # reshape input (train_X and test_X) to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
    #================================================
    # Create LSTM model for GridSearch
    #================================================
    optimizer = ['adam']
    batch_size = [300]
    epochs = [300]
    init_mode = ['glorot_uniform']
    layers = ['deep']
    callback = ['ES_30']
    param_grid = dict(batch_size=batch_size, epochs=epochs,optimizer=optimizer,init_mode=init_mode,layers=layers,callback=callback)
    
    model_id = 'deep'
    MODEL_ID.append(model_id)
    PARAMS.append(param_grid)
    # create model
    model01 = Sequential()
    model01.add(LSTM(16, return_sequences=True,input_shape=(train_X.shape[1], train_X.shape[2])))
    model01.add(LSTM(32, return_sequences=True))
    model01.add(LSTM(64, return_sequences=True))
    model01.add(LSTM(32, return_sequences=True))
    model01.add(LSTM(16))
    model01.add(Dense(1))
    
    optimizer = tf.keras.optimizers.Adam()
    model01.compile(loss='mse', optimizer=optimizer)
    
    # fit network
    from keras.callbacks import EarlyStopping
    stop = EarlyStopping(monitor='val_loss', patience=30,verbose=0,restore_best_weights=True)
    callbacks_list = [stop]
    
    history = model01.fit(train_X, train_y, epochs=300, batch_size=300, validation_data=(test_X, test_y),callbacks = callbacks_list, verbose=2, shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
     
    # make a prediction
    yhat = model01.predict(test_X)
    
    #================================================
    # Invert Scaler
    #================================================
    #Reshape test_X to get 7 random columns
    test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
    
    #gambiarra
    n_features = n_features-mode
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, -n_features:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    '''
    Q: why do we use -11 in 'test_X[:, -11:]' especially?
    A: We are only interested in inverting the target, but the transform requires 
    the same columns when inverting as when transforming. Therefore we are adding 
    the target with other input vars for the inverse operation.
    '''
    # invert scaling for actual
    
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, -n_features:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    mape = mape_vectorized(inv_y,inv_yhat)
    print('Test MAPE: %.3f' % mape)
    
    #NAIVE - MEAN VALUE: evaluate if I just predict the average of the timeseries
    naive = pd.Series(np.repeat(df['all'][n_train+n_hours:].mean(),len(df['all'][n_train+n_hours:])))
    naive.index = df['all'][n_train+n_hours:].index
    
    naive_mse = mean_squared_error(df['all'][n_train+n_hours:],naive)
    print('\nNAIVE RMSE: ',round(np.sqrt(naive_mse),3))
    naive_mape = mape_vectorized(df['all'][n_train+n_hours:],naive)
    print('NAIVE MAPE: ',round(naive_mape,2),'%')
    
    RMSE_one.append(round(rmse,3))
    MAPE_one.append(round(mape,2))
    NAIVE_rmse.append(round(np.sqrt(naive_mse),2))
    NAIVE_mape.append(round(naive_mape,2))
    
    #================================================
    # Plot Prediction
    #================================================
    # plot time series
    plt.figure(figsize=[20,3])
    #test.iloc[n_hours:].index
    plt.plot(test.iloc[n_hours:].index, inv_y[:], label='Observed',alpha=0.4,color='gray')
    plt.plot(test.iloc[n_hours:].index, inv_yhat[:], label='Predicted',color='dodgerblue')
    #plt.title('Prediction vs Observed\nLast 365 days (~1 year)')
    plt.xlabel('Time',fontsize=15)
    plt.ylabel('Deaths',fontsize=15)
    plt.legend(fontsize=15,loc='upper left')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
    
report_LSTM_deep05 = pd.DataFrame()
report_LSTM_deep05['MODEL'] = MODEL_ID
report_LSTM_deep05['PARAM'] = PARAMS
report_LSTM_deep05['VARS'] = VARS
report_LSTM_deep05['LAGS'] = LAGS
report_LSTM_deep05['RMSE_ONE'] = RMSE_one
report_LSTM_deep05['MAPE_ONE'] = MAPE_one
report_LSTM_deep05['NAIVE_RMSE'] = NAIVE_rmse
report_LSTM_deep05['NAIVE_MAPE'] = NAIVE_mape

os.chdir('C:/Users/trbella/Desktop/Thiago/Python/LSTM_GRID_LAGS/results/daily')
report_LSTM_deep05.to_excel('report_LSTM_deep05.xlsx')
    
#================================
# Clean the envinronment
#================================
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

import gc
gc.collect() 
   

#====================
# PARTE 06
#====================
   
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import os

#================================================
# FUNCTIONS
#================================================
# MAPE FUNCTION
def mape_vectorized(true, pred): 
    mask = true != 0
    return (np.fabs(true - pred)/true)[mask].mean()*100

#==================================================================
# series_to_supervised FUNCTION - author: Jason Brownlee
#================================================

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
  # dataset varaibles quantity
  n_vars = 1 if type(data) is list else data.shape[1]
  # dataframe transformation
  df = pd.DataFrame(data)
  cols, names = list(), list()

  # input sequence (t-n, ... t-1). lags before time t
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

  # forecast sequence (t, t+1, ... t+n). lags after time t
    
  for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
      names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
      names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
  # put it all together
  agg = pd.concat(cols, axis=1)
  agg.columns = names
  # drop rows with NaN values
  if dropnan:
    agg.dropna(inplace=True)
  return agg

#Variables for dataset creation with grid search results
MODEL_ID = []
PARAMS = []
VARS = []
LAGS = []
RMSE_one = []
MAPE_one = []
NAIVE_rmse = []
NAIVE_mape = []

#================================================
# LOAD DATA
#================================================
os.getcwd()
os.chdir('D:/CLISAU/RAW/DO/OBITOS_ATUALIZADO/processed')
#content/drive/Shared drives/Clima&Saúde/Dados/Dados_Saude/Obitos_SSC/data/processed/2001-2018_obitos_clima_diario.csv
#Deaths and environment
df = pd.read_csv('2001-2018_obitos_clima_diario.csv')
df['DATE'] = pd.to_datetime(df['DATE'],dayfirst=True)
df = df.set_index('DATE')

#================================================
#GRID 06
#================================================
variables = ['MP10_MEAN']

#================================================
# VARIABLE SELECTION
#================================================
# Variables
endog = df.loc[:]['all']
exog = df.loc[:][variables]
endog = endog.asfreq('D')
exog = exog.asfreq('D')
#nobs = endog.shape[0]

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))

#fill exogenous nan with mean (for each exogenous variable)
exog = exog.fillna(exog.mean())

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))
exog.head(2)

#================================================
# Train/Test split 80-20
#================================================
n_train = int(0.8*len(endog))

endog_train = endog.iloc[:n_train].asfreq('D')
endog_test = endog.iloc[n_train:].asfreq('D')

exog_train = exog.iloc[:n_train].asfreq('D')
exog_test = exog.iloc[n_train:].asfreq('D')

lags = list(range(1,16))
for n_hours in lags:
    LAGS.append(n_hours)
    VARS.append(variables)
    #================================================
    # Normalization
    #================================================
    train = pd.merge(endog_train,exog_train,on='DATE')
    test = pd.merge(endog_test,exog_test,on='DATE')
    test.head(3)
    train.head(3)
    
    '''
    if exog does contain endog, mode = 1
    if exog doesn't contain endog, mode = 0
    '''
    mode = 0
    
    if mode == 1:
      flag = 0
    else:
      flag = 1
    
    # load dataset
    values = train.values
    # ensure all data is float
    np.set_printoptions(suppress=True) #supress scientific notation
    values = values.astype('float32')
    
    # normalize train features
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(values)
    # normalize test based on train scaler
    test_scaled = scaler.transform(test.values)
    # specify the number of lag (hours in this case, since each line equals a hour)
    #n_features is the number of the exogenous variables
    
    n_features = len(exog.columns)+mode
    
    #================================================
    # Supervised Learning
    #================================================
    # frame as supervised learning
    
    #Here I used scaled[:,1:] because I exlcuded the y (deaths all) variable from the function below.
    reframed = series_to_supervised(train_scaled[:,flag:], n_hours, 0)
    reframed_test = series_to_supervised(test_scaled[:,flag:], n_hours, 0)
    print(reframed.shape) #nº observation x ((nºfeatures + 1) * nºlags)
    
    reframed.head()
    
    #================================================
    # Reframed as 3d data
    #================================================
    # split into train and test sets
    values = reframed.values
    #Train data = 1 year of data
    '''
    To speed up the training of the model for this demonstration, we will only fit 
    the model on the first year of data (365 days * 24hours (1 day)), then evaluate it on the remaining 4 years 
    of data. If you have time, consider exploring the inverted version of 
    this test harness.
    '''
    #n_train_hours = 365 * 24
    #train = values[:n_train_hours, :]
    #test = values[n_train_hours:, :]
    
    # split into input and outputs
    
    n_obs = n_hours * n_features
    #Train and test
    
    train_X, train_y = values, train_scaled[n_hours:,0]
    test_X, test_y = reframed_test.values, test_scaled[n_hours:,0]
    print(train_X.shape, len(train_X), train_y.shape)
    # reshape input (train_X and test_X) to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
    #================================================
    # Create LSTM model for GridSearch
    #================================================
    optimizer = ['adam']
    batch_size = [300]
    epochs = [300]
    init_mode = ['glorot_uniform']
    layers = ['deep']
    callback = ['ES_30']
    param_grid = dict(batch_size=batch_size, epochs=epochs,optimizer=optimizer,init_mode=init_mode,layers=layers,callback=callback)
    
    model_id = 'deep'
    MODEL_ID.append(model_id)
    PARAMS.append(param_grid)
    # create model
    model01 = Sequential()
    model01.add(LSTM(16, return_sequences=True,input_shape=(train_X.shape[1], train_X.shape[2])))
    model01.add(LSTM(32, return_sequences=True))
    model01.add(LSTM(64, return_sequences=True))
    model01.add(LSTM(32, return_sequences=True))
    model01.add(LSTM(16))
    model01.add(Dense(1))
    
    optimizer = tf.keras.optimizers.Adam()
    model01.compile(loss='mse', optimizer=optimizer)
    
    # fit network
    from keras.callbacks import EarlyStopping
    stop = EarlyStopping(monitor='val_loss', patience=30,verbose=0,restore_best_weights=True)
    callbacks_list = [stop]
    
    history = model01.fit(train_X, train_y, epochs=300, batch_size=300, validation_data=(test_X, test_y),callbacks = callbacks_list, verbose=2, shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
     
    # make a prediction
    yhat = model01.predict(test_X)
    
    #================================================
    # Invert Scaler
    #================================================
    #Reshape test_X to get 7 random columns
    test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
    
    #gambiarra
    n_features = n_features-mode
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, -n_features:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    '''
    Q: why do we use -11 in 'test_X[:, -11:]' especially?
    A: We are only interested in inverting the target, but the transform requires 
    the same columns when inverting as when transforming. Therefore we are adding 
    the target with other input vars for the inverse operation.
    '''
    # invert scaling for actual
    
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, -n_features:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    mape = mape_vectorized(inv_y,inv_yhat)
    print('Test MAPE: %.3f' % mape)
    
    #NAIVE - MEAN VALUE: evaluate if I just predict the average of the timeseries
    naive = pd.Series(np.repeat(df['all'][n_train+n_hours:].mean(),len(df['all'][n_train+n_hours:])))
    naive.index = df['all'][n_train+n_hours:].index
    
    naive_mse = mean_squared_error(df['all'][n_train+n_hours:],naive)
    print('\nNAIVE RMSE: ',round(np.sqrt(naive_mse),3))
    naive_mape = mape_vectorized(df['all'][n_train+n_hours:],naive)
    print('NAIVE MAPE: ',round(naive_mape,2),'%')
    
    RMSE_one.append(round(rmse,3))
    MAPE_one.append(round(mape,2))
    NAIVE_rmse.append(round(np.sqrt(naive_mse),2))
    NAIVE_mape.append(round(naive_mape,2))
    
    #================================================
    # Plot Prediction
    #================================================
    # plot time series
    plt.figure(figsize=[20,3])
    #test.iloc[n_hours:].index
    plt.plot(test.iloc[n_hours:].index, inv_y[:], label='Observed',alpha=0.4,color='gray')
    plt.plot(test.iloc[n_hours:].index, inv_yhat[:], label='Predicted',color='dodgerblue')
    #plt.title('Prediction vs Observed\nLast 365 days (~1 year)')
    plt.xlabel('Time',fontsize=15)
    plt.ylabel('Deaths',fontsize=15)
    plt.legend(fontsize=15,loc='upper left')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
    
report_LSTM_deep06 = pd.DataFrame()
report_LSTM_deep06['MODEL'] = MODEL_ID
report_LSTM_deep06['PARAM'] = PARAMS
report_LSTM_deep06['VARS'] = VARS
report_LSTM_deep06['LAGS'] = LAGS
report_LSTM_deep06['RMSE_ONE'] = RMSE_one
report_LSTM_deep06['MAPE_ONE'] = MAPE_one
report_LSTM_deep06['NAIVE_RMSE'] = NAIVE_rmse
report_LSTM_deep06['NAIVE_MAPE'] = NAIVE_mape

os.chdir('C:/Users/trbella/Desktop/Thiago/Python/LSTM_GRID_LAGS/results/daily')
report_LSTM_deep06.to_excel('report_LSTM_deep06.xlsx')
    
#================================
# Clean the envinronment
#================================
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

import gc
gc.collect() 
    
#=================
# PARTE 07
#=================
   
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import os

#================================================
# FUNCTIONS
#================================================
# MAPE FUNCTION
def mape_vectorized(true, pred): 
    mask = true != 0
    return (np.fabs(true - pred)/true)[mask].mean()*100

#==================================================================
# series_to_supervised FUNCTION - author: Jason Brownlee
#================================================

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
  # dataset varaibles quantity
  n_vars = 1 if type(data) is list else data.shape[1]
  # dataframe transformation
  df = pd.DataFrame(data)
  cols, names = list(), list()

  # input sequence (t-n, ... t-1). lags before time t
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

  # forecast sequence (t, t+1, ... t+n). lags after time t
    
  for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
      names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
      names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
  # put it all together
  agg = pd.concat(cols, axis=1)
  agg.columns = names
  # drop rows with NaN values
  if dropnan:
    agg.dropna(inplace=True)
  return agg

#Variables for dataset creation with grid search results
MODEL_ID = []
PARAMS = []
VARS = []
LAGS = []
RMSE_one = []
MAPE_one = []
NAIVE_rmse = []
NAIVE_mape = []

#================================================
# LOAD DATA
#================================================
os.getcwd()
os.chdir('D:/CLISAU/RAW/DO/OBITOS_ATUALIZADO/processed')
#content/drive/Shared drives/Clima&Saúde/Dados/Dados_Saude/Obitos_SSC/data/processed/2001-2018_obitos_clima_diario.csv
#Deaths and environment
df = pd.read_csv('2001-2018_obitos_clima_diario.csv')
df['DATE'] = pd.to_datetime(df['DATE'],dayfirst=True)
df = df.set_index('DATE')

#================================================
#GRID 06
#================================================
variables = ['CO_MEAN']

#================================================
# VARIABLE SELECTION
#================================================
# Variables
endog = df.loc[:]['all']
exog = df.loc[:][variables]
endog = endog.asfreq('D')
exog = exog.asfreq('D')
#nobs = endog.shape[0]

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))

#fill exogenous nan with mean (for each exogenous variable)
exog = exog.fillna(exog.mean())

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))
exog.head(2)

#================================================
# Train/Test split 80-20
#================================================
n_train = int(0.8*len(endog))

endog_train = endog.iloc[:n_train].asfreq('D')
endog_test = endog.iloc[n_train:].asfreq('D')

exog_train = exog.iloc[:n_train].asfreq('D')
exog_test = exog.iloc[n_train:].asfreq('D')

lags = list(range(1,16))
for n_hours in lags:
    LAGS.append(n_hours)
    VARS.append(variables)
    #================================================
    # Normalization
    #================================================
    train = pd.merge(endog_train,exog_train,on='DATE')
    test = pd.merge(endog_test,exog_test,on='DATE')
    test.head(3)
    train.head(3)
    
    '''
    if exog does contain endog, mode = 1
    if exog doesn't contain endog, mode = 0
    '''
    mode = 0
    
    if mode == 1:
      flag = 0
    else:
      flag = 1
    
    # load dataset
    values = train.values
    # ensure all data is float
    np.set_printoptions(suppress=True) #supress scientific notation
    values = values.astype('float32')
    
    # normalize train features
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(values)
    # normalize test based on train scaler
    test_scaled = scaler.transform(test.values)
    # specify the number of lag (hours in this case, since each line equals a hour)
    #n_features is the number of the exogenous variables
    
    n_features = len(exog.columns)+mode
    
    #================================================
    # Supervised Learning
    #================================================
    # frame as supervised learning
    
    #Here I used scaled[:,1:] because I exlcuded the y (deaths all) variable from the function below.
    reframed = series_to_supervised(train_scaled[:,flag:], n_hours, 0)
    reframed_test = series_to_supervised(test_scaled[:,flag:], n_hours, 0)
    print(reframed.shape) #nº observation x ((nºfeatures + 1) * nºlags)
    
    reframed.head()
    
    #================================================
    # Reframed as 3d data
    #================================================
    # split into train and test sets
    values = reframed.values
    #Train data = 1 year of data
    '''
    To speed up the training of the model for this demonstration, we will only fit 
    the model on the first year of data (365 days * 24hours (1 day)), then evaluate it on the remaining 4 years 
    of data. If you have time, consider exploring the inverted version of 
    this test harness.
    '''
    #n_train_hours = 365 * 24
    #train = values[:n_train_hours, :]
    #test = values[n_train_hours:, :]
    
    # split into input and outputs
    
    n_obs = n_hours * n_features
    #Train and test
    
    train_X, train_y = values, train_scaled[n_hours:,0]
    test_X, test_y = reframed_test.values, test_scaled[n_hours:,0]
    print(train_X.shape, len(train_X), train_y.shape)
    # reshape input (train_X and test_X) to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
    #================================================
    # Create LSTM model for GridSearch
    #================================================
    optimizer = ['adam']
    batch_size = [300]
    epochs = [300]
    init_mode = ['glorot_uniform']
    layers = ['deep']
    callback = ['ES_30']
    param_grid = dict(batch_size=batch_size, epochs=epochs,optimizer=optimizer,init_mode=init_mode,layers=layers,callback=callback)
    
    model_id = 'deep'
    MODEL_ID.append(model_id)
    PARAMS.append(param_grid)
    # create model
    model01 = Sequential()
    model01.add(LSTM(16, return_sequences=True,input_shape=(train_X.shape[1], train_X.shape[2])))
    model01.add(LSTM(32, return_sequences=True))
    model01.add(LSTM(64, return_sequences=True))
    model01.add(LSTM(32, return_sequences=True))
    model01.add(LSTM(16))
    model01.add(Dense(1))
    
    optimizer = tf.keras.optimizers.Adam()
    model01.compile(loss='mse', optimizer=optimizer)
    
    # fit network
    from keras.callbacks import EarlyStopping
    stop = EarlyStopping(monitor='val_loss', patience=30,verbose=0,restore_best_weights=True)
    callbacks_list = [stop]
    
    history = model01.fit(train_X, train_y, epochs=300, batch_size=300, validation_data=(test_X, test_y),callbacks = callbacks_list, verbose=2, shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
     
    # make a prediction
    yhat = model01.predict(test_X)
    
    #================================================
    # Invert Scaler
    #================================================
    #Reshape test_X to get 7 random columns
    test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
    
    #gambiarra
    n_features = n_features-mode
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, -n_features:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    '''
    Q: why do we use -11 in 'test_X[:, -11:]' especially?
    A: We are only interested in inverting the target, but the transform requires 
    the same columns when inverting as when transforming. Therefore we are adding 
    the target with other input vars for the inverse operation.
    '''
    # invert scaling for actual
    
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, -n_features:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    mape = mape_vectorized(inv_y,inv_yhat)
    print('Test MAPE: %.3f' % mape)
    
    #NAIVE - MEAN VALUE: evaluate if I just predict the average of the timeseries
    naive = pd.Series(np.repeat(df['all'][n_train+n_hours:].mean(),len(df['all'][n_train+n_hours:])))
    naive.index = df['all'][n_train+n_hours:].index
    
    naive_mse = mean_squared_error(df['all'][n_train+n_hours:],naive)
    print('\nNAIVE RMSE: ',round(np.sqrt(naive_mse),3))
    naive_mape = mape_vectorized(df['all'][n_train+n_hours:],naive)
    print('NAIVE MAPE: ',round(naive_mape,2),'%')
    
    RMSE_one.append(round(rmse,3))
    MAPE_one.append(round(mape,2))
    NAIVE_rmse.append(round(np.sqrt(naive_mse),2))
    NAIVE_mape.append(round(naive_mape,2))
    
    #================================================
    # Plot Prediction
    #================================================
    # plot time series
    plt.figure(figsize=[20,3])
    #test.iloc[n_hours:].index
    plt.plot(test.iloc[n_hours:].index, inv_y[:], label='Observed',alpha=0.4,color='gray')
    plt.plot(test.iloc[n_hours:].index, inv_yhat[:], label='Predicted',color='dodgerblue')
    #plt.title('Prediction vs Observed\nLast 365 days (~1 year)')
    plt.xlabel('Time',fontsize=15)
    plt.ylabel('Deaths',fontsize=15)
    plt.legend(fontsize=15,loc='upper left')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
    
report_LSTM_deep07 = pd.DataFrame()
report_LSTM_deep07['MODEL'] = MODEL_ID
report_LSTM_deep07['PARAM'] = PARAMS
report_LSTM_deep07['VARS'] = VARS
report_LSTM_deep07['LAGS'] = LAGS
report_LSTM_deep07['RMSE_ONE'] = RMSE_one
report_LSTM_deep07['MAPE_ONE'] = MAPE_one
report_LSTM_deep07['NAIVE_RMSE'] = NAIVE_rmse
report_LSTM_deep07['NAIVE_MAPE'] = NAIVE_mape

os.chdir('C:/Users/trbella/Desktop/Thiago/Python/LSTM_GRID_LAGS/results/daily')
report_LSTM_deep07.to_excel('report_LSTM_deep07.xlsx')
    
#================================
# Clean the envinronment
#================================
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

import gc
gc.collect() 
    

#============================================
#          SHALLOW NETWORKS
#============================================

# -*- coding: utf-8 -*-
"""
Created on Sat May 15 20:34:05 2021

@author: Thiago
"""
#================================================
# PACKAGES
#================================================
#RODAR ESSA LINHA SEPARADAMENTE - #run model on CPU (It's faster in this case)

import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import os

#================================================
# FUNCTIONS
#================================================
# MAPE FUNCTION
def mape_vectorized(true, pred): 
    mask = true != 0
    return (np.fabs(true - pred)/true)[mask].mean()*100

#==================================================================
# series_to_supervised FUNCTION - author: Jason Brownlee
#================================================

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
  # dataset varaibles quantity
  n_vars = 1 if type(data) is list else data.shape[1]
  # dataframe transformation
  df = pd.DataFrame(data)
  cols, names = list(), list()

  # input sequence (t-n, ... t-1). lags before time t
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

  # forecast sequence (t, t+1, ... t+n). lags after time t
    
  for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
      names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
      names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
  # put it all together
  agg = pd.concat(cols, axis=1)
  agg.columns = names
  # drop rows with NaN values
  if dropnan:
    agg.dropna(inplace=True)
  return agg

#Variables for dataset creation with grid search results
MODEL_ID = []
PARAMS = []
VARS = []
LAGS = []
RMSE_one = []
MAPE_one = []
NAIVE_rmse = []
NAIVE_mape = []

#================================================
# LOAD DATA
#================================================
os.getcwd()
os.chdir('D:/CLISAU/RAW/DO/OBITOS_ATUALIZADO/processed')
#content/drive/Shared drives/Clima&Saúde/Dados/Dados_Saude/Obitos_SSC/data/processed/2001-2018_obitos_clima_diario.csv
#Deaths and environment
df = pd.read_csv('2001-2018_obitos_clima_diario.csv')
df['DATE'] = pd.to_datetime(df['DATE'],dayfirst=True)
df = df.set_index('DATE')

#================================================
#GRID 01
#================================================
variables = ['CO_MEAN','MP10_MEAN','TMIN_IAC']

#================================================
# VARIABLE SELECTION
#================================================
# Variables
endog = df.loc[:]['all']
exog = df.loc[:][variables]
endog = endog.asfreq('D')
exog = exog.asfreq('D')
#nobs = endog.shape[0]

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))

#fill exogenous nan with mean (for each exogenous variable)
exog = exog.fillna(exog.mean())

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))
exog.head(2)

#================================================
# Train/Test split 80-20
#================================================
n_train = int(0.8*len(endog))

endog_train = endog.iloc[:n_train].asfreq('D')
endog_test = endog.iloc[n_train:].asfreq('D')

exog_train = exog.iloc[:n_train].asfreq('D')
exog_test = exog.iloc[n_train:].asfreq('D')

lags = list(range(1,16))
for n_hours in lags:
    LAGS.append(n_hours)
    VARS.append(variables)
    #================================================
    # Normalization
    #================================================
    train = pd.merge(endog_train,exog_train,on='DATE')
    test = pd.merge(endog_test,exog_test,on='DATE')
    test.head(3)
    train.head(3)
    
    '''
    if exog does contain endog, mode = 1
    if exog doesn't contain endog, mode = 0
    '''
    mode = 0
    
    if mode == 1:
      flag = 0
    else:
      flag = 1
    
    # load dataset
    values = train.values
    # ensure all data is float
    np.set_printoptions(suppress=True) #supress scientific notation
    values = values.astype('float32')
    
    # normalize train features
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(values)
    # normalize test based on train scaler
    test_scaled = scaler.transform(test.values)
    # specify the number of lag (hours in this case, since each line equals a hour)
    #n_features is the number of the exogenous variables
    
    n_features = len(exog.columns)+mode
    
    #================================================
    # Supervised Learning
    #================================================
    # frame as supervised learning
    
    #Here I used scaled[:,1:] because I exlcuded the y (deaths all) variable from the function below.
    reframed = series_to_supervised(train_scaled[:,flag:], n_hours, 0)
    reframed_test = series_to_supervised(test_scaled[:,flag:], n_hours, 0)
    print(reframed.shape) #nº observation x ((nºfeatures + 1) * nºlags)
    
    reframed.head()
    
    #================================================
    # Reframed as 3d data
    #================================================
    # split into train and test sets
    values = reframed.values
    #Train data = 1 year of data
    '''
    To speed up the training of the model for this demonstration, we will only fit 
    the model on the first year of data (365 days * 24hours (1 day)), then evaluate it on the remaining 4 years 
    of data. If you have time, consider exploring the inverted version of 
    this test harness.
    '''
    #n_train_hours = 365 * 24
    #train = values[:n_train_hours, :]
    #test = values[n_train_hours:, :]
    
    # split into input and outputs
    
    n_obs = n_hours * n_features
    #Train and test
    
    train_X, train_y = values, train_scaled[n_hours:,0]
    test_X, test_y = reframed_test.values, test_scaled[n_hours:,0]
    print(train_X.shape, len(train_X), train_y.shape)
    # reshape input (train_X and test_X) to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
    #================================================
    # Create LSTM model for GridSearch
    #================================================
    optimizer = ['adam']
    batch_size = [300]
    epochs = [300]
    init_mode = ['glorot_uniform']
    layers = ['shallow']
    callback = ['ES_30']
    param_grid = dict(batch_size=batch_size, epochs=epochs,optimizer=optimizer,init_mode=init_mode,layers=layers,callback=callback)
    
    model_id = 'shallow'
    MODEL_ID.append(model_id)
    PARAMS.append(param_grid)
    # create model
    model01 = Sequential()
    model01.add(LSTM(3, return_sequences=False,input_shape=(train_X.shape[1], train_X.shape[2])))
    model01.add(Dense(1))
    
    optimizer = tf.keras.optimizers.Adam()
    model01.compile(loss='mse', optimizer=optimizer)
    
    # fit network
    from keras.callbacks import EarlyStopping
    stop = EarlyStopping(monitor='val_loss', patience=30,verbose=0,restore_best_weights=True)
    callbacks_list = [stop]
    
    history = model01.fit(train_X, train_y, epochs=300, batch_size=300, validation_data=(test_X, test_y),callbacks = callbacks_list, verbose=2, shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
     
    # make a prediction
    yhat = model01.predict(test_X)
    
    #================================================
    # Invert Scaler
    #================================================
    #Reshape test_X to get 7 random columns
    test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
    
    #gambiarra
    n_features = n_features-mode
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, -n_features:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    '''
    Q: why do we use -11 in 'test_X[:, -11:]' especially?
    A: We are only interested in inverting the target, but the transform requires 
    the same columns when inverting as when transforming. Therefore we are adding 
    the target with other input vars for the inverse operation.
    '''
    # invert scaling for actual
    
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, -n_features:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    mape = mape_vectorized(inv_y,inv_yhat)
    print('Test MAPE: %.3f' % mape)
    
    #NAIVE - MEAN VALUE: evaluate if I just predict the average of the timeseries
    naive = pd.Series(np.repeat(df['all'][n_train+n_hours:].mean(),len(df['all'][n_train+n_hours:])))
    naive.index = df['all'][n_train+n_hours:].index
    
    naive_mse = mean_squared_error(df['all'][n_train+n_hours:],naive)
    print('\nNAIVE RMSE: ',round(np.sqrt(naive_mse),3))
    naive_mape = mape_vectorized(df['all'][n_train+n_hours:],naive)
    print('NAIVE MAPE: ',round(naive_mape,2),'%')
    
    RMSE_one.append(round(rmse,3))
    MAPE_one.append(round(mape,2))
    NAIVE_rmse.append(round(np.sqrt(naive_mse),2))
    NAIVE_mape.append(round(naive_mape,2))
    
    #================================================
    # Plot Prediction
    #================================================
    # plot time series
    plt.figure(figsize=[20,3])
    #test.iloc[n_hours:].index
    plt.plot(test.iloc[n_hours:].index, inv_y[:], label='Observed',alpha=0.4,color='gray')
    plt.plot(test.iloc[n_hours:].index, inv_yhat[:], label='Predicted',color='dodgerblue')
    #plt.title('Prediction vs Observed\nLast 365 days (~1 year)')
    plt.xlabel('Time',fontsize=15)
    plt.ylabel('Deaths',fontsize=15)
    plt.legend(fontsize=15,loc='upper left')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
    
report_LSTM_shallow01 = pd.DataFrame()
report_LSTM_shallow01['MODEL'] = MODEL_ID
report_LSTM_shallow01['PARAM'] = PARAMS
report_LSTM_shallow01['VARS'] = VARS
report_LSTM_shallow01['LAGS'] = LAGS
report_LSTM_shallow01['RMSE_ONE'] = RMSE_one
report_LSTM_shallow01['MAPE_ONE'] = MAPE_one
report_LSTM_shallow01['NAIVE_RMSE'] = NAIVE_rmse
report_LSTM_shallow01['NAIVE_MAPE'] = NAIVE_mape

os.chdir('C:/Users/trbella/Desktop/Thiago/Python/LSTM_GRID_LAGS/results/daily')
report_LSTM_shallow01.to_excel('report_LSTM_shallow01.xlsx')
    
#================================
# Clean the envinronment
#================================
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

import gc
gc.collect()
    
    
#=========================
#  PART 02
#=========================
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import os

#================================================
# FUNCTIONS
#================================================
# MAPE FUNCTION
def mape_vectorized(true, pred): 
    mask = true != 0
    return (np.fabs(true - pred)/true)[mask].mean()*100

#==================================================================
# series_to_supervised FUNCTION - author: Jason Brownlee
#================================================

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
  # dataset varaibles quantity
  n_vars = 1 if type(data) is list else data.shape[1]
  # dataframe transformation
  df = pd.DataFrame(data)
  cols, names = list(), list()

  # input sequence (t-n, ... t-1). lags before time t
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

  # forecast sequence (t, t+1, ... t+n). lags after time t
    
  for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
      names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
      names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
  # put it all together
  agg = pd.concat(cols, axis=1)
  agg.columns = names
  # drop rows with NaN values
  if dropnan:
    agg.dropna(inplace=True)
  return agg

#Variables for dataset creation with grid search results
MODEL_ID = []
PARAMS = []
VARS = []
LAGS = []
RMSE_one = []
MAPE_one = []
NAIVE_rmse = []
NAIVE_mape = []

#================================================
# LOAD DATA
#================================================
os.getcwd()
os.chdir('D:/CLISAU/RAW/DO/OBITOS_ATUALIZADO/processed')
#content/drive/Shared drives/Clima&Saúde/Dados/Dados_Saude/Obitos_SSC/data/processed/2001-2018_obitos_clima_diario.csv
#Deaths and environment
df = pd.read_csv('2001-2018_obitos_clima_diario.csv')
df['DATE'] = pd.to_datetime(df['DATE'],dayfirst=True)
df = df.set_index('DATE')

#================================================
#GRID 02
#================================================
variables = ['CO_MEAN','TMIN_IAC']

#================================================
# VARIABLE SELECTION
#================================================
# Variables
endog = df.loc[:]['all']
exog = df.loc[:][variables]
endog = endog.asfreq('D')
exog = exog.asfreq('D')
#nobs = endog.shape[0]

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))

#fill exogenous nan with mean (for each exogenous variable)
exog = exog.fillna(exog.mean())

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))
exog.head(2)

#================================================
# Train/Test split 80-20
#================================================
n_train = int(0.8*len(endog))

endog_train = endog.iloc[:n_train].asfreq('D')
endog_test = endog.iloc[n_train:].asfreq('D')

exog_train = exog.iloc[:n_train].asfreq('D')
exog_test = exog.iloc[n_train:].asfreq('D')

lags = list(range(1,16))
for n_hours in lags:
    LAGS.append(n_hours)
    VARS.append(variables)
    #================================================
    # Normalization
    #================================================
    train = pd.merge(endog_train,exog_train,on='DATE')
    test = pd.merge(endog_test,exog_test,on='DATE')
    test.head(3)
    train.head(3)
    
    '''
    if exog does contain endog, mode = 1
    if exog doesn't contain endog, mode = 0
    '''
    mode = 0
    
    if mode == 1:
      flag = 0
    else:
      flag = 1
    
    # load dataset
    values = train.values
    # ensure all data is float
    np.set_printoptions(suppress=True) #supress scientific notation
    values = values.astype('float32')
    
    # normalize train features
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(values)
    # normalize test based on train scaler
    test_scaled = scaler.transform(test.values)
    # specify the number of lag (hours in this case, since each line equals a hour)
    #n_features is the number of the exogenous variables
    
    n_features = len(exog.columns)+mode
    
    #================================================
    # Supervised Learning
    #================================================
    # frame as supervised learning
    
    #Here I used scaled[:,1:] because I exlcuded the y (deaths all) variable from the function below.
    reframed = series_to_supervised(train_scaled[:,flag:], n_hours, 0)
    reframed_test = series_to_supervised(test_scaled[:,flag:], n_hours, 0)
    print(reframed.shape) #nº observation x ((nºfeatures + 1) * nºlags)
    
    reframed.head()
    
    #================================================
    # Reframed as 3d data
    #================================================
    # split into train and test sets
    values = reframed.values
    #Train data = 1 year of data
    '''
    To speed up the training of the model for this demonstration, we will only fit 
    the model on the first year of data (365 days * 24hours (1 day)), then evaluate it on the remaining 4 years 
    of data. If you have time, consider exploring the inverted version of 
    this test harness.
    '''
    #n_train_hours = 365 * 24
    #train = values[:n_train_hours, :]
    #test = values[n_train_hours:, :]
    
    # split into input and outputs
    
    n_obs = n_hours * n_features
    #Train and test
    
    train_X, train_y = values, train_scaled[n_hours:,0]
    test_X, test_y = reframed_test.values, test_scaled[n_hours:,0]
    print(train_X.shape, len(train_X), train_y.shape)
    # reshape input (train_X and test_X) to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
    #================================================
    # Create LSTM model for GridSearch
    #================================================
    optimizer = ['adam']
    batch_size = [300]
    epochs = [300]
    init_mode = ['glorot_uniform']
    layers = ['shallow']
    callback = ['ES_30']
    param_grid = dict(batch_size=batch_size, epochs=epochs,optimizer=optimizer,init_mode=init_mode,layers=layers,callback=callback)
    
    model_id = 'shallow'
    MODEL_ID.append(model_id)
    PARAMS.append(param_grid)
    # create model
    model01 = Sequential()
    model01.add(LSTM(3, return_sequences=False,input_shape=(train_X.shape[1], train_X.shape[2])))
    model01.add(Dense(1))
    
    optimizer = tf.keras.optimizers.Adam()
    model01.compile(loss='mse', optimizer=optimizer)
    
    # fit network
    from keras.callbacks import EarlyStopping
    stop = EarlyStopping(monitor='val_loss', patience=30,verbose=0,restore_best_weights=True)
    callbacks_list = [stop]
    
    history = model01.fit(train_X, train_y, epochs=300, batch_size=300, validation_data=(test_X, test_y),callbacks = callbacks_list, verbose=2, shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
     
    # make a prediction
    yhat = model01.predict(test_X)
    
    #================================================
    # Invert Scaler
    #================================================
    #Reshape test_X to get 7 random columns
    test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
    
    #gambiarra
    n_features = n_features-mode
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, -n_features:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    '''
    Q: why do we use -11 in 'test_X[:, -11:]' especially?
    A: We are only interested in inverting the target, but the transform requires 
    the same columns when inverting as when transforming. Therefore we are adding 
    the target with other input vars for the inverse operation.
    '''
    # invert scaling for actual
    
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, -n_features:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    mape = mape_vectorized(inv_y,inv_yhat)
    print('Test MAPE: %.3f' % mape)
    
    #NAIVE - MEAN VALUE: evaluate if I just predict the average of the timeseries
    naive = pd.Series(np.repeat(df['all'][n_train+n_hours:].mean(),len(df['all'][n_train+n_hours:])))
    naive.index = df['all'][n_train+n_hours:].index
    
    naive_mse = mean_squared_error(df['all'][n_train+n_hours:],naive)
    print('\nNAIVE RMSE: ',round(np.sqrt(naive_mse),3))
    naive_mape = mape_vectorized(df['all'][n_train+n_hours:],naive)
    print('NAIVE MAPE: ',round(naive_mape,2),'%')
    
    RMSE_one.append(round(rmse,3))
    MAPE_one.append(round(mape,2))
    NAIVE_rmse.append(round(np.sqrt(naive_mse),2))
    NAIVE_mape.append(round(naive_mape,2))
    
    #================================================
    # Plot Prediction
    #================================================
    # plot time series
    plt.figure(figsize=[20,3])
    #test.iloc[n_hours:].index
    plt.plot(test.iloc[n_hours:].index, inv_y[:], label='Observed',alpha=0.4,color='gray')
    plt.plot(test.iloc[n_hours:].index, inv_yhat[:], label='Predicted',color='dodgerblue')
    #plt.title('Prediction vs Observed\nLast 365 days (~1 year)')
    plt.xlabel('Time',fontsize=15)
    plt.ylabel('Deaths',fontsize=15)
    plt.legend(fontsize=15,loc='upper left')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
    
report_LSTM_shallow02 = pd.DataFrame()
report_LSTM_shallow02['MODEL'] = MODEL_ID
report_LSTM_shallow02['PARAM'] = PARAMS
report_LSTM_shallow02['VARS'] = VARS
report_LSTM_shallow02['LAGS'] = LAGS
report_LSTM_shallow02['RMSE_ONE'] = RMSE_one
report_LSTM_shallow02['MAPE_ONE'] = MAPE_one
report_LSTM_shallow02['NAIVE_RMSE'] = NAIVE_rmse
report_LSTM_shallow02['NAIVE_MAPE'] = NAIVE_mape

os.chdir('C:/Users/trbella/Desktop/Thiago/Python/LSTM_GRID_LAGS/results/daily')
report_LSTM_shallow02.to_excel('report_LSTM_shallow02.xlsx')
    
#================================
# Clean the envinronment
#================================
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

import gc
gc.collect()
    
    
#=====================    
# PARTE 03
#=====================
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import os

#================================================
# FUNCTIONS
#================================================
# MAPE FUNCTION
def mape_vectorized(true, pred): 
    mask = true != 0
    return (np.fabs(true - pred)/true)[mask].mean()*100

#==================================================================
# series_to_supervised FUNCTION - author: Jason Brownlee
#================================================

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
  # dataset varaibles quantity
  n_vars = 1 if type(data) is list else data.shape[1]
  # dataframe transformation
  df = pd.DataFrame(data)
  cols, names = list(), list()

  # input sequence (t-n, ... t-1). lags before time t
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

  # forecast sequence (t, t+1, ... t+n). lags after time t
    
  for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
      names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
      names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
  # put it all together
  agg = pd.concat(cols, axis=1)
  agg.columns = names
  # drop rows with NaN values
  if dropnan:
    agg.dropna(inplace=True)
  return agg

#Variables for dataset creation with grid search results
MODEL_ID = []
PARAMS = []
VARS = []
LAGS = []
RMSE_one = []
MAPE_one = []
NAIVE_rmse = []
NAIVE_mape = []

#================================================
# LOAD DATA
#================================================
os.getcwd()
os.chdir('D:/CLISAU/RAW/DO/OBITOS_ATUALIZADO/processed')
#content/drive/Shared drives/Clima&Saúde/Dados/Dados_Saude/Obitos_SSC/data/processed/2001-2018_obitos_clima_diario.csv
#Deaths and environment
df = pd.read_csv('2001-2018_obitos_clima_diario.csv')
df['DATE'] = pd.to_datetime(df['DATE'],dayfirst=True)
df = df.set_index('DATE')

#================================================
#GRID 03
#================================================
variables = ['MP10_MEAN','TMIN_IAC']

#================================================
# VARIABLE SELECTION
#================================================
# Variables
endog = df.loc[:]['all']
exog = df.loc[:][variables]
endog = endog.asfreq('D')
exog = exog.asfreq('D')
#nobs = endog.shape[0]

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))

#fill exogenous nan with mean (for each exogenous variable)
exog = exog.fillna(exog.mean())

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))
exog.head(2)

#================================================
# Train/Test split 80-20
#================================================
n_train = int(0.8*len(endog))

endog_train = endog.iloc[:n_train].asfreq('D')
endog_test = endog.iloc[n_train:].asfreq('D')

exog_train = exog.iloc[:n_train].asfreq('D')
exog_test = exog.iloc[n_train:].asfreq('D')

lags = list(range(1,16))
for n_hours in lags:
    LAGS.append(n_hours)
    VARS.append(variables)
    #================================================
    # Normalization
    #================================================
    train = pd.merge(endog_train,exog_train,on='DATE')
    test = pd.merge(endog_test,exog_test,on='DATE')
    test.head(3)
    train.head(3)
    
    '''
    if exog does contain endog, mode = 1
    if exog doesn't contain endog, mode = 0
    '''
    mode = 0
    
    if mode == 1:
      flag = 0
    else:
      flag = 1
    
    # load dataset
    values = train.values
    # ensure all data is float
    np.set_printoptions(suppress=True) #supress scientific notation
    values = values.astype('float32')
    
    # normalize train features
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(values)
    # normalize test based on train scaler
    test_scaled = scaler.transform(test.values)
    # specify the number of lag (hours in this case, since each line equals a hour)
    #n_features is the number of the exogenous variables
    
    n_features = len(exog.columns)+mode
    
    #================================================
    # Supervised Learning
    #================================================
    # frame as supervised learning
    
    #Here I used scaled[:,1:] because I exlcuded the y (deaths all) variable from the function below.
    reframed = series_to_supervised(train_scaled[:,flag:], n_hours, 0)
    reframed_test = series_to_supervised(test_scaled[:,flag:], n_hours, 0)
    print(reframed.shape) #nº observation x ((nºfeatures + 1) * nºlags)
    
    reframed.head()
    
    #================================================
    # Reframed as 3d data
    #================================================
    # split into train and test sets
    values = reframed.values
    #Train data = 1 year of data
    '''
    To speed up the training of the model for this demonstration, we will only fit 
    the model on the first year of data (365 days * 24hours (1 day)), then evaluate it on the remaining 4 years 
    of data. If you have time, consider exploring the inverted version of 
    this test harness.
    '''
    #n_train_hours = 365 * 24
    #train = values[:n_train_hours, :]
    #test = values[n_train_hours:, :]
    
    # split into input and outputs
    
    n_obs = n_hours * n_features
    #Train and test
    
    train_X, train_y = values, train_scaled[n_hours:,0]
    test_X, test_y = reframed_test.values, test_scaled[n_hours:,0]
    print(train_X.shape, len(train_X), train_y.shape)
    # reshape input (train_X and test_X) to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
    #================================================
    # Create LSTM model for GridSearch
    #================================================
    optimizer = ['adam']
    batch_size = [300]
    epochs = [300]
    init_mode = ['glorot_uniform']
    layers = ['shallow']
    callback = ['ES_30']
    param_grid = dict(batch_size=batch_size, epochs=epochs,optimizer=optimizer,init_mode=init_mode,layers=layers,callback=callback)
    
    model_id = 'shallow'
    MODEL_ID.append(model_id)
    PARAMS.append(param_grid)
    # create model
    model01 = Sequential()
    model01.add(LSTM(3, return_sequences=False,input_shape=(train_X.shape[1], train_X.shape[2])))
    model01.add(Dense(1))
    
    optimizer = tf.keras.optimizers.Adam()
    model01.compile(loss='mse', optimizer=optimizer)
    
    # fit network
    from keras.callbacks import EarlyStopping
    stop = EarlyStopping(monitor='val_loss', patience=30,verbose=0,restore_best_weights=True)
    callbacks_list = [stop]
    
    history = model01.fit(train_X, train_y, epochs=300, batch_size=300, validation_data=(test_X, test_y),callbacks = callbacks_list, verbose=2, shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
     
    # make a prediction
    yhat = model01.predict(test_X)
    
    #================================================
    # Invert Scaler
    #================================================
    #Reshape test_X to get 7 random columns
    test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
    
    #gambiarra
    n_features = n_features-mode
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, -n_features:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    '''
    Q: why do we use -11 in 'test_X[:, -11:]' especially?
    A: We are only interested in inverting the target, but the transform requires 
    the same columns when inverting as when transforming. Therefore we are adding 
    the target with other input vars for the inverse operation.
    '''
    # invert scaling for actual
    
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, -n_features:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    mape = mape_vectorized(inv_y,inv_yhat)
    print('Test MAPE: %.3f' % mape)
    
    #NAIVE - MEAN VALUE: evaluate if I just predict the average of the timeseries
    naive = pd.Series(np.repeat(df['all'][n_train+n_hours:].mean(),len(df['all'][n_train+n_hours:])))
    naive.index = df['all'][n_train+n_hours:].index
    
    naive_mse = mean_squared_error(df['all'][n_train+n_hours:],naive)
    print('\nNAIVE RMSE: ',round(np.sqrt(naive_mse),3))
    naive_mape = mape_vectorized(df['all'][n_train+n_hours:],naive)
    print('NAIVE MAPE: ',round(naive_mape,2),'%')
    
    RMSE_one.append(round(rmse,3))
    MAPE_one.append(round(mape,2))
    NAIVE_rmse.append(round(np.sqrt(naive_mse),2))
    NAIVE_mape.append(round(naive_mape,2))
    
    #================================================
    # Plot Prediction
    #================================================
    # plot time series
    plt.figure(figsize=[20,3])
    #test.iloc[n_hours:].index
    plt.plot(test.iloc[n_hours:].index, inv_y[:], label='Observed',alpha=0.4,color='gray')
    plt.plot(test.iloc[n_hours:].index, inv_yhat[:], label='Predicted',color='dodgerblue')
    #plt.title('Prediction vs Observed\nLast 365 days (~1 year)')
    plt.xlabel('Time',fontsize=15)
    plt.ylabel('Deaths',fontsize=15)
    plt.legend(fontsize=15,loc='upper left')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
    
report_LSTM_shallow03 = pd.DataFrame()
report_LSTM_shallow03['MODEL'] = MODEL_ID
report_LSTM_shallow03['PARAM'] = PARAMS
report_LSTM_shallow03['VARS'] = VARS
report_LSTM_shallow03['LAGS'] = LAGS
report_LSTM_shallow03['RMSE_ONE'] = RMSE_one
report_LSTM_shallow03['MAPE_ONE'] = MAPE_one
report_LSTM_shallow03['NAIVE_RMSE'] = NAIVE_rmse
report_LSTM_shallow03['NAIVE_MAPE'] = NAIVE_mape

os.chdir('C:/Users/trbella/Desktop/Thiago/Python/LSTM_GRID_LAGS/results/daily')
report_LSTM_shallow03.to_excel('report_LSTM_shallow03.xlsx')
    
#================================
# Clean the envinronment
#================================
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

import gc
gc.collect()
    
    
#=======================    
# PARTE 04
#=======================
    
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import os

#================================================
# FUNCTIONS
#================================================
# MAPE FUNCTION
def mape_vectorized(true, pred): 
    mask = true != 0
    return (np.fabs(true - pred)/true)[mask].mean()*100

#==================================================================
# series_to_supervised FUNCTION - author: Jason Brownlee
#================================================

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
  # dataset varaibles quantity
  n_vars = 1 if type(data) is list else data.shape[1]
  # dataframe transformation
  df = pd.DataFrame(data)
  cols, names = list(), list()

  # input sequence (t-n, ... t-1). lags before time t
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

  # forecast sequence (t, t+1, ... t+n). lags after time t
    
  for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
      names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
      names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
  # put it all together
  agg = pd.concat(cols, axis=1)
  agg.columns = names
  # drop rows with NaN values
  if dropnan:
    agg.dropna(inplace=True)
  return agg

#Variables for dataset creation with grid search results
MODEL_ID = []
PARAMS = []
VARS = []
LAGS = []
RMSE_one = []
MAPE_one = []
NAIVE_rmse = []
NAIVE_mape = []

#================================================
# LOAD DATA
#================================================
os.getcwd()
os.chdir('D:/CLISAU/RAW/DO/OBITOS_ATUALIZADO/processed')
#content/drive/Shared drives/Clima&Saúde/Dados/Dados_Saude/Obitos_SSC/data/processed/2001-2018_obitos_clima_diario.csv
#Deaths and environment
df = pd.read_csv('2001-2018_obitos_clima_diario.csv')
df['DATE'] = pd.to_datetime(df['DATE'],dayfirst=True)
df = df.set_index('DATE')

#================================================
#GRID 04
#================================================
variables = ['MP10_MEAN','CO_MEAN']

#================================================
# VARIABLE SELECTION
#================================================
# Variables
endog = df.loc[:]['all']
exog = df.loc[:][variables]
endog = endog.asfreq('D')
exog = exog.asfreq('D')
#nobs = endog.shape[0]

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))

#fill exogenous nan with mean (for each exogenous variable)
exog = exog.fillna(exog.mean())

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))
exog.head(2)

#================================================
# Train/Test split 80-20
#================================================
n_train = int(0.8*len(endog))

endog_train = endog.iloc[:n_train].asfreq('D')
endog_test = endog.iloc[n_train:].asfreq('D')

exog_train = exog.iloc[:n_train].asfreq('D')
exog_test = exog.iloc[n_train:].asfreq('D')

lags = list(range(1,16))
for n_hours in lags:
    LAGS.append(n_hours)
    VARS.append(variables)
    #================================================
    # Normalization
    #================================================
    train = pd.merge(endog_train,exog_train,on='DATE')
    test = pd.merge(endog_test,exog_test,on='DATE')
    test.head(3)
    train.head(3)
    
    '''
    if exog does contain endog, mode = 1
    if exog doesn't contain endog, mode = 0
    '''
    mode = 0
    
    if mode == 1:
      flag = 0
    else:
      flag = 1
    
    # load dataset
    values = train.values
    # ensure all data is float
    np.set_printoptions(suppress=True) #supress scientific notation
    values = values.astype('float32')
    
    # normalize train features
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(values)
    # normalize test based on train scaler
    test_scaled = scaler.transform(test.values)
    # specify the number of lag (hours in this case, since each line equals a hour)
    #n_features is the number of the exogenous variables
    
    n_features = len(exog.columns)+mode
    
    #================================================
    # Supervised Learning
    #================================================
    # frame as supervised learning
    
    #Here I used scaled[:,1:] because I exlcuded the y (deaths all) variable from the function below.
    reframed = series_to_supervised(train_scaled[:,flag:], n_hours, 0)
    reframed_test = series_to_supervised(test_scaled[:,flag:], n_hours, 0)
    print(reframed.shape) #nº observation x ((nºfeatures + 1) * nºlags)
    
    reframed.head()
    
    #================================================
    # Reframed as 3d data
    #================================================
    # split into train and test sets
    values = reframed.values
    #Train data = 1 year of data
    '''
    To speed up the training of the model for this demonstration, we will only fit 
    the model on the first year of data (365 days * 24hours (1 day)), then evaluate it on the remaining 4 years 
    of data. If you have time, consider exploring the inverted version of 
    this test harness.
    '''
    #n_train_hours = 365 * 24
    #train = values[:n_train_hours, :]
    #test = values[n_train_hours:, :]
    
    # split into input and outputs
    
    n_obs = n_hours * n_features
    #Train and test
    
    train_X, train_y = values, train_scaled[n_hours:,0]
    test_X, test_y = reframed_test.values, test_scaled[n_hours:,0]
    print(train_X.shape, len(train_X), train_y.shape)
    # reshape input (train_X and test_X) to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
    #================================================
    # Create LSTM model for GridSearch
    #================================================
    optimizer = ['adam']
    batch_size = [300]
    epochs = [300]
    init_mode = ['glorot_uniform']
    layers = ['shallow']
    callback = ['ES_30']
    param_grid = dict(batch_size=batch_size, epochs=epochs,optimizer=optimizer,init_mode=init_mode,layers=layers,callback=callback)
    
    model_id = 'shallow'
    MODEL_ID.append(model_id)
    PARAMS.append(param_grid)
    # create model
    model01 = Sequential()
    model01.add(LSTM(3, return_sequences=False,input_shape=(train_X.shape[1], train_X.shape[2])))
    model01.add(Dense(1))
    
    optimizer = tf.keras.optimizers.Adam()
    model01.compile(loss='mse', optimizer=optimizer)
    
    # fit network
    from keras.callbacks import EarlyStopping
    stop = EarlyStopping(monitor='val_loss', patience=30,verbose=0,restore_best_weights=True)
    callbacks_list = [stop]
    
    history = model01.fit(train_X, train_y, epochs=300, batch_size=300, validation_data=(test_X, test_y),callbacks = callbacks_list, verbose=2, shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
     
    # make a prediction
    yhat = model01.predict(test_X)
    
    #================================================
    # Invert Scaler
    #================================================
    #Reshape test_X to get 7 random columns
    test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
    
    #gambiarra
    n_features = n_features-mode
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, -n_features:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    '''
    Q: why do we use -11 in 'test_X[:, -11:]' especially?
    A: We are only interested in inverting the target, but the transform requires 
    the same columns when inverting as when transforming. Therefore we are adding 
    the target with other input vars for the inverse operation.
    '''
    # invert scaling for actual
    
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, -n_features:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    mape = mape_vectorized(inv_y,inv_yhat)
    print('Test MAPE: %.3f' % mape)
    
    #NAIVE - MEAN VALUE: evaluate if I just predict the average of the timeseries
    naive = pd.Series(np.repeat(df['all'][n_train+n_hours:].mean(),len(df['all'][n_train+n_hours:])))
    naive.index = df['all'][n_train+n_hours:].index
    
    naive_mse = mean_squared_error(df['all'][n_train+n_hours:],naive)
    print('\nNAIVE RMSE: ',round(np.sqrt(naive_mse),3))
    naive_mape = mape_vectorized(df['all'][n_train+n_hours:],naive)
    print('NAIVE MAPE: ',round(naive_mape,2),'%')
    
    RMSE_one.append(round(rmse,3))
    MAPE_one.append(round(mape,2))
    NAIVE_rmse.append(round(np.sqrt(naive_mse),2))
    NAIVE_mape.append(round(naive_mape,2))
    
    #================================================
    # Plot Prediction
    #================================================
    # plot time series
    plt.figure(figsize=[20,3])
    #test.iloc[n_hours:].index
    plt.plot(test.iloc[n_hours:].index, inv_y[:], label='Observed',alpha=0.4,color='gray')
    plt.plot(test.iloc[n_hours:].index, inv_yhat[:], label='Predicted',color='dodgerblue')
    #plt.title('Prediction vs Observed\nLast 365 days (~1 year)')
    plt.xlabel('Time',fontsize=15)
    plt.ylabel('Deaths',fontsize=15)
    plt.legend(fontsize=15,loc='upper left')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
    
report_LSTM_shallow04 = pd.DataFrame()
report_LSTM_shallow04['MODEL'] = MODEL_ID
report_LSTM_shallow04['PARAM'] = PARAMS
report_LSTM_shallow04['VARS'] = VARS
report_LSTM_shallow04['LAGS'] = LAGS
report_LSTM_shallow04['RMSE_ONE'] = RMSE_one
report_LSTM_shallow04['MAPE_ONE'] = MAPE_one
report_LSTM_shallow04['NAIVE_RMSE'] = NAIVE_rmse
report_LSTM_shallow04['NAIVE_MAPE'] = NAIVE_mape

os.chdir('C:/Users/trbella/Desktop/Thiago/Python/LSTM_GRID_LAGS/results/daily')
report_LSTM_shallow04.to_excel('report_LSTM_shallow04.xlsx')
    
#================================
# Clean the envinronment
#================================
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

import gc
gc.collect() 
    

#=================
# PARTE 05
#=================
   
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import os

#================================================
# FUNCTIONS
#================================================
# MAPE FUNCTION
def mape_vectorized(true, pred): 
    mask = true != 0
    return (np.fabs(true - pred)/true)[mask].mean()*100

#==================================================================
# series_to_supervised FUNCTION - author: Jason Brownlee
#================================================

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
  # dataset varaibles quantity
  n_vars = 1 if type(data) is list else data.shape[1]
  # dataframe transformation
  df = pd.DataFrame(data)
  cols, names = list(), list()

  # input sequence (t-n, ... t-1). lags before time t
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

  # forecast sequence (t, t+1, ... t+n). lags after time t
    
  for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
      names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
      names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
  # put it all together
  agg = pd.concat(cols, axis=1)
  agg.columns = names
  # drop rows with NaN values
  if dropnan:
    agg.dropna(inplace=True)
  return agg

#Variables for dataset creation with grid search results
MODEL_ID = []
PARAMS = []
VARS = []
LAGS = []
RMSE_one = []
MAPE_one = []
NAIVE_rmse = []
NAIVE_mape = []

#================================================
# LOAD DATA
#================================================
os.getcwd()
os.chdir('D:/CLISAU/RAW/DO/OBITOS_ATUALIZADO/processed')
#content/drive/Shared drives/Clima&Saúde/Dados/Dados_Saude/Obitos_SSC/data/processed/2001-2018_obitos_clima_diario.csv
#Deaths and environment
df = pd.read_csv('2001-2018_obitos_clima_diario.csv')
df['DATE'] = pd.to_datetime(df['DATE'],dayfirst=True)
df = df.set_index('DATE')

#================================================
#GRID 05
#================================================
variables = ['TMIN_IAC']

#================================================
# VARIABLE SELECTION
#================================================
# Variables
endog = df.loc[:]['all']
exog = df.loc[:][variables]
endog = endog.asfreq('D')
exog = exog.asfreq('D')
#nobs = endog.shape[0]

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))

#fill exogenous nan with mean (for each exogenous variable)
exog = exog.fillna(exog.mean())

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))
exog.head(2)

#================================================
# Train/Test split 80-20
#================================================
n_train = int(0.8*len(endog))

endog_train = endog.iloc[:n_train].asfreq('D')
endog_test = endog.iloc[n_train:].asfreq('D')

exog_train = exog.iloc[:n_train].asfreq('D')
exog_test = exog.iloc[n_train:].asfreq('D')

lags = list(range(1,16))
for n_hours in lags:
    LAGS.append(n_hours)
    VARS.append(variables)
    #================================================
    # Normalization
    #================================================
    train = pd.merge(endog_train,exog_train,on='DATE')
    test = pd.merge(endog_test,exog_test,on='DATE')
    test.head(3)
    train.head(3)
    
    '''
    if exog does contain endog, mode = 1
    if exog doesn't contain endog, mode = 0
    '''
    mode = 0
    
    if mode == 1:
      flag = 0
    else:
      flag = 1
    
    # load dataset
    values = train.values
    # ensure all data is float
    np.set_printoptions(suppress=True) #supress scientific notation
    values = values.astype('float32')
    
    # normalize train features
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(values)
    # normalize test based on train scaler
    test_scaled = scaler.transform(test.values)
    # specify the number of lag (hours in this case, since each line equals a hour)
    #n_features is the number of the exogenous variables
    
    n_features = len(exog.columns)+mode
    
    #================================================
    # Supervised Learning
    #================================================
    # frame as supervised learning
    
    #Here I used scaled[:,1:] because I exlcuded the y (deaths all) variable from the function below.
    reframed = series_to_supervised(train_scaled[:,flag:], n_hours, 0)
    reframed_test = series_to_supervised(test_scaled[:,flag:], n_hours, 0)
    print(reframed.shape) #nº observation x ((nºfeatures + 1) * nºlags)
    
    reframed.head()
    
    #================================================
    # Reframed as 3d data
    #================================================
    # split into train and test sets
    values = reframed.values
    #Train data = 1 year of data
    '''
    To speed up the training of the model for this demonstration, we will only fit 
    the model on the first year of data (365 days * 24hours (1 day)), then evaluate it on the remaining 4 years 
    of data. If you have time, consider exploring the inverted version of 
    this test harness.
    '''
    #n_train_hours = 365 * 24
    #train = values[:n_train_hours, :]
    #test = values[n_train_hours:, :]
    
    # split into input and outputs
    
    n_obs = n_hours * n_features
    #Train and test
    
    train_X, train_y = values, train_scaled[n_hours:,0]
    test_X, test_y = reframed_test.values, test_scaled[n_hours:,0]
    print(train_X.shape, len(train_X), train_y.shape)
    # reshape input (train_X and test_X) to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
    #================================================
    # Create LSTM model for GridSearch
    #================================================
    optimizer = ['adam']
    batch_size = [300]
    epochs = [300]
    init_mode = ['glorot_uniform']
    layers = ['shallow']
    callback = ['ES_30']
    param_grid = dict(batch_size=batch_size, epochs=epochs,optimizer=optimizer,init_mode=init_mode,layers=layers,callback=callback)
    
    model_id = 'shallow'
    MODEL_ID.append(model_id)
    PARAMS.append(param_grid)
    # create model
    model01 = Sequential()
    model01.add(LSTM(3, return_sequences=False,input_shape=(train_X.shape[1], train_X.shape[2])))
    model01.add(Dense(1))
    
    optimizer = tf.keras.optimizers.Adam()
    model01.compile(loss='mse', optimizer=optimizer)
    
    # fit network
    from keras.callbacks import EarlyStopping
    stop = EarlyStopping(monitor='val_loss', patience=30,verbose=0,restore_best_weights=True)
    callbacks_list = [stop]
    
    history = model01.fit(train_X, train_y, epochs=300, batch_size=300, validation_data=(test_X, test_y),callbacks = callbacks_list, verbose=2, shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
     
    # make a prediction
    yhat = model01.predict(test_X)
    
    #================================================
    # Invert Scaler
    #================================================
    #Reshape test_X to get 7 random columns
    test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
    
    #gambiarra
    n_features = n_features-mode
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, -n_features:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    '''
    Q: why do we use -11 in 'test_X[:, -11:]' especially?
    A: We are only interested in inverting the target, but the transform requires 
    the same columns when inverting as when transforming. Therefore we are adding 
    the target with other input vars for the inverse operation.
    '''
    # invert scaling for actual
    
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, -n_features:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    mape = mape_vectorized(inv_y,inv_yhat)
    print('Test MAPE: %.3f' % mape)
    
    #NAIVE - MEAN VALUE: evaluate if I just predict the average of the timeseries
    naive = pd.Series(np.repeat(df['all'][n_train+n_hours:].mean(),len(df['all'][n_train+n_hours:])))
    naive.index = df['all'][n_train+n_hours:].index
    
    naive_mse = mean_squared_error(df['all'][n_train+n_hours:],naive)
    print('\nNAIVE RMSE: ',round(np.sqrt(naive_mse),3))
    naive_mape = mape_vectorized(df['all'][n_train+n_hours:],naive)
    print('NAIVE MAPE: ',round(naive_mape,2),'%')
    
    RMSE_one.append(round(rmse,3))
    MAPE_one.append(round(mape,2))
    NAIVE_rmse.append(round(np.sqrt(naive_mse),2))
    NAIVE_mape.append(round(naive_mape,2))
    
    #================================================
    # Plot Prediction
    #================================================
    # plot time series
    plt.figure(figsize=[20,3])
    #test.iloc[n_hours:].index
    plt.plot(test.iloc[n_hours:].index, inv_y[:], label='Observed',alpha=0.4,color='gray')
    plt.plot(test.iloc[n_hours:].index, inv_yhat[:], label='Predicted',color='dodgerblue')
    #plt.title('Prediction vs Observed\nLast 365 days (~1 year)')
    plt.xlabel('Time',fontsize=15)
    plt.ylabel('Deaths',fontsize=15)
    plt.legend(fontsize=15,loc='upper left')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
    
report_LSTM_shallow05 = pd.DataFrame()
report_LSTM_shallow05['MODEL'] = MODEL_ID
report_LSTM_shallow05['PARAM'] = PARAMS
report_LSTM_shallow05['VARS'] = VARS
report_LSTM_shallow05['LAGS'] = LAGS
report_LSTM_shallow05['RMSE_ONE'] = RMSE_one
report_LSTM_shallow05['MAPE_ONE'] = MAPE_one
report_LSTM_shallow05['NAIVE_RMSE'] = NAIVE_rmse
report_LSTM_shallow05['NAIVE_MAPE'] = NAIVE_mape

os.chdir('C:/Users/trbella/Desktop/Thiago/Python/LSTM_GRID_LAGS/results/daily')
report_LSTM_shallow05.to_excel('report_LSTM_shallow05.xlsx')
    
#================================
# Clean the envinronment
#================================
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

import gc
gc.collect() 
   

#====================
# PARTE 06
#====================
   
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import os

#================================================
# FUNCTIONS
#================================================
# MAPE FUNCTION
def mape_vectorized(true, pred): 
    mask = true != 0
    return (np.fabs(true - pred)/true)[mask].mean()*100

#==================================================================
# series_to_supervised FUNCTION - author: Jason Brownlee
#================================================

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
  # dataset varaibles quantity
  n_vars = 1 if type(data) is list else data.shape[1]
  # dataframe transformation
  df = pd.DataFrame(data)
  cols, names = list(), list()

  # input sequence (t-n, ... t-1). lags before time t
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

  # forecast sequence (t, t+1, ... t+n). lags after time t
    
  for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
      names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
      names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
  # put it all together
  agg = pd.concat(cols, axis=1)
  agg.columns = names
  # drop rows with NaN values
  if dropnan:
    agg.dropna(inplace=True)
  return agg

#Variables for dataset creation with grid search results
MODEL_ID = []
PARAMS = []
VARS = []
LAGS = []
RMSE_one = []
MAPE_one = []
NAIVE_rmse = []
NAIVE_mape = []

#================================================
# LOAD DATA
#================================================
os.getcwd()
os.chdir('D:/CLISAU/RAW/DO/OBITOS_ATUALIZADO/processed')
#content/drive/Shared drives/Clima&Saúde/Dados/Dados_Saude/Obitos_SSC/data/processed/2001-2018_obitos_clima_diario.csv
#Deaths and environment
df = pd.read_csv('2001-2018_obitos_clima_diario.csv')
df['DATE'] = pd.to_datetime(df['DATE'],dayfirst=True)
df = df.set_index('DATE')

#================================================
#GRID 06
#================================================
variables = ['MP10_MEAN']

#================================================
# VARIABLE SELECTION
#================================================
# Variables
endog = df.loc[:]['all']
exog = df.loc[:][variables]
endog = endog.asfreq('D')
exog = exog.asfreq('D')
#nobs = endog.shape[0]

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))

#fill exogenous nan with mean (for each exogenous variable)
exog = exog.fillna(exog.mean())

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))
exog.head(2)

#================================================
# Train/Test split 80-20
#================================================
n_train = int(0.8*len(endog))

endog_train = endog.iloc[:n_train].asfreq('D')
endog_test = endog.iloc[n_train:].asfreq('D')

exog_train = exog.iloc[:n_train].asfreq('D')
exog_test = exog.iloc[n_train:].asfreq('D')

lags = list(range(1,16))
for n_hours in lags:
    LAGS.append(n_hours)
    VARS.append(variables)
    #================================================
    # Normalization
    #================================================
    train = pd.merge(endog_train,exog_train,on='DATE')
    test = pd.merge(endog_test,exog_test,on='DATE')
    test.head(3)
    train.head(3)
    
    '''
    if exog does contain endog, mode = 1
    if exog doesn't contain endog, mode = 0
    '''
    mode = 0
    
    if mode == 1:
      flag = 0
    else:
      flag = 1
    
    # load dataset
    values = train.values
    # ensure all data is float
    np.set_printoptions(suppress=True) #supress scientific notation
    values = values.astype('float32')
    
    # normalize train features
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(values)
    # normalize test based on train scaler
    test_scaled = scaler.transform(test.values)
    # specify the number of lag (hours in this case, since each line equals a hour)
    #n_features is the number of the exogenous variables
    
    n_features = len(exog.columns)+mode
    
    #================================================
    # Supervised Learning
    #================================================
    # frame as supervised learning
    
    #Here I used scaled[:,1:] because I exlcuded the y (deaths all) variable from the function below.
    reframed = series_to_supervised(train_scaled[:,flag:], n_hours, 0)
    reframed_test = series_to_supervised(test_scaled[:,flag:], n_hours, 0)
    print(reframed.shape) #nº observation x ((nºfeatures + 1) * nºlags)
    
    reframed.head()
    
    #================================================
    # Reframed as 3d data
    #================================================
    # split into train and test sets
    values = reframed.values
    #Train data = 1 year of data
    '''
    To speed up the training of the model for this demonstration, we will only fit 
    the model on the first year of data (365 days * 24hours (1 day)), then evaluate it on the remaining 4 years 
    of data. If you have time, consider exploring the inverted version of 
    this test harness.
    '''
    #n_train_hours = 365 * 24
    #train = values[:n_train_hours, :]
    #test = values[n_train_hours:, :]
    
    # split into input and outputs
    
    n_obs = n_hours * n_features
    #Train and test
    
    train_X, train_y = values, train_scaled[n_hours:,0]
    test_X, test_y = reframed_test.values, test_scaled[n_hours:,0]
    print(train_X.shape, len(train_X), train_y.shape)
    # reshape input (train_X and test_X) to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
    #================================================
    # Create LSTM model for GridSearch
    #================================================
    optimizer = ['adam']
    batch_size = [300]
    epochs = [300]
    init_mode = ['glorot_uniform']
    layers = ['shallow']
    callback = ['ES_30']
    param_grid = dict(batch_size=batch_size, epochs=epochs,optimizer=optimizer,init_mode=init_mode,layers=layers,callback=callback)
    
    model_id = 'shallow'
    MODEL_ID.append(model_id)
    PARAMS.append(param_grid)
    # create model
    model01 = Sequential()
    model01.add(LSTM(3, return_sequences=False,input_shape=(train_X.shape[1], train_X.shape[2])))
    model01.add(Dense(1))
    
    optimizer = tf.keras.optimizers.Adam()
    model01.compile(loss='mse', optimizer=optimizer)
    
    # fit network
    from keras.callbacks import EarlyStopping
    stop = EarlyStopping(monitor='val_loss', patience=30,verbose=0,restore_best_weights=True)
    callbacks_list = [stop]
    
    history = model01.fit(train_X, train_y, epochs=300, batch_size=300, validation_data=(test_X, test_y),callbacks = callbacks_list, verbose=2, shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
     
    # make a prediction
    yhat = model01.predict(test_X)
    
    #================================================
    # Invert Scaler
    #================================================
    #Reshape test_X to get 7 random columns
    test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
    
    #gambiarra
    n_features = n_features-mode
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, -n_features:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    '''
    Q: why do we use -11 in 'test_X[:, -11:]' especially?
    A: We are only interested in inverting the target, but the transform requires 
    the same columns when inverting as when transforming. Therefore we are adding 
    the target with other input vars for the inverse operation.
    '''
    # invert scaling for actual
    
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, -n_features:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    mape = mape_vectorized(inv_y,inv_yhat)
    print('Test MAPE: %.3f' % mape)
    
    #NAIVE - MEAN VALUE: evaluate if I just predict the average of the timeseries
    naive = pd.Series(np.repeat(df['all'][n_train+n_hours:].mean(),len(df['all'][n_train+n_hours:])))
    naive.index = df['all'][n_train+n_hours:].index
    
    naive_mse = mean_squared_error(df['all'][n_train+n_hours:],naive)
    print('\nNAIVE RMSE: ',round(np.sqrt(naive_mse),3))
    naive_mape = mape_vectorized(df['all'][n_train+n_hours:],naive)
    print('NAIVE MAPE: ',round(naive_mape,2),'%')
    
    RMSE_one.append(round(rmse,3))
    MAPE_one.append(round(mape,2))
    NAIVE_rmse.append(round(np.sqrt(naive_mse),2))
    NAIVE_mape.append(round(naive_mape,2))
    
    #================================================
    # Plot Prediction
    #================================================
    # plot time series
    plt.figure(figsize=[20,3])
    #test.iloc[n_hours:].index
    plt.plot(test.iloc[n_hours:].index, inv_y[:], label='Observed',alpha=0.4,color='gray')
    plt.plot(test.iloc[n_hours:].index, inv_yhat[:], label='Predicted',color='dodgerblue')
    #plt.title('Prediction vs Observed\nLast 365 days (~1 year)')
    plt.xlabel('Time',fontsize=15)
    plt.ylabel('Deaths',fontsize=15)
    plt.legend(fontsize=15,loc='upper left')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
    
report_LSTM_shallow06 = pd.DataFrame()
report_LSTM_shallow06['MODEL'] = MODEL_ID
report_LSTM_shallow06['PARAM'] = PARAMS
report_LSTM_shallow06['VARS'] = VARS
report_LSTM_shallow06['LAGS'] = LAGS
report_LSTM_shallow06['RMSE_ONE'] = RMSE_one
report_LSTM_shallow06['MAPE_ONE'] = MAPE_one
report_LSTM_shallow06['NAIVE_RMSE'] = NAIVE_rmse
report_LSTM_shallow06['NAIVE_MAPE'] = NAIVE_mape

os.chdir('C:/Users/trbella/Desktop/Thiago/Python/LSTM_GRID_LAGS/results/daily')
report_LSTM_shallow06.to_excel('report_LSTM_shallow06.xlsx')
    
#================================
# Clean the envinronment
#================================
try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

import gc
gc.collect() 
    
#=================
# PARTE 07
#=================
   
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import os

#================================================
# FUNCTIONS
#================================================
# MAPE FUNCTION
def mape_vectorized(true, pred): 
    mask = true != 0
    return (np.fabs(true - pred)/true)[mask].mean()*100

#==================================================================
# series_to_supervised FUNCTION - author: Jason Brownlee
#================================================

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
  # dataset varaibles quantity
  n_vars = 1 if type(data) is list else data.shape[1]
  # dataframe transformation
  df = pd.DataFrame(data)
  cols, names = list(), list()

  # input sequence (t-n, ... t-1). lags before time t
  for i in range(n_in, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

  # forecast sequence (t, t+1, ... t+n). lags after time t
    
  for i in range(0, n_out):
    cols.append(df.shift(-i))
    if i == 0:
      names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
      names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
  # put it all together
  agg = pd.concat(cols, axis=1)
  agg.columns = names
  # drop rows with NaN values
  if dropnan:
    agg.dropna(inplace=True)
  return agg

#Variables for dataset creation with grid search results
MODEL_ID = []
PARAMS = []
VARS = []
LAGS = []
RMSE_one = []
MAPE_one = []
NAIVE_rmse = []
NAIVE_mape = []

#================================================
# LOAD DATA
#================================================
os.getcwd()
os.chdir('D:/CLISAU/RAW/DO/OBITOS_ATUALIZADO/processed')
#content/drive/Shared drives/Clima&Saúde/Dados/Dados_Saude/Obitos_SSC/data/processed/2001-2018_obitos_clima_diario.csv
#Deaths and environment
df = pd.read_csv('2001-2018_obitos_clima_diario.csv')
df['DATE'] = pd.to_datetime(df['DATE'],dayfirst=True)
df = df.set_index('DATE')

#================================================
#GRID 06
#================================================
variables = ['CO_MEAN']

#================================================
# VARIABLE SELECTION
#================================================
# Variables
endog = df.loc[:]['all']
exog = df.loc[:][variables]
endog = endog.asfreq('D')
exog = exog.asfreq('D')
#nobs = endog.shape[0]

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))

#fill exogenous nan with mean (for each exogenous variable)
exog = exog.fillna(exog.mean())

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))
exog.head(2)

#================================================
# Train/Test split 80-20
#================================================
n_train = int(0.8*len(endog))

endog_train = endog.iloc[:n_train].asfreq('D')
endog_test = endog.iloc[n_train:].asfreq('D')

exog_train = exog.iloc[:n_train].asfreq('D')
exog_test = exog.iloc[n_train:].asfreq('D')

lags = list(range(1,16))
for n_hours in lags:
    LAGS.append(n_hours)
    VARS.append(variables)
    #================================================
    # Normalization
    #================================================
    train = pd.merge(endog_train,exog_train,on='DATE')
    test = pd.merge(endog_test,exog_test,on='DATE')
    test.head(3)
    train.head(3)
    
    '''
    if exog does contain endog, mode = 1
    if exog doesn't contain endog, mode = 0
    '''
    mode = 0
    
    if mode == 1:
      flag = 0
    else:
      flag = 1
    
    # load dataset
    values = train.values
    # ensure all data is float
    np.set_printoptions(suppress=True) #supress scientific notation
    values = values.astype('float32')
    
    # normalize train features
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(values)
    # normalize test based on train scaler
    test_scaled = scaler.transform(test.values)
    # specify the number of lag (hours in this case, since each line equals a hour)
    #n_features is the number of the exogenous variables
    
    n_features = len(exog.columns)+mode
    
    #================================================
    # Supervised Learning
    #================================================
    # frame as supervised learning
    
    #Here I used scaled[:,1:] because I exlcuded the y (deaths all) variable from the function below.
    reframed = series_to_supervised(train_scaled[:,flag:], n_hours, 0)
    reframed_test = series_to_supervised(test_scaled[:,flag:], n_hours, 0)
    print(reframed.shape) #nº observation x ((nºfeatures + 1) * nºlags)
    
    reframed.head()
    
    #================================================
    # Reframed as 3d data
    #================================================
    # split into train and test sets
    values = reframed.values
    #Train data = 1 year of data
    '''
    To speed up the training of the model for this demonstration, we will only fit 
    the model on the first year of data (365 days * 24hours (1 day)), then evaluate it on the remaining 4 years 
    of data. If you have time, consider exploring the inverted version of 
    this test harness.
    '''
    #n_train_hours = 365 * 24
    #train = values[:n_train_hours, :]
    #test = values[n_train_hours:, :]
    
    # split into input and outputs
    
    n_obs = n_hours * n_features
    #Train and test
    
    train_X, train_y = values, train_scaled[n_hours:,0]
    test_X, test_y = reframed_test.values, test_scaled[n_hours:,0]
    print(train_X.shape, len(train_X), train_y.shape)
    # reshape input (train_X and test_X) to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    
    #================================================
    # Create LSTM model for GridSearch
    #================================================
    optimizer = ['adam']
    batch_size = [300]
    epochs = [300]
    init_mode = ['glorot_uniform']
    layers = ['shallow']
    callback = ['ES_30']
    param_grid = dict(batch_size=batch_size, epochs=epochs,optimizer=optimizer,init_mode=init_mode,layers=layers,callback=callback)
    
    model_id = 'shallow'
    MODEL_ID.append(model_id)
    PARAMS.append(param_grid)
    # create model
    model01 = Sequential()
    model01.add(LSTM(3, return_sequences=False,input_shape=(train_X.shape[1], train_X.shape[2])))
    model01.add(Dense(1))
    
    optimizer = tf.keras.optimizers.Adam()
    model01.compile(loss='mse', optimizer=optimizer)
    
    # fit network
    from keras.callbacks import EarlyStopping
    stop = EarlyStopping(monitor='val_loss', patience=30,verbose=0,restore_best_weights=True)
    callbacks_list = [stop]
    
    history = model01.fit(train_X, train_y, epochs=300, batch_size=300, validation_data=(test_X, test_y),callbacks = callbacks_list, verbose=2, shuffle=False)
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()
     
    # make a prediction
    yhat = model01.predict(test_X)
    
    #================================================
    # Invert Scaler
    #================================================
    #Reshape test_X to get 7 random columns
    test_X = test_X.reshape((test_X.shape[0], n_hours*n_features))
    
    #gambiarra
    n_features = n_features-mode
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, -n_features:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,0]
    '''
    Q: why do we use -11 in 'test_X[:, -11:]' especially?
    A: We are only interested in inverting the target, but the transform requires 
    the same columns when inverting as when transforming. Therefore we are adding 
    the target with other input vars for the inverse operation.
    '''
    # invert scaling for actual
    
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, -n_features:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    mape = mape_vectorized(inv_y,inv_yhat)
    print('Test MAPE: %.3f' % mape)
    
    #NAIVE - MEAN VALUE: evaluate if I just predict the average of the timeseries
    naive = pd.Series(np.repeat(df['all'][n_train+n_hours:].mean(),len(df['all'][n_train+n_hours:])))
    naive.index = df['all'][n_train+n_hours:].index
    
    naive_mse = mean_squared_error(df['all'][n_train+n_hours:],naive)
    print('\nNAIVE RMSE: ',round(np.sqrt(naive_mse),3))
    naive_mape = mape_vectorized(df['all'][n_train+n_hours:],naive)
    print('NAIVE MAPE: ',round(naive_mape,2),'%')
    
    RMSE_one.append(round(rmse,3))
    MAPE_one.append(round(mape,2))
    NAIVE_rmse.append(round(np.sqrt(naive_mse),2))
    NAIVE_mape.append(round(naive_mape,2))
    
    #================================================
    # Plot Prediction
    #================================================
    # plot time series
    plt.figure(figsize=[20,3])
    #test.iloc[n_hours:].index
    plt.plot(test.iloc[n_hours:].index, inv_y[:], label='Observed',alpha=0.4,color='gray')
    plt.plot(test.iloc[n_hours:].index, inv_yhat[:], label='Predicted',color='dodgerblue')
    #plt.title('Prediction vs Observed\nLast 365 days (~1 year)')
    plt.xlabel('Time',fontsize=15)
    plt.ylabel('Deaths',fontsize=15)
    plt.legend(fontsize=15,loc='upper left')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
    
report_LSTM_shallow07 = pd.DataFrame()
report_LSTM_shallow07['MODEL'] = MODEL_ID
report_LSTM_shallow07['PARAM'] = PARAMS
report_LSTM_shallow07['VARS'] = VARS
report_LSTM_shallow07['LAGS'] = LAGS
report_LSTM_shallow07['RMSE_ONE'] = RMSE_one
report_LSTM_shallow07['MAPE_ONE'] = MAPE_one
report_LSTM_shallow07['NAIVE_RMSE'] = NAIVE_rmse
report_LSTM_shallow07['NAIVE_MAPE'] = NAIVE_mape

os.chdir('C:/Users/trbella/Desktop/Thiago/Python/LSTM_GRID_LAGS/results/daily')
report_LSTM_shallow07.to_excel('report_LSTM_shallow07.xlsx')

