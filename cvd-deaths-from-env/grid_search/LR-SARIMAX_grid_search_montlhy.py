# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 12:24:55 2022

@author: Thiago Ribas Bella
"""

#==================================================================
# PACKAGES
#==================================================================
import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import urllib.request, json
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as np
import seaborn as sns
from statsmodels.stats.diagnostic import acorr_ljungbox as lbox
from statsmodels.graphics.tsaplots import plot_acf 
from statsmodels.graphics.tsaplots import plot_pacf 
import os
import gc


#==================================================================
# series_to_supervised FUNCTION - author - Jason Brownlee
#==================================================================
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
#==================================================================
# MAPE FUNCTION
#==================================================================
def mape_vectorized(true, pred): 
    mask = true != 0
    return (np.fabs(true - pred)/true)[mask].mean()*100

#Variables for dataset creation with grid search results
MODEL_ID = []
VARS = []
LAGS = []
MODEL_AIC = []
NORM_MW = []
RESID = []
HETERO = []
RMSE_dy = []
RMSE_one = []
MAPE_dy = []
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
# Variables
#================================================
endog = df.loc[:]['all']
exog = df.loc[:][variables]
endog = endog.asfreq('D')
exog = exog.asfreq('D')
#nobs = endog.shape[0]

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))

endog = endog.resample('M',label='right').sum()
exog = exog.resample('M',label='right').sum()

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
n_train = int(0.8*len(exog))
endog_train = endog.iloc[:n_train].asfreq('M')
endog_test = endog.iloc[n_train:].asfreq('M')
exog_train = exog.iloc[:n_train].asfreq('M')
exog_test = exog.iloc[n_train:].asfreq('M')

lags = list(range(1,4))
for lag in lags:
    VARS.append(variables)
    LAGS.append(lag)
    #================================================
    #Supervised Learning
    #================================================
    # load dataset
    train_np = exog_train[variables].values
    # ensure all data is float
    np.set_printoptions(suppress=True) #supress scientific notation
    train_np = train_np.astype('float32')
    
    # load dataset
    test_np = exog_test[variables].values
    # ensure all data is float
    np.set_printoptions(suppress=True) #supress scientific notation
    test_np = test_np.astype('float32')
    
    # specify the number of lag (hours in this case, since each line equals a hour)
    #n_features is the number of the exogenous variables
    n_hours = lag
    n_features = len(exog[variables].columns)
    
    # frame as supervised learning
    
    #Here I used scaled[:,1:] because I exlcuded the y (deaths all) variable from the function below.
    reframed = series_to_supervised(train_np[:,:], lag, 0)
    reframed_test = series_to_supervised(test_np[:,:], lag, 0)
    reframed_all = series_to_supervised(exog[variables], lag, 0)
    print(reframed.shape) #nº observation x ((nºfeatures + 1) * nºlags)
    
    reframed.head()
        
    #================================================
    # Automatic Model Fit - pmdarima
    #================================================
    
    #See Tips and Tricks: https://alkaline-ml.com/pmdarima/tips_and_tricks.html
    
    # SARIMAX Model
    print('PMDARIMA AUTO-MODEL')
    sxmodel = pm.auto_arima(endog_train[lag:],reframed, start_p=0, start_q=0, max_p=lag , max_q=lag ,
                               start_P=0,start_Q=0, max_P=lag ,max_Q=lag , m=12, seasonal=True,
                            trace=True, error_action='trace',suppress_warnings=True, stepwise=True,method='lbfgs')
    
    print(sxmodel.summary())
    
    sxmodel.plot_diagnostics(figsize=(10,5))
    plt.show()
    
    #check if the best model has trend
    a = pd.Series(sxmodel.pvalues()).index
    b = 'intercept'
    if len([i for i in a if i in b])>=1:
      inter = 'int.'
      trend = 'c'
    else:
      trend = None
      inter = None
      
    model_id = f'{sxmodel.order} {sxmodel.seasonal_order} {inter}'
    MODEL_ID.append(model_id)
  
    model_aic = sxmodel.aic()
    MODEL_AIC.append(round(model_aic,3))

    RESID.append(round(sxmodel.resid().mean(),5))
    
    #================================================
    #Statsmodels model - training set
    #From here below I will use Statsmodels instead of pmdarima
    #================================================
    print('STATSMODELS SARIMAX')
    mod = SARIMAX(endog_train[lag:].values, reframed[:], order=sxmodel.order,seasonal_order= sxmodel.seasonal_order,trend=trend)
    #mod = SARIMAX(endog_train[lag:].values, reframed[['var1(t-5)','var1(t-3)']][:], order=(5,1,0),seasonal_order= (0,0,1,7),trend=None)
    
    fit_res = mod.fit(maxiter=500,method='bfgs',disp=0) #disp controls verbosity
    
    print('\033[1m'+'Residuals mean ='+'\033[0m', round(fit_res.resid.mean(),5))
    from scipy.stats import mannwhitneyu as mwu
    normal = np.random.normal(0,fit_res.resid.std(),len(fit_res.resid))
    mann = mwu(normal,fit_res.resid,alternative = 'two-sided')
    print('\033[4m'+'\nTheoretical normal vs Residuals - hypothesis test:'+'\033[0m')
    print('\033[1m'+'Mann-Whitney U p-value:'+'\033[0m', round(mann[1],5))
    print(fit_res.summary())
    fit_res.plot_diagnostics(figsize=(10,5))
    plt.show()
    
    NORM_MW.append(round(mann[1],4))
    
    #================================================
    #Heteroscedasticity - Breusch-Pagan Lagrange Multiplier test
    #================================================
    #Heteroskedasticity is indicated if p <0.05
    from statsmodels.stats.diagnostic import het_breuschpagan
    bp_test = het_breuschpagan(fit_res.resid[0:], reframed[0:])
    #values: 'LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value'
    hetero = bp_test[1]
    
    HETERO.append(hetero)
    
    #================================================
    #Statsmodels model - all data
    #================================================
    #Here we will fit the model with all data, but just in order to do the 
    #one-step-ahead prediction. The model coefficients will be the same found 
    #using only the training set.
    
    mod = SARIMAX(endog[lag:], exog=reframed_all, order=sxmodel.order,seasonal_order=sxmodel.seasonal_order,trend=trend)
    #mod = SARIMAX(endog[lag:], exog=reframed_all[['var1(t-5)','var1(t-3)']][:], order=(5,1,0),seasonal_order= (0,0,1,7),trend=None)
    res = mod.filter(fit_res.params) #estimated parameters of training set
    print(res.summary())
    
    #================================================
    #Prediction and Forecast
    #================================================
    # In-sample one-step-ahead predictions
    predict = res.get_prediction()
    predict_ci = predict.conf_int()
    
    # Dynamic predictions
    predict_dy = res.get_prediction(dynamic=endog.iloc[:n_train].index[-1])
    predict_dy_ci = predict_dy.conf_int()
    
    #================================================
    #Root Mean Squared Error (RMSE) - Test set
    #================================================
    from sklearn.metrics import mean_squared_error
    mse_dynamic = mean_squared_error(endog[n_train+lag:],predict_dy.predicted_mean[n_train:])
    #print('RMSE Dynamic: ',round(np.sqrt(mse_dynamic),3))
    mse_one = mean_squared_error(endog[n_train+lag:],predict.predicted_mean[n_train:])
    print('RMSE One-Step-Ahead: ',round(np.sqrt(mse_one),3))
    
    mape_dy = mape_vectorized(endog[n_train+lag:],predict_dy.predicted_mean[n_train:])
    #print('\nMAPE Dynamic: ',round(mape_dy,2),'%')
    mape_one = mape_vectorized(endog[n_train+lag:],predict.predicted_mean[n_train:])
    print('MAPE One-Step-Ahead: ',round(mape_one,2),'%')
    
    
    #NAIVE - MEAN VALUE: evaluate if I just predict the average of the timeseries
    naive = pd.Series(np.repeat(endog[n_train+lag:].mean(),len(endog[n_train+lag:])))
    naive.index = endog[n_train+lag:].index
    
    naive_mse = mean_squared_error(endog[n_train+lag:],naive)
    print('\nNAIVE RMSE: ',round(np.sqrt(naive_mse),3))
    naive_mape = mape_vectorized(endog[n_train+lag:],naive)
    print('NAIVE MAPE: ',round(naive_mape,2),'%')
    
    RMSE_dy.append(round(np.sqrt(mse_dynamic),3))
    RMSE_one.append(round(np.sqrt(mse_one),3))

    MAPE_dy.append(round(mape_dy,2))
    MAPE_one.append(round(mape_one,2))

    NAIVE_rmse.append(round(np.sqrt(naive_mse),2))
    NAIVE_mape.append(round(naive_mape,2))
    
report01_monthly = pd.DataFrame()
report01_monthly['PARAMS'] = MODEL_ID
report01_monthly['VARS'] = VARS
report01_monthly['LAGS'] = LAGS
report01_monthly['AIC'] = MODEL_AIC
report01_monthly['NORM_TEST_p'] = NORM_MW
report01_monthly['RESID.'] = RESID
report01_monthly['HET_TEST'] = HETERO
report01_monthly['RMSE_DY'] = RMSE_dy
report01_monthly['RMSE_ONE'] = RMSE_one
report01_monthly['MAPE_DY'] = MAPE_dy
report01_monthly['MAPE_ONE'] = MAPE_one
report01_monthly['NAIVE_RMSE'] = NAIVE_rmse
report01_monthly['NAIVE_MAPE'] = NAIVE_mape

os.chdir('C:/Users/trbella/Desktop/Thiago/Python/SARIMAX_GRID_LAGS/results/monthly_rev01')
report01_monthly.to_excel('report01_monthly_rev01.xlsx')

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


#================================
#PARTE 02
#================================
#==================================================================
# PACKAGES
#==================================================================
import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import urllib.request, json
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as np
import seaborn as sns
from statsmodels.stats.diagnostic import acorr_ljungbox as lbox
from statsmodels.graphics.tsaplots import plot_acf 
from statsmodels.graphics.tsaplots import plot_pacf 
import os
import gc


#==================================================================
# series_to_supervised FUNCTION - author - Jason Brownlee
#==================================================================
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
#==================================================================
# MAPE FUNCTION
#==================================================================
def mape_vectorized(true, pred): 
    mask = true != 0
    return (np.fabs(true - pred)/true)[mask].mean()*100

#Variables for dataset creation with grid search results
MODEL_ID = []
VARS = []
LAGS = []
MODEL_AIC = []
NORM_MW = []
RESID = []
HETERO = []
RMSE_dy = []
RMSE_one = []
MAPE_dy = []
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
# Variables
#================================================
endog = df.loc[:]['all']
exog = df.loc[:][variables]
endog = endog.asfreq('D')
exog = exog.asfreq('D')
#nobs = endog.shape[0]

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))

endog = endog.resample('M',label='right').sum()
exog = exog.resample('M',label='right').sum()

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
n_train = int(0.8*len(exog))
endog_train = endog.iloc[:n_train].asfreq('M')
endog_test = endog.iloc[n_train:].asfreq('M')
exog_train = exog.iloc[:n_train].asfreq('M')
exog_test = exog.iloc[n_train:].asfreq('M')

lags = list(range(1,4))
for lag in lags:
    VARS.append(variables)
    LAGS.append(lag)
    #================================================
    #Supervised Learning
    #================================================
    # load dataset
    train_np = exog_train[variables].values
    # ensure all data is float
    np.set_printoptions(suppress=True) #supress scientific notation
    train_np = train_np.astype('float32')
    
    # load dataset
    test_np = exog_test[variables].values
    # ensure all data is float
    np.set_printoptions(suppress=True) #supress scientific notation
    test_np = test_np.astype('float32')
    
    # specify the number of lag (hours in this case, since each line equals a hour)
    #n_features is the number of the exogenous variables
    n_hours = lag
    n_features = len(exog[variables].columns)
    
    # frame as supervised learning
    
    #Here I used scaled[:,1:] because I exlcuded the y (deaths all) variable from the function below.
    reframed = series_to_supervised(train_np[:,:], lag, 0)
    reframed_test = series_to_supervised(test_np[:,:], lag, 0)
    reframed_all = series_to_supervised(exog[variables], lag, 0)
    print(reframed.shape) #nº observation x ((nºfeatures + 1) * nºlags)
    
    reframed.head()
        
    #================================================
    # Automatic Model Fit - pmdarima
    #================================================
    
    #See Tips and Tricks: https://alkaline-ml.com/pmdarima/tips_and_tricks.html
    
    # SARIMAX Model
    print('PMDARIMA AUTO-MODEL')
    sxmodel = pm.auto_arima(endog_train[lag:],reframed, start_p=0, start_q=0, max_p=lag , max_q=lag ,
                               start_P=0,start_Q=0, max_P=lag ,max_Q=lag , m=12, seasonal=True,
                            trace=True, error_action='trace',suppress_warnings=True, stepwise=True,method='lbfgs')
    
    print(sxmodel.summary())
    
    sxmodel.plot_diagnostics(figsize=(10,5))
    plt.show()
    
    #check if the best model has trend
    a = pd.Series(sxmodel.pvalues()).index
    b = 'intercept'
    if len([i for i in a if i in b])>=1:
      inter = 'int.'
      trend = 'c'
    else:
      trend = None
      inter = None
      
    model_id = f'{sxmodel.order} {sxmodel.seasonal_order} {inter}'
    MODEL_ID.append(model_id)
  
    model_aic = sxmodel.aic()
    MODEL_AIC.append(round(model_aic,3))

    RESID.append(round(sxmodel.resid().mean(),5))
    
    #================================================
    #Statsmodels model - training set
    #From here below I will use Statsmodels instead of pmdarima
    #================================================
    print('STATSMODELS SARIMAX')
    mod = SARIMAX(endog_train[lag:].values, reframed[:], order=sxmodel.order,seasonal_order= sxmodel.seasonal_order,trend=trend)
    #mod = SARIMAX(endog_train[lag:].values, reframed[['var1(t-5)','var1(t-3)']][:], order=(5,1,0),seasonal_order= (0,0,1,7),trend=None)
    
    fit_res = mod.fit(maxiter=500,method='bfgs',disp=0) #disp controls verbosity
    
    print('\033[1m'+'Residuals mean ='+'\033[0m', round(fit_res.resid.mean(),5))
    from scipy.stats import mannwhitneyu as mwu
    normal = np.random.normal(0,fit_res.resid.std(),len(fit_res.resid))
    mann = mwu(normal,fit_res.resid,alternative = 'two-sided')
    print('\033[4m'+'\nTheoretical normal vs Residuals - hypothesis test:'+'\033[0m')
    print('\033[1m'+'Mann-Whitney U p-value:'+'\033[0m', round(mann[1],5))
    print(fit_res.summary())
    fit_res.plot_diagnostics(figsize=(10,5))
    plt.show()
    
    NORM_MW.append(round(mann[1],4))
    
    #================================================
    #Heteroscedasticity - Breusch-Pagan Lagrange Multiplier test
    #================================================
    #Heteroskedasticity is indicated if p <0.05
    from statsmodels.stats.diagnostic import het_breuschpagan
    bp_test = het_breuschpagan(fit_res.resid[0:], reframed[0:])
    #values: 'LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value'
    hetero = bp_test[1]
    
    HETERO.append(hetero)
    
    #================================================
    #Statsmodels model - all data
    #================================================
    #Here we will fit the model with all data, but just in order to do the 
    #one-step-ahead prediction. The model coefficients will be the same found 
    #using only the training set.
    
    mod = SARIMAX(endog[lag:], exog=reframed_all, order=sxmodel.order,seasonal_order=sxmodel.seasonal_order,trend=trend)
    #mod = SARIMAX(endog[lag:], exog=reframed_all[['var1(t-5)','var1(t-3)']][:], order=(5,1,0),seasonal_order= (0,0,1,7),trend=None)
    res = mod.filter(fit_res.params) #estimated parameters of training set
    print(res.summary())
    
    #================================================
    #Prediction and Forecast
    #================================================
    # In-sample one-step-ahead predictions
    predict = res.get_prediction()
    predict_ci = predict.conf_int()
    
    # Dynamic predictions
    predict_dy = res.get_prediction(dynamic=endog.iloc[:n_train].index[-1])
    predict_dy_ci = predict_dy.conf_int()
    
    #================================================
    #Root Mean Squared Error (RMSE) - Test set
    #================================================
    from sklearn.metrics import mean_squared_error
    mse_dynamic = mean_squared_error(endog[n_train+lag:],predict_dy.predicted_mean[n_train:])
    #print('RMSE Dynamic: ',round(np.sqrt(mse_dynamic),3))
    mse_one = mean_squared_error(endog[n_train+lag:],predict.predicted_mean[n_train:])
    print('RMSE One-Step-Ahead: ',round(np.sqrt(mse_one),3))
    
    mape_dy = mape_vectorized(endog[n_train+lag:],predict_dy.predicted_mean[n_train:])
    #print('\nMAPE Dynamic: ',round(mape_dy,2),'%')
    mape_one = mape_vectorized(endog[n_train+lag:],predict.predicted_mean[n_train:])
    print('MAPE One-Step-Ahead: ',round(mape_one,2),'%')
    
    
    #NAIVE - MEAN VALUE: evaluate if I just predict the average of the timeseries
    naive = pd.Series(np.repeat(endog[n_train+lag:].mean(),len(endog[n_train+lag:])))
    naive.index = endog[n_train+lag:].index
    
    naive_mse = mean_squared_error(endog[n_train+lag:],naive)
    print('\nNAIVE RMSE: ',round(np.sqrt(naive_mse),3))
    naive_mape = mape_vectorized(endog[n_train+lag:],naive)
    print('NAIVE MAPE: ',round(naive_mape,2),'%')
    
    RMSE_dy.append(round(np.sqrt(mse_dynamic),3))
    RMSE_one.append(round(np.sqrt(mse_one),3))

    MAPE_dy.append(round(mape_dy,2))
    MAPE_one.append(round(mape_one,2))

    NAIVE_rmse.append(round(np.sqrt(naive_mse),2))
    NAIVE_mape.append(round(naive_mape,2))
    
report02_monthly = pd.DataFrame()
report02_monthly['PARAMS'] = MODEL_ID
report02_monthly['VARS'] = VARS
report02_monthly['LAGS'] = LAGS
report02_monthly['AIC'] = MODEL_AIC
report02_monthly['NORM_TEST_p'] = NORM_MW
report02_monthly['RESID.'] = RESID
report02_monthly['HET_TEST'] = HETERO
report02_monthly['RMSE_DY'] = RMSE_dy
report02_monthly['RMSE_ONE'] = RMSE_one
report02_monthly['MAPE_DY'] = MAPE_dy
report02_monthly['MAPE_ONE'] = MAPE_one
report02_monthly['NAIVE_RMSE'] = NAIVE_rmse
report02_monthly['NAIVE_MAPE'] = NAIVE_mape

os.chdir('C:/Users/trbella/Desktop/Thiago/Python/SARIMAX_GRID_LAGS/results/monthly_rev01')
report02_monthly.to_excel('report02_monthly_rev01.xlsx')

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

#================================
#PARTE 03
#================================
#==================================================================
# PACKAGES
#==================================================================
import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import urllib.request, json
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as np
import seaborn as sns
from statsmodels.stats.diagnostic import acorr_ljungbox as lbox
from statsmodels.graphics.tsaplots import plot_acf 
from statsmodels.graphics.tsaplots import plot_pacf 
import os
import gc


#==================================================================
# series_to_supervised FUNCTION - author - Jason Brownlee
#==================================================================
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
#==================================================================
# MAPE FUNCTION
#==================================================================
def mape_vectorized(true, pred): 
    mask = true != 0
    return (np.fabs(true - pred)/true)[mask].mean()*100

#Variables for dataset creation with grid search results
MODEL_ID = []
VARS = []
LAGS = []
MODEL_AIC = []
NORM_MW = []
RESID = []
HETERO = []
RMSE_dy = []
RMSE_one = []
MAPE_dy = []
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
# Variables
#================================================
endog = df.loc[:]['all']
exog = df.loc[:][variables]
endog = endog.asfreq('D')
exog = exog.asfreq('D')
#nobs = endog.shape[0]

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))

endog = endog.resample('M',label='right').sum()
exog = exog.resample('M',label='right').sum()

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
n_train = int(0.8*len(exog))
endog_train = endog.iloc[:n_train].asfreq('M')
endog_test = endog.iloc[n_train:].asfreq('M')
exog_train = exog.iloc[:n_train].asfreq('M')
exog_test = exog.iloc[n_train:].asfreq('M')

lags = list(range(1,4))
for lag in lags:
    VARS.append(variables)
    LAGS.append(lag)
    #================================================
    #Supervised Learning
    #================================================
    # load dataset
    train_np = exog_train[variables].values
    # ensure all data is float
    np.set_printoptions(suppress=True) #supress scientific notation
    train_np = train_np.astype('float32')
    
    # load dataset
    test_np = exog_test[variables].values
    # ensure all data is float
    np.set_printoptions(suppress=True) #supress scientific notation
    test_np = test_np.astype('float32')
    
    # specify the number of lag (hours in this case, since each line equals a hour)
    #n_features is the number of the exogenous variables
    n_hours = lag
    n_features = len(exog[variables].columns)
    
    # frame as supervised learning
    
    #Here I used scaled[:,1:] because I exlcuded the y (deaths all) variable from the function below.
    reframed = series_to_supervised(train_np[:,:], lag, 0)
    reframed_test = series_to_supervised(test_np[:,:], lag, 0)
    reframed_all = series_to_supervised(exog[variables], lag, 0)
    print(reframed.shape) #nº observation x ((nºfeatures + 1) * nºlags)
    
    reframed.head()
        
    #================================================
    # Automatic Model Fit - pmdarima
    #================================================
    
    #See Tips and Tricks: https://alkaline-ml.com/pmdarima/tips_and_tricks.html
    
    # SARIMAX Model
    print('PMDARIMA AUTO-MODEL')
    sxmodel = pm.auto_arima(endog_train[lag:],reframed, start_p=0, start_q=0, max_p=lag , max_q=lag ,
                               start_P=0,start_Q=0, max_P=lag ,max_Q=lag , m=12, seasonal=True,
                            trace=True, error_action='trace',suppress_warnings=True, stepwise=True,method='lbfgs')
    
    print(sxmodel.summary())
    
    sxmodel.plot_diagnostics(figsize=(10,5))
    plt.show()
    
    #check if the best model has trend
    a = pd.Series(sxmodel.pvalues()).index
    b = 'intercept'
    if len([i for i in a if i in b])>=1:
      inter = 'int.'
      trend = 'c'
    else:
      trend = None
      inter = None
      
    model_id = f'{sxmodel.order} {sxmodel.seasonal_order} {inter}'
    MODEL_ID.append(model_id)
  
    model_aic = sxmodel.aic()
    MODEL_AIC.append(round(model_aic,3))

    RESID.append(round(sxmodel.resid().mean(),5))
    
    #================================================
    #Statsmodels model - training set
    #From here below I will use Statsmodels instead of pmdarima
    #================================================
    print('STATSMODELS SARIMAX')
    mod = SARIMAX(endog_train[lag:].values, reframed[:], order=sxmodel.order,seasonal_order= sxmodel.seasonal_order,trend=trend)
    #mod = SARIMAX(endog_train[lag:].values, reframed[['var1(t-5)','var1(t-3)']][:], order=(5,1,0),seasonal_order= (0,0,1,7),trend=None)
    
    fit_res = mod.fit(maxiter=500,method='bfgs',disp=0) #disp controls verbosity
    
    print('\033[1m'+'Residuals mean ='+'\033[0m', round(fit_res.resid.mean(),5))
    from scipy.stats import mannwhitneyu as mwu
    normal = np.random.normal(0,fit_res.resid.std(),len(fit_res.resid))
    mann = mwu(normal,fit_res.resid,alternative = 'two-sided')
    print('\033[4m'+'\nTheoretical normal vs Residuals - hypothesis test:'+'\033[0m')
    print('\033[1m'+'Mann-Whitney U p-value:'+'\033[0m', round(mann[1],5))
    print(fit_res.summary())
    fit_res.plot_diagnostics(figsize=(10,5))
    plt.show()
    
    NORM_MW.append(round(mann[1],4))
    
    #================================================
    #Heteroscedasticity - Breusch-Pagan Lagrange Multiplier test
    #================================================
    #Heteroskedasticity is indicated if p <0.05
    from statsmodels.stats.diagnostic import het_breuschpagan
    bp_test = het_breuschpagan(fit_res.resid[0:], reframed[0:])
    #values: 'LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value'
    hetero = bp_test[1]
    
    HETERO.append(hetero)
    
    #================================================
    #Statsmodels model - all data
    #================================================
    #Here we will fit the model with all data, but just in order to do the 
    #one-step-ahead prediction. The model coefficients will be the same found 
    #using only the training set.
    
    mod = SARIMAX(endog[lag:], exog=reframed_all, order=sxmodel.order,seasonal_order=sxmodel.seasonal_order,trend=trend)
    #mod = SARIMAX(endog[lag:], exog=reframed_all[['var1(t-5)','var1(t-3)']][:], order=(5,1,0),seasonal_order= (0,0,1,7),trend=None)
    res = mod.filter(fit_res.params) #estimated parameters of training set
    print(res.summary())
    
    #================================================
    #Prediction and Forecast
    #================================================
    # In-sample one-step-ahead predictions
    predict = res.get_prediction()
    predict_ci = predict.conf_int()
    
    # Dynamic predictions
    predict_dy = res.get_prediction(dynamic=endog.iloc[:n_train].index[-1])
    predict_dy_ci = predict_dy.conf_int()
    
    #================================================
    #Root Mean Squared Error (RMSE) - Test set
    #================================================
    from sklearn.metrics import mean_squared_error
    mse_dynamic = mean_squared_error(endog[n_train+lag:],predict_dy.predicted_mean[n_train:])
    #print('RMSE Dynamic: ',round(np.sqrt(mse_dynamic),3))
    mse_one = mean_squared_error(endog[n_train+lag:],predict.predicted_mean[n_train:])
    print('RMSE One-Step-Ahead: ',round(np.sqrt(mse_one),3))
    
    mape_dy = mape_vectorized(endog[n_train+lag:],predict_dy.predicted_mean[n_train:])
    #print('\nMAPE Dynamic: ',round(mape_dy,2),'%')
    mape_one = mape_vectorized(endog[n_train+lag:],predict.predicted_mean[n_train:])
    print('MAPE One-Step-Ahead: ',round(mape_one,2),'%')
    
    
    #NAIVE - MEAN VALUE: evaluate if I just predict the average of the timeseries
    naive = pd.Series(np.repeat(endog[n_train+lag:].mean(),len(endog[n_train+lag:])))
    naive.index = endog[n_train+lag:].index
    
    naive_mse = mean_squared_error(endog[n_train+lag:],naive)
    print('\nNAIVE RMSE: ',round(np.sqrt(naive_mse),3))
    naive_mape = mape_vectorized(endog[n_train+lag:],naive)
    print('NAIVE MAPE: ',round(naive_mape,2),'%')
    
    RMSE_dy.append(round(np.sqrt(mse_dynamic),3))
    RMSE_one.append(round(np.sqrt(mse_one),3))

    MAPE_dy.append(round(mape_dy,2))
    MAPE_one.append(round(mape_one,2))

    NAIVE_rmse.append(round(np.sqrt(naive_mse),2))
    NAIVE_mape.append(round(naive_mape,2))
    
report03_monthly = pd.DataFrame()
report03_monthly['PARAMS'] = MODEL_ID
report03_monthly['VARS'] = VARS
report03_monthly['LAGS'] = LAGS
report03_monthly['AIC'] = MODEL_AIC
report03_monthly['NORM_TEST_p'] = NORM_MW
report03_monthly['RESID.'] = RESID
report03_monthly['HET_TEST'] = HETERO
report03_monthly['RMSE_DY'] = RMSE_dy
report03_monthly['RMSE_ONE'] = RMSE_one
report03_monthly['MAPE_DY'] = MAPE_dy
report03_monthly['MAPE_ONE'] = MAPE_one
report03_monthly['NAIVE_RMSE'] = NAIVE_rmse
report03_monthly['NAIVE_MAPE'] = NAIVE_mape

os.chdir('C:/Users/trbella/Desktop/Thiago/Python/SARIMAX_GRID_LAGS/results/monthly_rev01')
report03_monthly.to_excel('report03_monthly_rev01.xlsx')

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

#================================
#PARTE 04
#================================
#==================================================================
# PACKAGES
#==================================================================
import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import urllib.request, json
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as np
import seaborn as sns
from statsmodels.stats.diagnostic import acorr_ljungbox as lbox
from statsmodels.graphics.tsaplots import plot_acf 
from statsmodels.graphics.tsaplots import plot_pacf 
import os
import gc


#==================================================================
# series_to_supervised FUNCTION - author - Jason Brownlee
#==================================================================
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
#==================================================================
# MAPE FUNCTION
#==================================================================
def mape_vectorized(true, pred): 
    mask = true != 0
    return (np.fabs(true - pred)/true)[mask].mean()*100

#Variables for dataset creation with grid search results
MODEL_ID = []
VARS = []
LAGS = []
MODEL_AIC = []
NORM_MW = []
RESID = []
HETERO = []
RMSE_dy = []
RMSE_one = []
MAPE_dy = []
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
# Variables
#================================================
endog = df.loc[:]['all']
exog = df.loc[:][variables]
endog = endog.asfreq('D')
exog = exog.asfreq('D')
#nobs = endog.shape[0]

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))

endog = endog.resample('M',label='right').sum()
exog = exog.resample('M',label='right').sum()

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
n_train = int(0.8*len(exog))
endog_train = endog.iloc[:n_train].asfreq('M')
endog_test = endog.iloc[n_train:].asfreq('M')
exog_train = exog.iloc[:n_train].asfreq('M')
exog_test = exog.iloc[n_train:].asfreq('M')

lags = list(range(1,4))
for lag in lags:
    VARS.append(variables)
    LAGS.append(lag)
    #================================================
    #Supervised Learning
    #================================================
    # load dataset
    train_np = exog_train[variables].values
    # ensure all data is float
    np.set_printoptions(suppress=True) #supress scientific notation
    train_np = train_np.astype('float32')
    
    # load dataset
    test_np = exog_test[variables].values
    # ensure all data is float
    np.set_printoptions(suppress=True) #supress scientific notation
    test_np = test_np.astype('float32')
    
    # specify the number of lag (hours in this case, since each line equals a hour)
    #n_features is the number of the exogenous variables
    n_hours = lag
    n_features = len(exog[variables].columns)
    
    # frame as supervised learning
    
    #Here I used scaled[:,1:] because I exlcuded the y (deaths all) variable from the function below.
    reframed = series_to_supervised(train_np[:,:], lag, 0)
    reframed_test = series_to_supervised(test_np[:,:], lag, 0)
    reframed_all = series_to_supervised(exog[variables], lag, 0)
    print(reframed.shape) #nº observation x ((nºfeatures + 1) * nºlags)
    
    reframed.head()
        
    #================================================
    # Automatic Model Fit - pmdarima
    #================================================
    
    #See Tips and Tricks: https://alkaline-ml.com/pmdarima/tips_and_tricks.html
    
    # SARIMAX Model
    print('PMDARIMA AUTO-MODEL')
    sxmodel = pm.auto_arima(endog_train[lag:],reframed, start_p=0, start_q=0, max_p=lag , max_q=lag ,
                               start_P=0,start_Q=0, max_P=lag ,max_Q=lag , m=12, seasonal=True,
                            trace=True, error_action='trace',suppress_warnings=True, stepwise=True,method='lbfgs')
    
    print(sxmodel.summary())
    
    sxmodel.plot_diagnostics(figsize=(10,5))
    plt.show()
    
    #check if the best model has trend
    a = pd.Series(sxmodel.pvalues()).index
    b = 'intercept'
    if len([i for i in a if i in b])>=1:
      inter = 'int.'
      trend = 'c'
    else:
      trend = None
      inter = None
      
    model_id = f'{sxmodel.order} {sxmodel.seasonal_order} {inter}'
    MODEL_ID.append(model_id)
  
    model_aic = sxmodel.aic()
    MODEL_AIC.append(round(model_aic,3))

    RESID.append(round(sxmodel.resid().mean(),5))
    
    #================================================
    #Statsmodels model - training set
    #From here below I will use Statsmodels instead of pmdarima
    #================================================
    print('STATSMODELS SARIMAX')
    mod = SARIMAX(endog_train[lag:].values, reframed[:], order=sxmodel.order,seasonal_order= sxmodel.seasonal_order,trend=trend)
    #mod = SARIMAX(endog_train[lag:].values, reframed[['var1(t-5)','var1(t-3)']][:], order=(5,1,0),seasonal_order= (0,0,1,7),trend=None)
    
    fit_res = mod.fit(maxiter=500,method='bfgs',disp=0) #disp controls verbosity
    
    print('\033[1m'+'Residuals mean ='+'\033[0m', round(fit_res.resid.mean(),5))
    from scipy.stats import mannwhitneyu as mwu
    normal = np.random.normal(0,fit_res.resid.std(),len(fit_res.resid))
    mann = mwu(normal,fit_res.resid,alternative = 'two-sided')
    print('\033[4m'+'\nTheoretical normal vs Residuals - hypothesis test:'+'\033[0m')
    print('\033[1m'+'Mann-Whitney U p-value:'+'\033[0m', round(mann[1],5))
    print(fit_res.summary())
    fit_res.plot_diagnostics(figsize=(10,5))
    plt.show()
    
    NORM_MW.append(round(mann[1],4))
    
    #================================================
    #Heteroscedasticity - Breusch-Pagan Lagrange Multiplier test
    #================================================
    #Heteroskedasticity is indicated if p <0.05
    from statsmodels.stats.diagnostic import het_breuschpagan
    bp_test = het_breuschpagan(fit_res.resid[0:], reframed[0:])
    #values: 'LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value'
    hetero = bp_test[1]
    
    HETERO.append(hetero)
    
    #================================================
    #Statsmodels model - all data
    #================================================
    #Here we will fit the model with all data, but just in order to do the 
    #one-step-ahead prediction. The model coefficients will be the same found 
    #using only the training set.
    
    mod = SARIMAX(endog[lag:], exog=reframed_all, order=sxmodel.order,seasonal_order=sxmodel.seasonal_order,trend=trend)
    #mod = SARIMAX(endog[lag:], exog=reframed_all[['var1(t-5)','var1(t-3)']][:], order=(5,1,0),seasonal_order= (0,0,1,7),trend=None)
    res = mod.filter(fit_res.params) #estimated parameters of training set
    print(res.summary())
    
    #================================================
    #Prediction and Forecast
    #================================================
    # In-sample one-step-ahead predictions
    predict = res.get_prediction()
    predict_ci = predict.conf_int()
    
    # Dynamic predictions
    predict_dy = res.get_prediction(dynamic=endog.iloc[:n_train].index[-1])
    predict_dy_ci = predict_dy.conf_int()
    
    #================================================
    #Root Mean Squared Error (RMSE) - Test set
    #================================================
    from sklearn.metrics import mean_squared_error
    mse_dynamic = mean_squared_error(endog[n_train+lag:],predict_dy.predicted_mean[n_train:])
    #print('RMSE Dynamic: ',round(np.sqrt(mse_dynamic),3))
    mse_one = mean_squared_error(endog[n_train+lag:],predict.predicted_mean[n_train:])
    print('RMSE One-Step-Ahead: ',round(np.sqrt(mse_one),3))
    
    mape_dy = mape_vectorized(endog[n_train+lag:],predict_dy.predicted_mean[n_train:])
    #print('\nMAPE Dynamic: ',round(mape_dy,2),'%')
    mape_one = mape_vectorized(endog[n_train+lag:],predict.predicted_mean[n_train:])
    print('MAPE One-Step-Ahead: ',round(mape_one,2),'%')
    
    
    #NAIVE - MEAN VALUE: evaluate if I just predict the average of the timeseries
    naive = pd.Series(np.repeat(endog[n_train+lag:].mean(),len(endog[n_train+lag:])))
    naive.index = endog[n_train+lag:].index
    
    naive_mse = mean_squared_error(endog[n_train+lag:],naive)
    print('\nNAIVE RMSE: ',round(np.sqrt(naive_mse),3))
    naive_mape = mape_vectorized(endog[n_train+lag:],naive)
    print('NAIVE MAPE: ',round(naive_mape,2),'%')
    
    RMSE_dy.append(round(np.sqrt(mse_dynamic),3))
    RMSE_one.append(round(np.sqrt(mse_one),3))

    MAPE_dy.append(round(mape_dy,2))
    MAPE_one.append(round(mape_one,2))

    NAIVE_rmse.append(round(np.sqrt(naive_mse),2))
    NAIVE_mape.append(round(naive_mape,2))
    
report04_monthly = pd.DataFrame()
report04_monthly['PARAMS'] = MODEL_ID
report04_monthly['VARS'] = VARS
report04_monthly['LAGS'] = LAGS
report04_monthly['AIC'] = MODEL_AIC
report04_monthly['NORM_TEST_p'] = NORM_MW
report04_monthly['RESID.'] = RESID
report04_monthly['HET_TEST'] = HETERO
report04_monthly['RMSE_DY'] = RMSE_dy
report04_monthly['RMSE_ONE'] = RMSE_one
report04_monthly['MAPE_DY'] = MAPE_dy
report04_monthly['MAPE_ONE'] = MAPE_one
report04_monthly['NAIVE_RMSE'] = NAIVE_rmse
report04_monthly['NAIVE_MAPE'] = NAIVE_mape

os.chdir('C:/Users/trbella/Desktop/Thiago/Python/SARIMAX_GRID_LAGS/results/monthly_rev01')
report04_monthly.to_excel('report04_monthly_rev01.xlsx')

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

#================================
#PARTE 05
#================================
#==================================================================
# PACKAGES
#==================================================================
import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import urllib.request, json
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as np
import seaborn as sns
from statsmodels.stats.diagnostic import acorr_ljungbox as lbox
from statsmodels.graphics.tsaplots import plot_acf 
from statsmodels.graphics.tsaplots import plot_pacf 
import os
import gc


#==================================================================
# series_to_supervised FUNCTION - author - Jason Brownlee
#==================================================================
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
#==================================================================
# MAPE FUNCTION
#==================================================================
def mape_vectorized(true, pred): 
    mask = true != 0
    return (np.fabs(true - pred)/true)[mask].mean()*100

#Variables for dataset creation with grid search results
MODEL_ID = []
VARS = []
LAGS = []
MODEL_AIC = []
NORM_MW = []
RESID = []
HETERO = []
RMSE_dy = []
RMSE_one = []
MAPE_dy = []
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
# Variables
#================================================
endog = df.loc[:]['all']
exog = df.loc[:][variables]
endog = endog.asfreq('D')
exog = exog.asfreq('D')
#nobs = endog.shape[0]

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))

endog = endog.resample('M',label='right').sum()
exog = exog.resample('M',label='right').sum()

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
n_train = int(0.8*len(exog))
endog_train = endog.iloc[:n_train].asfreq('M')
endog_test = endog.iloc[n_train:].asfreq('M')
exog_train = exog.iloc[:n_train].asfreq('M')
exog_test = exog.iloc[n_train:].asfreq('M')

lags = list(range(1,4))
for lag in lags:
    VARS.append(variables)
    LAGS.append(lag)
    #================================================
    #Supervised Learning
    #================================================
    # load dataset
    train_np = exog_train[variables].values
    # ensure all data is float
    np.set_printoptions(suppress=True) #supress scientific notation
    train_np = train_np.astype('float32')
    
    # load dataset
    test_np = exog_test[variables].values
    # ensure all data is float
    np.set_printoptions(suppress=True) #supress scientific notation
    test_np = test_np.astype('float32')
    
    # specify the number of lag (hours in this case, since each line equals a hour)
    #n_features is the number of the exogenous variables
    n_hours = lag
    n_features = len(exog[variables].columns)
    
    # frame as supervised learning
    
    #Here I used scaled[:,1:] because I exlcuded the y (deaths all) variable from the function below.
    reframed = series_to_supervised(train_np[:,:], lag, 0)
    reframed_test = series_to_supervised(test_np[:,:], lag, 0)
    reframed_all = series_to_supervised(exog[variables], lag, 0)
    print(reframed.shape) #nº observation x ((nºfeatures + 1) * nºlags)
    
    reframed.head()
        
    #================================================
    # Automatic Model Fit - pmdarima
    #================================================
    
    #See Tips and Tricks: https://alkaline-ml.com/pmdarima/tips_and_tricks.html
    
    # SARIMAX Model
    print('PMDARIMA AUTO-MODEL')
    sxmodel = pm.auto_arima(endog_train[lag:],reframed, start_p=0, start_q=0, max_p=lag , max_q=lag ,
                               start_P=0,start_Q=0, max_P=lag ,max_Q=lag , m=12, seasonal=True,
                            trace=True, error_action='trace',suppress_warnings=True, stepwise=True,method='lbfgs')
    
    print(sxmodel.summary())
    
    sxmodel.plot_diagnostics(figsize=(10,5))
    plt.show()
    
    #check if the best model has trend
    a = pd.Series(sxmodel.pvalues()).index
    b = 'intercept'
    if len([i for i in a if i in b])>=1:
      inter = 'int.'
      trend = 'c'
    else:
      trend = None
      inter = None
      
    model_id = f'{sxmodel.order} {sxmodel.seasonal_order} {inter}'
    MODEL_ID.append(model_id)
  
    model_aic = sxmodel.aic()
    MODEL_AIC.append(round(model_aic,3))

    RESID.append(round(sxmodel.resid().mean(),5))
    
    #================================================
    #Statsmodels model - training set
    #From here below I will use Statsmodels instead of pmdarima
    #================================================
    print('STATSMODELS SARIMAX')
    mod = SARIMAX(endog_train[lag:].values, reframed[:], order=sxmodel.order,seasonal_order= sxmodel.seasonal_order,trend=trend)
    #mod = SARIMAX(endog_train[lag:].values, reframed[['var1(t-5)','var1(t-3)']][:], order=(5,1,0),seasonal_order= (0,0,1,7),trend=None)
    
    fit_res = mod.fit(maxiter=500,method='bfgs',disp=0) #disp controls verbosity
    
    print('\033[1m'+'Residuals mean ='+'\033[0m', round(fit_res.resid.mean(),5))
    from scipy.stats import mannwhitneyu as mwu
    normal = np.random.normal(0,fit_res.resid.std(),len(fit_res.resid))
    mann = mwu(normal,fit_res.resid,alternative = 'two-sided')
    print('\033[4m'+'\nTheoretical normal vs Residuals - hypothesis test:'+'\033[0m')
    print('\033[1m'+'Mann-Whitney U p-value:'+'\033[0m', round(mann[1],5))
    print(fit_res.summary())
    fit_res.plot_diagnostics(figsize=(10,5))
    plt.show()
    
    NORM_MW.append(round(mann[1],4))
    
    #================================================
    #Heteroscedasticity - Breusch-Pagan Lagrange Multiplier test
    #================================================
    #Heteroskedasticity is indicated if p <0.05
    from statsmodels.stats.diagnostic import het_breuschpagan
    bp_test = het_breuschpagan(fit_res.resid[0:], reframed[0:])
    #values: 'LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value'
    hetero = bp_test[1]
    
    HETERO.append(hetero)
    
    #================================================
    #Statsmodels model - all data
    #================================================
    #Here we will fit the model with all data, but just in order to do the 
    #one-step-ahead prediction. The model coefficients will be the same found 
    #using only the training set.
    
    mod = SARIMAX(endog[lag:], exog=reframed_all, order=sxmodel.order,seasonal_order=sxmodel.seasonal_order,trend=trend)
    #mod = SARIMAX(endog[lag:], exog=reframed_all[['var1(t-5)','var1(t-3)']][:], order=(5,1,0),seasonal_order= (0,0,1,7),trend=None)
    res = mod.filter(fit_res.params) #estimated parameters of training set
    print(res.summary())
    
    #================================================
    #Prediction and Forecast
    #================================================
    # In-sample one-step-ahead predictions
    predict = res.get_prediction()
    predict_ci = predict.conf_int()
    
    # Dynamic predictions
    predict_dy = res.get_prediction(dynamic=endog.iloc[:n_train].index[-1])
    predict_dy_ci = predict_dy.conf_int()
    
    #================================================
    #Root Mean Squared Error (RMSE) - Test set
    #================================================
    from sklearn.metrics import mean_squared_error
    mse_dynamic = mean_squared_error(endog[n_train+lag:],predict_dy.predicted_mean[n_train:])
    #print('RMSE Dynamic: ',round(np.sqrt(mse_dynamic),3))
    mse_one = mean_squared_error(endog[n_train+lag:],predict.predicted_mean[n_train:])
    print('RMSE One-Step-Ahead: ',round(np.sqrt(mse_one),3))
    
    mape_dy = mape_vectorized(endog[n_train+lag:],predict_dy.predicted_mean[n_train:])
    #print('\nMAPE Dynamic: ',round(mape_dy,2),'%')
    mape_one = mape_vectorized(endog[n_train+lag:],predict.predicted_mean[n_train:])
    print('MAPE One-Step-Ahead: ',round(mape_one,2),'%')
    
    
    #NAIVE - MEAN VALUE: evaluate if I just predict the average of the timeseries
    naive = pd.Series(np.repeat(endog[n_train+lag:].mean(),len(endog[n_train+lag:])))
    naive.index = endog[n_train+lag:].index
    
    naive_mse = mean_squared_error(endog[n_train+lag:],naive)
    print('\nNAIVE RMSE: ',round(np.sqrt(naive_mse),3))
    naive_mape = mape_vectorized(endog[n_train+lag:],naive)
    print('NAIVE MAPE: ',round(naive_mape,2),'%')
    
    RMSE_dy.append(round(np.sqrt(mse_dynamic),3))
    RMSE_one.append(round(np.sqrt(mse_one),3))

    MAPE_dy.append(round(mape_dy,2))
    MAPE_one.append(round(mape_one,2))

    NAIVE_rmse.append(round(np.sqrt(naive_mse),2))
    NAIVE_mape.append(round(naive_mape,2))
    
report05_monthly = pd.DataFrame()
report05_monthly['PARAMS'] = MODEL_ID
report05_monthly['VARS'] = VARS
report05_monthly['LAGS'] = LAGS
report05_monthly['AIC'] = MODEL_AIC
report05_monthly['NORM_TEST_p'] = NORM_MW
report05_monthly['RESID.'] = RESID
report05_monthly['HET_TEST'] = HETERO
report05_monthly['RMSE_DY'] = RMSE_dy
report05_monthly['RMSE_ONE'] = RMSE_one
report05_monthly['MAPE_DY'] = MAPE_dy
report05_monthly['MAPE_ONE'] = MAPE_one
report05_monthly['NAIVE_RMSE'] = NAIVE_rmse
report05_monthly['NAIVE_MAPE'] = NAIVE_mape

os.chdir('C:/Users/trbella/Desktop/Thiago/Python/SARIMAX_GRID_LAGS/results/monthly_rev01')
report05_monthly.to_excel('report05_monthly_rev01.xlsx')

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


#================================
#PARTE 06
#================================
#==================================================================
# PACKAGES
#==================================================================
import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import urllib.request, json
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as np
import seaborn as sns
from statsmodels.stats.diagnostic import acorr_ljungbox as lbox
from statsmodels.graphics.tsaplots import plot_acf 
from statsmodels.graphics.tsaplots import plot_pacf 
import os
import gc


#==================================================================
# series_to_supervised FUNCTION - author - Jason Brownlee
#==================================================================
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
#==================================================================
# MAPE FUNCTION
#==================================================================
def mape_vectorized(true, pred): 
    mask = true != 0
    return (np.fabs(true - pred)/true)[mask].mean()*100

#Variables for dataset creation with grid search results
MODEL_ID = []
VARS = []
LAGS = []
MODEL_AIC = []
NORM_MW = []
RESID = []
HETERO = []
RMSE_dy = []
RMSE_one = []
MAPE_dy = []
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
# Variables
#================================================
endog = df.loc[:]['all']
exog = df.loc[:][variables]
endog = endog.asfreq('D')
exog = exog.asfreq('D')
#nobs = endog.shape[0]

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))

endog = endog.resample('M',label='right').sum()
exog = exog.resample('M',label='right').sum()

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
n_train = int(0.8*len(exog))
endog_train = endog.iloc[:n_train].asfreq('M')
endog_test = endog.iloc[n_train:].asfreq('M')
exog_train = exog.iloc[:n_train].asfreq('M')
exog_test = exog.iloc[n_train:].asfreq('M')

lags = list(range(1,4))
for lag in lags:
    VARS.append(variables)
    LAGS.append(lag)
    #================================================
    #Supervised Learning
    #================================================
    # load dataset
    train_np = exog_train[variables].values
    # ensure all data is float
    np.set_printoptions(suppress=True) #supress scientific notation
    train_np = train_np.astype('float32')
    
    # load dataset
    test_np = exog_test[variables].values
    # ensure all data is float
    np.set_printoptions(suppress=True) #supress scientific notation
    test_np = test_np.astype('float32')
    
    # specify the number of lag (hours in this case, since each line equals a hour)
    #n_features is the number of the exogenous variables
    n_hours = lag
    n_features = len(exog[variables].columns)
    
    # frame as supervised learning
    
    #Here I used scaled[:,1:] because I exlcuded the y (deaths all) variable from the function below.
    reframed = series_to_supervised(train_np[:,:], lag, 0)
    reframed_test = series_to_supervised(test_np[:,:], lag, 0)
    reframed_all = series_to_supervised(exog[variables], lag, 0)
    print(reframed.shape) #nº observation x ((nºfeatures + 1) * nºlags)
    
    reframed.head()
        
    #================================================
    # Automatic Model Fit - pmdarima
    #================================================
    
    #See Tips and Tricks: https://alkaline-ml.com/pmdarima/tips_and_tricks.html
    
    # SARIMAX Model
    print('PMDARIMA AUTO-MODEL')
    sxmodel = pm.auto_arima(endog_train[lag:],reframed, start_p=0, start_q=0, max_p=lag , max_q=lag ,
                               start_P=0,start_Q=0, max_P=lag ,max_Q=lag , m=12, seasonal=True,
                            trace=True, error_action='trace',suppress_warnings=True, stepwise=True,method='lbfgs')
    
    print(sxmodel.summary())
    
    sxmodel.plot_diagnostics(figsize=(10,5))
    plt.show()
    
    #check if the best model has trend
    a = pd.Series(sxmodel.pvalues()).index
    b = 'intercept'
    if len([i for i in a if i in b])>=1:
      inter = 'int.'
      trend = 'c'
    else:
      trend = None
      inter = None
      
    model_id = f'{sxmodel.order} {sxmodel.seasonal_order} {inter}'
    MODEL_ID.append(model_id)
  
    model_aic = sxmodel.aic()
    MODEL_AIC.append(round(model_aic,3))

    RESID.append(round(sxmodel.resid().mean(),5))
    
    #================================================
    #Statsmodels model - training set
    #From here below I will use Statsmodels instead of pmdarima
    #================================================
    print('STATSMODELS SARIMAX')
    mod = SARIMAX(endog_train[lag:].values, reframed[:], order=sxmodel.order,seasonal_order= sxmodel.seasonal_order,trend=trend)
    #mod = SARIMAX(endog_train[lag:].values, reframed[['var1(t-5)','var1(t-3)']][:], order=(5,1,0),seasonal_order= (0,0,1,7),trend=None)
    
    fit_res = mod.fit(maxiter=500,method='bfgs',disp=0) #disp controls verbosity
    
    print('\033[1m'+'Residuals mean ='+'\033[0m', round(fit_res.resid.mean(),5))
    from scipy.stats import mannwhitneyu as mwu
    normal = np.random.normal(0,fit_res.resid.std(),len(fit_res.resid))
    mann = mwu(normal,fit_res.resid,alternative = 'two-sided')
    print('\033[4m'+'\nTheoretical normal vs Residuals - hypothesis test:'+'\033[0m')
    print('\033[1m'+'Mann-Whitney U p-value:'+'\033[0m', round(mann[1],5))
    print(fit_res.summary())
    fit_res.plot_diagnostics(figsize=(10,5))
    plt.show()
    
    NORM_MW.append(round(mann[1],4))
    
    #================================================
    #Heteroscedasticity - Breusch-Pagan Lagrange Multiplier test
    #================================================
    #Heteroskedasticity is indicated if p <0.05
    from statsmodels.stats.diagnostic import het_breuschpagan
    bp_test = het_breuschpagan(fit_res.resid[0:], reframed[0:])
    #values: 'LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value'
    hetero = bp_test[1]
    
    HETERO.append(hetero)
    
    #================================================
    #Statsmodels model - all data
    #================================================
    #Here we will fit the model with all data, but just in order to do the 
    #one-step-ahead prediction. The model coefficients will be the same found 
    #using only the training set.
    
    mod = SARIMAX(endog[lag:], exog=reframed_all, order=sxmodel.order,seasonal_order=sxmodel.seasonal_order,trend=trend)
    #mod = SARIMAX(endog[lag:], exog=reframed_all[['var1(t-5)','var1(t-3)']][:], order=(5,1,0),seasonal_order= (0,0,1,7),trend=None)
    res = mod.filter(fit_res.params) #estimated parameters of training set
    print(res.summary())
    
    #================================================
    #Prediction and Forecast
    #================================================
    # In-sample one-step-ahead predictions
    predict = res.get_prediction()
    predict_ci = predict.conf_int()
    
    # Dynamic predictions
    predict_dy = res.get_prediction(dynamic=endog.iloc[:n_train].index[-1])
    predict_dy_ci = predict_dy.conf_int()
    
    #================================================
    #Root Mean Squared Error (RMSE) - Test set
    #================================================
    from sklearn.metrics import mean_squared_error
    mse_dynamic = mean_squared_error(endog[n_train+lag:],predict_dy.predicted_mean[n_train:])
    #print('RMSE Dynamic: ',round(np.sqrt(mse_dynamic),3))
    mse_one = mean_squared_error(endog[n_train+lag:],predict.predicted_mean[n_train:])
    print('RMSE One-Step-Ahead: ',round(np.sqrt(mse_one),3))
    
    mape_dy = mape_vectorized(endog[n_train+lag:],predict_dy.predicted_mean[n_train:])
    #print('\nMAPE Dynamic: ',round(mape_dy,2),'%')
    mape_one = mape_vectorized(endog[n_train+lag:],predict.predicted_mean[n_train:])
    print('MAPE One-Step-Ahead: ',round(mape_one,2),'%')
    
    
    #NAIVE - MEAN VALUE: evaluate if I just predict the average of the timeseries
    naive = pd.Series(np.repeat(endog[n_train+lag:].mean(),len(endog[n_train+lag:])))
    naive.index = endog[n_train+lag:].index
    
    naive_mse = mean_squared_error(endog[n_train+lag:],naive)
    print('\nNAIVE RMSE: ',round(np.sqrt(naive_mse),3))
    naive_mape = mape_vectorized(endog[n_train+lag:],naive)
    print('NAIVE MAPE: ',round(naive_mape,2),'%')
    
    RMSE_dy.append(round(np.sqrt(mse_dynamic),3))
    RMSE_one.append(round(np.sqrt(mse_one),3))

    MAPE_dy.append(round(mape_dy,2))
    MAPE_one.append(round(mape_one,2))

    NAIVE_rmse.append(round(np.sqrt(naive_mse),2))
    NAIVE_mape.append(round(naive_mape,2))
    
report06_monthly = pd.DataFrame()
report06_monthly['PARAMS'] = MODEL_ID
report06_monthly['VARS'] = VARS
report06_monthly['LAGS'] = LAGS
report06_monthly['AIC'] = MODEL_AIC
report06_monthly['NORM_TEST_p'] = NORM_MW
report06_monthly['RESID.'] = RESID
report06_monthly['HET_TEST'] = HETERO
report06_monthly['RMSE_DY'] = RMSE_dy
report06_monthly['RMSE_ONE'] = RMSE_one
report06_monthly['MAPE_DY'] = MAPE_dy
report06_monthly['MAPE_ONE'] = MAPE_one
report06_monthly['NAIVE_RMSE'] = NAIVE_rmse
report06_monthly['NAIVE_MAPE'] = NAIVE_mape

os.chdir('C:/Users/trbella/Desktop/Thiago/Python/SARIMAX_GRID_LAGS/results/monthly_rev01')
report06_monthly.to_excel('report06_monthly_rev01.xlsx')

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


#================================
#PARTE 07
#================================
#==================================================================
# PACKAGES
#==================================================================
import pandas as pd
import pmdarima as pm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import urllib.request, json
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
import numpy as np
import seaborn as sns
from statsmodels.stats.diagnostic import acorr_ljungbox as lbox
from statsmodels.graphics.tsaplots import plot_acf 
from statsmodels.graphics.tsaplots import plot_pacf 
import os
import gc


#==================================================================
# series_to_supervised FUNCTION - author - Jason Brownlee
#==================================================================
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
#==================================================================
# MAPE FUNCTION
#==================================================================
def mape_vectorized(true, pred): 
    mask = true != 0
    return (np.fabs(true - pred)/true)[mask].mean()*100

#Variables for dataset creation with grid search results
MODEL_ID = []
VARS = []
LAGS = []
MODEL_AIC = []
NORM_MW = []
RESID = []
HETERO = []
RMSE_dy = []
RMSE_one = []
MAPE_dy = []
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
#GRID 07
#================================================
variables = ['CO_MEAN']

#================================================
# Variables
#================================================
endog = df.loc[:]['all']
exog = df.loc[:][variables]
endog = endog.asfreq('D')
exog = exog.asfreq('D')
#nobs = endog.shape[0]

#Check NaN
print('NaN count\n')
print('Endog            ',np.sum(endog.isnull()))
print(np.sum(exog.isnull()))

endog = endog.resample('M',label='right').sum()
exog = exog.resample('M',label='right').sum()

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
n_train = int(0.8*len(exog))
endog_train = endog.iloc[:n_train].asfreq('M')
endog_test = endog.iloc[n_train:].asfreq('M')
exog_train = exog.iloc[:n_train].asfreq('M')
exog_test = exog.iloc[n_train:].asfreq('M')

lags = list(range(1,4))
for lag in lags:
    VARS.append(variables)
    LAGS.append(lag)
    #================================================
    #Supervised Learning
    #================================================
    # load dataset
    train_np = exog_train[variables].values
    # ensure all data is float
    np.set_printoptions(suppress=True) #supress scientific notation
    train_np = train_np.astype('float32')
    
    # load dataset
    test_np = exog_test[variables].values
    # ensure all data is float
    np.set_printoptions(suppress=True) #supress scientific notation
    test_np = test_np.astype('float32')
    
    # specify the number of lag (hours in this case, since each line equals a hour)
    #n_features is the number of the exogenous variables
    n_hours = lag
    n_features = len(exog[variables].columns)
    
    # frame as supervised learning
    
    #Here I used scaled[:,1:] because I exlcuded the y (deaths all) variable from the function below.
    reframed = series_to_supervised(train_np[:,:], lag, 0)
    reframed_test = series_to_supervised(test_np[:,:], lag, 0)
    reframed_all = series_to_supervised(exog[variables], lag, 0)
    print(reframed.shape) #nº observation x ((nºfeatures + 1) * nºlags)
    
    reframed.head()
        
    #================================================
    # Automatic Model Fit - pmdarima
    #================================================
    
    #See Tips and Tricks: https://alkaline-ml.com/pmdarima/tips_and_tricks.html
    
    # SARIMAX Model
    print('PMDARIMA AUTO-MODEL')
    sxmodel = pm.auto_arima(endog_train[lag:],reframed, start_p=0, start_q=0, max_p=lag , max_q=lag ,
                               start_P=0,start_Q=0, max_P=lag ,max_Q=lag , m=12, seasonal=True,
                            trace=True, error_action='trace',suppress_warnings=True, stepwise=True,method='lbfgs')
    
    print(sxmodel.summary())
    
    sxmodel.plot_diagnostics(figsize=(10,5))
    plt.show()
    
    #check if the best model has trend
    a = pd.Series(sxmodel.pvalues()).index
    b = 'intercept'
    if len([i for i in a if i in b])>=1:
      inter = 'int.'
      trend = 'c'
    else:
      trend = None
      inter = None
      
    model_id = f'{sxmodel.order} {sxmodel.seasonal_order} {inter}'
    MODEL_ID.append(model_id)
  
    model_aic = sxmodel.aic()
    MODEL_AIC.append(round(model_aic,3))

    RESID.append(round(sxmodel.resid().mean(),5))
    
    #================================================
    #Statsmodels model - training set
    #From here below I will use Statsmodels instead of pmdarima
    #================================================
    print('STATSMODELS SARIMAX')
    mod = SARIMAX(endog_train[lag:].values, reframed[:], order=sxmodel.order,seasonal_order= sxmodel.seasonal_order,trend=trend)
    #mod = SARIMAX(endog_train[lag:].values, reframed[['var1(t-5)','var1(t-3)']][:], order=(5,1,0),seasonal_order= (0,0,1,7),trend=None)
    
    fit_res = mod.fit(maxiter=500,method='bfgs',disp=0) #disp controls verbosity
    
    print('\033[1m'+'Residuals mean ='+'\033[0m', round(fit_res.resid.mean(),5))
    from scipy.stats import mannwhitneyu as mwu
    normal = np.random.normal(0,fit_res.resid.std(),len(fit_res.resid))
    mann = mwu(normal,fit_res.resid,alternative = 'two-sided')
    print('\033[4m'+'\nTheoretical normal vs Residuals - hypothesis test:'+'\033[0m')
    print('\033[1m'+'Mann-Whitney U p-value:'+'\033[0m', round(mann[1],5))
    print(fit_res.summary())
    fit_res.plot_diagnostics(figsize=(10,5))
    plt.show()
    
    NORM_MW.append(round(mann[1],4))
    
    #================================================
    #Heteroscedasticity - Breusch-Pagan Lagrange Multiplier test
    #================================================
    #Heteroskedasticity is indicated if p <0.05
    from statsmodels.stats.diagnostic import het_breuschpagan
    bp_test = het_breuschpagan(fit_res.resid[0:], reframed[0:])
    #values: 'LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value'
    hetero = bp_test[1]
    
    HETERO.append(hetero)
    
    #================================================
    #Statsmodels model - all data
    #================================================
    #Here we will fit the model with all data, but just in order to do the 
    #one-step-ahead prediction. The model coefficients will be the same found 
    #using only the training set.
    
    mod = SARIMAX(endog[lag:], exog=reframed_all, order=sxmodel.order,seasonal_order=sxmodel.seasonal_order,trend=trend)
    #mod = SARIMAX(endog[lag:], exog=reframed_all[['var1(t-5)','var1(t-3)']][:], order=(5,1,0),seasonal_order= (0,0,1,7),trend=None)
    res = mod.filter(fit_res.params) #estimated parameters of training set
    print(res.summary())
    
    #================================================
    #Prediction and Forecast
    #================================================
    # In-sample one-step-ahead predictions
    predict = res.get_prediction()
    predict_ci = predict.conf_int()
    
    # Dynamic predictions
    predict_dy = res.get_prediction(dynamic=endog.iloc[:n_train].index[-1])
    predict_dy_ci = predict_dy.conf_int()
    
    #================================================
    #Root Mean Squared Error (RMSE) - Test set
    #================================================
    from sklearn.metrics import mean_squared_error
    mse_dynamic = mean_squared_error(endog[n_train+lag:],predict_dy.predicted_mean[n_train:])
    #print('RMSE Dynamic: ',round(np.sqrt(mse_dynamic),3))
    mse_one = mean_squared_error(endog[n_train+lag:],predict.predicted_mean[n_train:])
    print('RMSE One-Step-Ahead: ',round(np.sqrt(mse_one),3))
    
    mape_dy = mape_vectorized(endog[n_train+lag:],predict_dy.predicted_mean[n_train:])
    #print('\nMAPE Dynamic: ',round(mape_dy,2),'%')
    mape_one = mape_vectorized(endog[n_train+lag:],predict.predicted_mean[n_train:])
    print('MAPE One-Step-Ahead: ',round(mape_one,2),'%')
    
    
    #NAIVE - MEAN VALUE: evaluate if I just predict the average of the timeseries
    naive = pd.Series(np.repeat(endog[n_train+lag:].mean(),len(endog[n_train+lag:])))
    naive.index = endog[n_train+lag:].index
    
    naive_mse = mean_squared_error(endog[n_train+lag:],naive)
    print('\nNAIVE RMSE: ',round(np.sqrt(naive_mse),3))
    naive_mape = mape_vectorized(endog[n_train+lag:],naive)
    print('NAIVE MAPE: ',round(naive_mape,2),'%')
    
    RMSE_dy.append(round(np.sqrt(mse_dynamic),3))
    RMSE_one.append(round(np.sqrt(mse_one),3))

    MAPE_dy.append(round(mape_dy,2))
    MAPE_one.append(round(mape_one,2))

    NAIVE_rmse.append(round(np.sqrt(naive_mse),2))
    NAIVE_mape.append(round(naive_mape,2))
    
report07_monthly = pd.DataFrame()
report07_monthly['PARAMS'] = MODEL_ID
report07_monthly['VARS'] = VARS
report07_monthly['LAGS'] = LAGS
report07_monthly['AIC'] = MODEL_AIC
report07_monthly['NORM_TEST_p'] = NORM_MW
report07_monthly['RESID.'] = RESID
report07_monthly['HET_TEST'] = HETERO
report07_monthly['RMSE_DY'] = RMSE_dy
report07_monthly['RMSE_ONE'] = RMSE_one
report07_monthly['MAPE_DY'] = MAPE_dy
report07_monthly['MAPE_ONE'] = MAPE_one
report07_monthly['NAIVE_RMSE'] = NAIVE_rmse
report07_monthly['NAIVE_MAPE'] = NAIVE_mape

os.chdir('C:/Users/trbella/Desktop/Thiago/Python/SARIMAX_GRID_LAGS/results/monthly_rev01')
report07_monthly.to_excel('report07_monthly_rev01.xlsx')

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