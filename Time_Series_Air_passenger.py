# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 09:37:45 2024

@author: Ashish
Quick review of Time-Series
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('dark_background')

#load the datasets

df = pd.read_csv("C:/Data_Science/17-Time_Series/DataSet/AirPassengers.csv")
df.columns
df = df.rename({"#Passengers": "Passengers"} , axis = 1)

print(df.dtypes)
#month is text and passengers as int
#now let us convert into date and time
df['Month'] = pd.to_datetime(df['Month'])
print(df.dtypes)

df.set_index('Month' , inplace = True)

plt.plot(df.Passengers)
#there is incresing trend and has got seasonality

#IS the data Stationary?
#Dickey fuller test
from statsmodels.tsa.stattools import adfuller
adf , pvalue , usedlag_ , nobs_ , critical_values_ , icbest = adfuller(df)
print("pvalue = " , pvalue , "if above 0.05 , data is not stationary")
#Since data is not stationary , we may need SARIMA and not just ARIMA
# now let us extrct the year and month from the date and time column

df['year'] = [d.year for d in df.index]
df['month'] = [d.strftime('%b') for d in df.index]
years = df['year'].unique()

# plot yearly and monthly values as boxplot
sns.boxplot(x = 'year' , y = "Passengers" , data = df)
# no of passengers are going up year by year
sns.boxplot(x = 'month' , y="Passengers" , data = df)
# Overall there is higher trend in july and august compared to rest of the 

#Extract and plot trend ,seasonal and residuals
from statsmodels.tsa.seasonal import seasonal_decompose
decomposed = seasonal_decompose(df['Passengers'] , model = 'additive')

#Additive time series:
#value = base_level + Trend + seasonality + Error
# multiplicative time series:
# value = base_level + Trend + seasonality + error

trend = decomposed.trend
seasonal = decomposed.seasonal #Cyclic behaviour may not be seasonal
residual = decomposed.resid

plt.figure(figsize = (12,8))
plt.subplot(411)
plt.plot(df['Passengers'] , label = 'Original' , color = "yellow")
plt.legend(loc = 'upper left')
plt.subplot(412)
plt.plot(trend , label = 'trend' , color = "yellow")
plt.legend(loc = 'upper left')
plt.subplot(413)
plt.plot(seasonal , label = 'seasonal' , color = "yellow")
plt.legend(loc = 'upper left')
plt.subplot(414)
plt.plot(residual , label = 'residual' , color = "yellow")
plt.legend(loc = 'upper left')
plt.show()

""" 
Trend is going up from 1950s to 60s 
It is highly seasonal and showing peaks at particular interval
This helps to select specific prediction model
"""


# AUTOCORRELATION 
# Values are not correlated with x-axis but with its lag
# meaning yestardays value is depend on day before yesterdays so on and soforth
# Autocorrelation is simply the correlation of a series with its own lags
# plot lag on x axis and correlation on y axis
# Any correlation above confidence lnes are statistically significant

from statsmodels.tsa.stattools import acf

acf_144 = acf(df.Passengers , nlags=144)
plt.plot(acf_144)
#Auto correlation above zero means positive correlation and below as negative 
#obtain the same but with single line and more line...

from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df.Passengers)
#Any lag before 40 has positive correlation
# Horizontal band indicates 95% and 99% (dashed) confidence bands


























