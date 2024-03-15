# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 08:48:45 2024

@author: Ashish Chincholikar
Smoothing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import Holt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

#load the Dataset
cocacola = pd.read_excel("C:/Data_Science/17-Time_Series/DataSet/CocaCola_Sales_Rawdata.xlsx")

# let us plot the dataset and it's nature
cocacola.Sales.plot()
# splitting the data into tarin and test set data
# since we are working on quaterly dataset and in year there are 
# test data = 4 quaters
# train data = 38

Train = cocacola.head(38)
Test = cocacola.tail(4)

# Here we are considering performance parameters as mean absolute
# rather than mean square error
# custom function is written to calculate MPSE
def MAPE(pred , org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)

# EDA which comprises identification of level , trends and seasonality
# IN order to seperate trend and seasonality moving averages can be used
mv_pred = cocacola["Sales"].rolling(4).mean()
mv_pred.tail(4)

# now let us calculate mean absolute percentage error of these values
MAPE(mv_pred.tail(4) , Test.Sales)

# moving average is predicting complete values , out of which last
# are condiereed as predicted values and last four values of test
# basic purpos of moving average is deseasonalizing

cocacola.Sales.plot(label = 'org')
# this is the original plot
# now let us seperate out trend and seasonality

for i in range(2,9,2):
    # it will take window size 2,4,6,8
    cocacola['Sales'].rolling(i).mean().plot(label = str(i))
    plt.legend(loc = 3)
    
# you can see i = 4  and 8 are deseasonable plots

# Time series decomposition is the another technique of seperating
# seasonality

decompose_ts_add = seasonal_decompose(cocacola.Sales , model = "additive" , period = 4)
print(decompose_ts_add.trend)
print(decompose_ts_add.seasonal)
print(decompose_ts_add.resid)
print(decompose_ts_add.observed)
decompose_ts_add.plot()

# similar plot can be decomposed using multiplicative
decompose_ts_mul = seasonal_decompose(cocacola.Sales , model="multiplicative" , period = 4)
print(decompose_ts_mul.trend)
print(decompose_ts_mul.seasonal)
print(decompose_ts_mul.resid)
print(decompose_ts_mul.observed)
decompose_ts_mul.plot()

# you can observe the difference between these plots
# now let us plot ACF plot to check the auto correlation
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(cocacola.Sales , lags = 4)
# we can observe the output in which r1 , r2, r3 and r4 has higher
# this is all about EDA
# let us apply data to data driven models
# simple exponential method

from statsmodels.tsa.holtwinters import SimpleExpSmoothing
ses_model = SimpleExpSmoothing(Train['Sales']).fit()
pred_ses = ses_model.predict(start = Test.index[0], end = Test.index[-1])

# now calculate MAPE
MAPE(pred_ses , Test.Sales)
# we are getting 8.272016681540089
# holts exponential smoothing here only trend is captured
hw_model = Holt(Train['Sales']).fit()
pred_hw = hw_model.predict(start = Test.index[0] , end = Test.index[-1])
MAPE(pred_hw , Test.Sales)

# Holt's winter exponential smoothing with additive seasonality
hwe_model_add_add = ExponentialSmoothing(Train['Sales'] , seasonal='add' , trend = "add" , seasonal_periods=4).fit()
pred_hwe_model_add_add = hwe_model_add_add.predict(start = Test.index[0] , end = Test.index[-1])
MAPE(pred_hwe_model_add_add , Test.Sales)
# 3.372576621498227

# holts winter exponential smoothing with multiplicative seasonality
hwe_model_mul_add = ExponentialSmoothing(Train["Sales"] , seasonal='add' , trend = "add" , seasonal_periods=4).fit()
pred_hwe_model_mul_add = hwe_model_mul_add.predict(start = Test.index[0] , end = Test.index[-1])
MAPE(pred_hwe_model_mul_add , Test.Sales)

# let us apply to complete data of cocacola
# we have seen that hwe_model_add_add has got lowest
hwe_model_add_add = ExponentialSmoothing(cocacola["Sales"])

# import the new datasets for which the predictions has to be made
new_data = pd.read_excel("C:/Data_Science/17-Time_Series/DataSet/CocaCola_Sales_New_Pred.xlsx")

newdata_pred = hwe_model_add_add.predict(start = newdata)
MAPE(newdata_pred , Test.Sales)
newdata_pred











