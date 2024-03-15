# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:11:35 2024

@author: Ashish
Time series forcasting
"""


import pandas as pd

Walmart = pd.read_csv("C:/Data_Science/17-Time_Series/DataSet/Walmart Footfalls Raw.csv")

#preprocessing
import numpy as np

Walmart["t"] = np.arange(1, 160)

Walmart["t_square"] = Walmart["t"] * Walmart["t"]

Walmart["log_footfalls"] = np.log(Walmart["t"])
Walmart.columns

#Months = ["Jan" ,"feb" , "March" , ..... "Dec"]
# in wallmart data we have jan 1991 in 0th column , we nedd only first three letters
 
p = Walmart["Month"][0]
p
#before we will extract , let us create a new columns
#called months to store extracted values
p[0:3]

Walmart["months"] = 0

#you can check the dataframe with months name with all values 0
# the total records are 159 in walmart
for i in range(159):
    p = Walmart["Month"][i]
    Walmart["months"][i] = p[0:3]

month_dummies = pd.DataFrame(pd.get_dummies(Walmart['months']))
#now let us concatenate these dummy values to dataframe
Walmart1 = pd.concat([Walmart , month_dummies] , axis = 1)

#visulization - Time plot
Walmart.Footfalls.plot()

#Data partition
Train = Walmart1.head(147)
Test = Walmart1.tail(12)

# to change the index value in pandas data frame
# Test.set_index(np.arange(1,13))

#-------------------------linear--------------------------------------
import statsmodels.formula.api as smf

linear_model = smf.ols('Footfalls ~ t' , data = Train).fit()
pred_linear = pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_linear))**2))
rmse_linear

#------------------------ Exponent model ------------------------------------
Exp = smf.ols('log_footfalls ~ t' , data = Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_Exp))**2))
rmse_Exp

#------------------------ Quadratic model --------------------------------
Quad = smf.ols("Footfalls ~ t + t_square" , data = Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t" , "t_square"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_Quad))**2))
rmse_Quad
#------------------------- Additive Seasonality ---------------------------
add_sea = smf.ols('Footfalls ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov' , data = Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan' , 'Feb' , 'Mar' , 'Apr' , 'May' , 'Jun' , 'Jul' , 'Aug' , 'Sep' , 'Oct' , 'Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_add_sea))**2))
rmse_add_sea
#------------------------- Multiplicative Seasonality ----------------------
Mul_sea = smf.ols('log_footfalls ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov+Dec' , data=Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea  =  np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_Mult_sea))**2))
rmse_Mult_sea

#-----------------------Additive Seasonality Quadratic trend--------------
add_sea_Quad = smf.ols('Footfalls ~ t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov' , data = Train).fit()
pred_add_sea_quad = pd.Series(add_sea.predict(Test[['Jan' , 'Feb' , 'Mar' , 'Apr' , 'May' , 'Jun' , 'Jul' , 'Aug' , 'Sep' , 'Oct' , 'Nov' , 't' , 't_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad
#------------Multiplicative Seasonality linear trend--------------------
Mul_sea_add = smf.ols('log_footfalls ~ t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov' , data=Train).fit()
pred_Mult_add_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_add_sea  =  np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_Mult_add_sea))**2))
rmse_Mult_add_sea
#Testing

data = {"MODEL" : pd.Series(["rmse_linear" , "rmse_Exp" , "rmse_Quad" , "rmse_add_sea" , "rmse_add_sea_quad" , "rmse_Mult_sea" , "rmse_Mult_add_sea"]) , "RMSE_Values" : pd.Series([rmse_linear , rmse_Exp , rmse_Quad , rmse_add_sea , rmse_add_sea_quad , rmse_Mult_add_sea])}
table_rmse = pd.DataFrame(data)
table_rmse

#---------------------------TESTING----------------------------------

#rmse_add_sea has the least value among the models prepared so far
predict_data = pd.read_excel("C:/Data_Science/17-Time_Series/DataSet/Predict_new.xlsx")

model_full = smf.ols('Footfalls ~ t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov' , data = Train).fit()
pred_new = pd.Series(model_full.predict(predict_data))
pred_new

predict_data["forecasted_Footfalls"] = pd.Series(pred_new)

#Autoregressive model (AR)
# Calculating Residuals from best model applied on full data
# AV - FV
full_res = Walmart.Footfalls - model_full.predict(Walmart1)

#ACF plot on residuals
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(full_res , lags = 12)
#ACF is na complete auto-correlateed function given values
# of auto correlation of any time series with its lagged values

#PACF is partial auto-correlation function
#it finds correlations of present with lags of the residuals of the

# Alternative approch for ACF plot
#From pandas.plotting import autocorrelation_plot
# autocorrelation_ppyplot.show()

tsa_plots.plot_pacf(full_res , lags =12)

#AR model
from statsmodels.tsa.ar_model import AutoReg
model_ar = AutoReg(full_res , lags=[1])
model_fit = model_ar.fit()

print("Coefficient is %s" % model_fit.params)

pred_res = model_fit.predict(start = len(full_res) , end =len(full_res)+len(predict_data)-1  , dynamic = False)
pred_res.reset_index(drop = True , inplace = True)

# the final predictions using ASQT and AR(1) Model
final_pred = pred_new + pred_res
final_pred










