# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:00:29 2024

@author: Ashish Chincholikar
ARMA
Moving averages
"""

import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme()
# Using the available dowjones data in seaborn
dowjones = sns.load_dataset("dowjones")
dowjones.head()
sns.lineplot(data = dowjones , x = "Date" , y="Price")

""" 
 A simple moving average calculates the average of a selected range of variables
 by the number of periods, in that range
 the most typical moving averages are 30-days , 100-days and 365-days
 moving averages . Moving averages are nice cause they can determine trends
 while ignoring short-term fluctuations
 one can calculate the sma by simply using
 
"""
dowjones['sma_30'] = dowjones['Price'].rolling(window = 30 , min_periods = 1).mean()
dowjones['sma_50'] = dowjones['Price'].rolling(window = 50 , min_periods = 1).mean()
dowjones['sma_100'] = dowjones['Price'].rolling(window = 100 , min_periods = 1).mean()
dowjones['sma_365'] = dowjones['Price'].rolling(window = 365 , min_periods = 1).mean()

sns.lineplot(x = "Date" , y = "value" , legend = 'auto' , hue = 'variable' , data = dowjones.melt('Date'))

"""
 As you can see the higher the values of the window , 
 the lesser it is affected by short-term fluctuations
 and it captures long-terms trend in the data.
 Simple moving averages are oftern used by traders
 in the stock market for technical analysis """

#Exponent Moving Averages
""" 
  Simple moving averages are nice , but 
  they give equal weightage to each of the data points,
  what if you wanted an average that will give higher weight
  to more recent points and lesser to points in the past. in that case,
  what you want is to compute the exponential moving average
"""

dowjones['ema_50'] = dowjones['Price'].ewm(span = 50 , adjust = False).mean()
dowjones['ema_100'] = dowjones['Price'].ewm(span = 100 , adjust = False).mean()

sns.lineplot(x = "Date" , y="value" , legend = "auto" , hue = "variable" , data = dowjones[['Date' , 'Price' , 'ema_50' , 'ema_50']].melt('Date'))

""" 
 As you can see the ema_50 follows the price chart more closley
 than the sma_50 and is more sensitive to the recent data points
"""














