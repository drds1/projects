import numpy as np
import matplotlib.pylab as plt
import pandas as pd


n 	  = 100

datelo =  pd.Timestamp.today()
date = pd.date_range(start=datelo, periods=n)
t 	  = np.arange(n)
noise = 0.1

y = np.sin(2*np.pi/30 * t) + noise*np.random.randn(n)



import fish_forecast
forecast = fish_forecast.forecast()