#generate fake data
import numpy as np
import pandas as pd
n = 3 * 365
dt = 1
time = np.arange(0, n, dt)
predict = 100
time_extra = np.arange(0, time[-1] + (predict + 1) * dt, dt)
noise = 0.3
y_flows_true = np.sin(2 * np.pi * 1. / 365 * time_extra) + 0.3 * np.sin(2 * np.pi * 1. / 90 * time_extra)
y = y_flows_true + noise * np.random.randn(n + predict)
y_train = y[:-predict]
# select an arb that leads by 3 days
y_arb = np.roll(y_flows_true, -100) + noise * np.random.randn(n + predict)
y_components = np.zeros((n + predict, 1))
y_components[:, 0] = y_arb
y_components_train = y_components[:-predict]
# specify the dates of the observations (list of dataframes)
today = pd.datetime.now()
dates = pd.date_range(start=today, periods=n)






#this is how to use the forecast class
import matplotlib.pylab as plt
import fish
a = fish.forecast()
a.set_dates(dates)
a.set_main_timeseries(y_train)
a.set_contributing_timeseries(y_components_train)
a.predict(days=90)
a.plot()
plt.show()
