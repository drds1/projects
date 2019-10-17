import numpy as np
import pandas as pd
import pyflux as pf
from datetime import datetime
import matplotlib.pyplot as plt


data = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/MASS/drivers.csv")
data = data.rename(columns = {'value':'drivers'}).iloc[:,1:]
data.loc[(data['time']>=1983.05), 'seat_belt'] = 1
data.loc[(data['time']<1983.05), 'seat_belt'] = 0
data.loc[(data['time']>=1974.00), 'oil_crisis'] = 1
data.loc[(data['time']<1974.00), 'oil_crisis'] = 0
data.set_index('time',inplace=True)
plt.figure(figsize=(15,5))
plt.plot(data['drivers'])
plt.ylabel('Driver Deaths')
plt.title('Deaths of Car Drivers in Great Britain 1969-84')
plt.plot()
plt.show()


model = pf.ARIMAX(data=data, formula='drivers~seat_belt+oil_crisis',
                  ar=1, ma=1, family=pf.Normal())
x = model.fit("MLE")
x.summary()

model.plot_fit(figsize=(15,10))



model.plot_predict(h=10, oos_data=data.iloc[-12:], past_values=100, figsize=(15,5))
