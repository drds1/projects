import numpy as np
import pandas as pd


tnow = pd.Timestamp.now()
x = [tnow]*10
y = list(np.arange(10))


dfr = pd.DataFrame([x,y]).T
dfr.columns=['date','vals']
dfr['date'] = pd.to_datetime(dfr['date'])
dfr.set_index('date',inplace=True)

writer = pd.ExcelWriter('output.xlsx',datetime_format='dd-mmm-yy')
dfr.to_excel(writer,'Sheet1')
dfr.to_excel(writer,'Sheet2')
writer.save()

                        
                        