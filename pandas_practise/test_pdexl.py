import numpy as np
import pandas as pd


tnow = pd.Timestamp.now()
x = [tnow]*10
y = list(np.arange(10))


dfr = pd.DataFrame([x,y]).T


writer = pd.ExcelWriter('output.xlsx',datetime_format='mmm d yyyy hh:mm:ss',date_format='mmmm dd yyyy')
dfr.to_excel(writer,'Sheet1')
dfr.to_excel(writer,'Sheet2')
writer.save()

                        
                        