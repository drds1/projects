import numpy as np
import pandas as pd





#input list of pandas data frames (timestamp, quantity) with different timestamps
#rebin onto single axis and convert to numpy array

#merge dates of all other frames onto first (if no date entry matching up with first 
#date frame then input nan

#added column_names argument to specify the date and value column of each dat frame to combine

def combine_dataframes(df_list,datemin=-1,column_names = None):
 nlist = len(df_list)
 df_combine = df_list[0]
 df_0 = df_list[0]
 n0 = len(df_0)
 output_array = np.zeros((n0,nlist))
 output_array[:,0] = df_0.values[:,1]
 
 for i in range(nlist):
  df_now = df_list[i]
  
  columns = df_now.columns
  if column_names != None:
   column_names_now = column_names[i]
  else:
   column_names_now = [columns[0],columns[1]]
  
  
  
  
  cdate = column_names_now[0]
  columns0 = df_combine.columns
  cdate0 = columns0[0]  
  df_0[cdate0] = pd.to_datetime(df_0[cdate0])
  df_now[cdate] = pd.to_datetime(df_now[cdate])
  

  
  df_new = pd.merge(df_0, df_now[column_names_now], left_on = cdate0, right_on = cdate,how='left')
  vals_new = df_new.values[:,-1]
  output_array[:,i] = vals_new

 
 dates = df_0.values[:,0]
 output_data = output_array
 

 sums = np.sum(output_data,axis=1)
 idgood = np.where(sums == sums)[0]
 dates = dates[idgood]
 output_data = output_data[idgood,:]
 
 return(dates,output_data)