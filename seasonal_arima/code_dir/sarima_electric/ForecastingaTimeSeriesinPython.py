
# coding: utf-8

# # How to Forecast a Time Series with Python
# 
# Wouldn't it be nice to know the future? This is the notebook that relates to the blog post on medium. Please check the blog for visualizations and explanations, this notebook is really just for the code :)
# 
# 
# ## Processing the Data
# 
# Let's explore the Industrial production of electric and gas utilities in the United States, from the years 1985-2018, with our frequency being Monthly production output.
# 
# You can access this data here: https://fred.stlouisfed.org/series/IPG2211A2N
# 
# This data measures the real output of all relevant establishments located in the United States, regardless of their ownership, but not those located in U.S. territories.

# In[1]:


get_ipython().magic(u'matplotlib inline')
import pandas as pd
data = pd.read_csv("Electric_Production.csv",index_col=0)
data.head()


# Right now our index is actually just a list of strings that look like a date, we'll want to adjust these to be timestamps, that way our forecasting analysis will be able to interpret these values:

# In[2]:


data.index


# In[3]:


data.index = pd.to_datetime(data.index)


# In[4]:


data.head()


# In[5]:


data.index


# Let's first make sure that the data doesn't have any missing data points:

# In[6]:


data[pd.isnull(data['IPG2211A2N'])]


# Let's also rename this column since its hard to remember what "IPG2211A2N" code stands for:

# In[7]:


data.columns = ['Energy Production']


# In[8]:


data.head()


# In[49]:


import plotly
# plotly.tools.set_credentials_file()


# In[27]:


from plotly.plotly import plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(data, model='multiplicative')
fig = result.plot()
plot_mpl(fig)


# In[28]:


import plotly.plotly as ply
import cufflinks as cf
# Check the docs on setting up offline plotting


# In[ ]:


data.iplot(title="Energy Production Jan 1985--Jan 2018", theme='pearl')


# In[10]:


from pyramid.arima import auto_arima


# **he AIC measures how well a model fits the data while taking into account the overall complexity of the model. A model that fits the data very well while using lots of features will be assigned a larger AIC score than a model that uses fewer features to achieve the same goodness-of-fit. Therefore, we are interested in finding the model that yields the lowest AIC value.

# In[30]:


stepwise_model = auto_arima(data, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True) 


# In[31]:


stepwise_model.aic()


# ## Train Test Split

# In[32]:


data.head()


# In[33]:


data.info()


# We'll train on 20 years of data, from the years 1985-2015 and test our forcast on the years after that and compare it to the real data.

# In[34]:


train = data.loc['1985-01-01':'2016-12-01']


# In[35]:


train.tail()


# In[36]:


test = data.loc['2015-01-01':]


# In[37]:


test.head()


# In[38]:


test.tail()


# In[39]:


len(test)


# In[40]:


stepwise_model.fit(train)


# In[41]:


future_forecast = stepwise_model.predict(n_periods=37)


# In[42]:


future_forecast


# In[43]:


future_forecast = pd.DataFrame(future_forecast,index = test.index,columns=['Prediction'])


# In[44]:


future_forecast.head()


# In[45]:


test.head()


# In[46]:


pd.concat([test,future_forecast],axis=1).iplot()


# In[47]:


future_forecast2 = future_forcast


# In[48]:


pd.concat([data,future_forecast2],axis=1).iplot()

