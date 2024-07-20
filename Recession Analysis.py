#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "plotly_white"


# LOADING THE MONTHLY GDP GROWTH DATA OF UNITED STATES AND UNITED KINGDOM

# In[2]:


#US gdp
data = pd.read_csv(r"C:\Users\Arjun\Downloads\US_monthly_gdp.csv")
print(data.head())


# In[3]:


#UK gdp
data1 = pd.read_csv(r"C:\Users\Arjun\Downloads\UK_monthly_gdp.csv")
print(data1.head())


# GDP GROWTH OVERTIME IN UNITED STATES

# In[4]:


import plotly.graph_objects as go

fig = go.Figure(data=go.Heatmap(
                   z=[data['GDP Growth']],
                   x=data.index,
                   y=['GDP Growth'],
                   colorscale='Viridis'))

fig.update_layout(title='GDP Growth over Time in US',
                  xaxis_title='Time Period',
                  yaxis_title='')

fig.show()


# In[5]:


import plotly.graph_objects as go

fig1 = go.Figure(data=go.Heatmap(
                   z=[data1['GDP Growth']],
                   x=data1.index,
                   y=['GDP Growth'],
                   colorscale='Viridis'))

fig1.update_layout(title='GDP Growth over Time in UK',
                  xaxis_title='Time Period',
                  yaxis_title='')

fig1.show()


# In[6]:


# Convert monthly data to quarterly data using resample method
data['Time Period'] = pd.to_datetime(data['Time Period'], format='%d-%m-%Y')
data.set_index('Time Period', inplace=True)
quarterly_data = data.resample('Q').mean()
print(quarterly_data.head(15))


# In[7]:


import pandas as pd

# Assuming 'Time Period' is the correct column name containing the time data
# Replace 'Time Period' with the actual column name if it's different
data1['Time Period'] = pd.to_datetime(data1['Time Period'], format='/%m/%Y')

# Ensure that the column 'Time Period' is set as the index
data1.set_index('Time Period', inplace=True)

# Resample the data to quarterly frequency
quarterly_data1 = data1.resample('Q').mean()

print(quarterly_data1.head())


# ANALYSIS BASED ON QUATERLY GDP

# i)US

# In[8]:


# Calculate recession based on quarterly GDP growth
quarterly_data['Recession'] = ((quarterly_data['GDP Growth'] < 0) & (quarterly_data['GDP Growth'].shift(1) < 0))

# Fill missing values with False (since the first quarter cannot be in a recession)
quarterly_data['Recession'].fillna(False, inplace=True)


# In[9]:


# Plot the GDP growth and recession data
fig = go.Figure()
fig.add_trace(go.Scatter(x=quarterly_data.index, 
                         y=quarterly_data['GDP Growth'], 
                         name='GDP Growth', 
                         line=dict(color='green', width=2)))
fig.add_trace(go.Scatter(x=quarterly_data[quarterly_data['Recession']].index, 
                         y=quarterly_data[quarterly_data['Recession']]['GDP Growth'], 
                         name='Recession', line=dict(color='red', width=2)))

fig.update_layout(title='GDP Growth and Recession over Time (Quarterly Data)',
                  xaxis_title='Time Period',
                  yaxis_title='GDP Growth in US')

fig.show()


# ii)UK

# In[10]:


# Calculate recession based on quarterly GDP growth
quarterly_data1['Recession'] = ((quarterly_data1['GDP Growth'] < 0) & (quarterly_data1['GDP Growth'].shift(1) < 0))

# Fill missing values with False (since the first quarter cannot be in a recession)
quarterly_data1['Recession'].fillna(False, inplace=True)


# In[11]:


# Plot the GDP growth and recession data
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=quarterly_data1.index, 
                         y=quarterly_data1['GDP Growth'], 
                         name='GDP Growth', 
                         line=dict(color='green', width=2)))
fig1.add_trace(go.Scatter(x=quarterly_data1[quarterly_data1['Recession']].index, 
                         y=quarterly_data1[quarterly_data1['Recession']]['GDP Growth'], 
                         name='Recession', line=dict(color='red', width=2)))

fig1.update_layout(title='GDP Growth and Recession over Time (Quarterly Data)',
                  xaxis_title='Time Period',
                  yaxis_title='GDP Growth in UK')

fig1.show()


# In[12]:


#The red line shows the periods of negative GDP growth (considered recessions), and the green line shows the overall trend in GDP growth over time.


# In[13]:


quarterly_data['Recession Start'] = quarterly_data['Recession'].ne(quarterly_data['Recession'].shift()).cumsum()
recession_periods = quarterly_data.groupby('Recession Start')
recession_duration = recession_periods.size()
recession_severity = recession_periods['GDP Growth'].sum()

fig = go.Figure()
fig.add_trace(go.Bar(x=recession_duration.index, y=recession_duration,
                     name='Recession Duration'))
fig.add_trace(go.Bar(x=recession_severity.index, y=recession_severity,
                     name='Recession Severity'))

fig.update_layout(title='Duration and Severity of Recession in US',
                  xaxis_title='Recession Periods',
                  yaxis_title='Duration/Severity')

fig.show()


# In[14]:


quarterly_data1['Recession Start'] = quarterly_data1['Recession'].ne(quarterly_data1['Recession'].shift()).cumsum()
recession_periods1 = quarterly_data1.groupby('Recession Start')
recession_duration1 = recession_periods1.size()
recession_severity1 = recession_periods1['GDP Growth'].sum()

fig1 = go.Figure()
fig1.add_trace(go.Bar(x=recession_duration1.index, y=recession_duration1,
                     name='Recession Duration'))
fig1.add_trace(go.Bar(x=recession_severity1.index, y=recession_severity1,
                     name='Recession Severity'))

fig1.update_layout(title='Duration and Severity of Recession',
                  xaxis_title='Recession Periods',
                  yaxis_title='Duration/Severity')

fig1.show()


# In[15]:


import pandas as pd
import plotly.graph_objects as go

# Assuming you already have defined `quarterly_data` and `quarterly_data1` DataFrames

# Calculate recession periods, duration, and severity for the first DataFrame
quarterly_data['Recession Start'] = quarterly_data['Recession'].ne(quarterly_data['Recession'].shift()).cumsum()
recession_periods = quarterly_data.groupby('Recession Start')
recession_duration = recession_periods.size()
recession_severity = recession_periods['GDP Growth'].sum()

# Calculate recession periods, duration, and severity for the second DataFrame
quarterly_data1['Recession Start'] = quarterly_data1['Recession'].ne(quarterly_data1['Recession'].shift()).cumsum()
recession_periods1 = quarterly_data1.groupby('Recession Start')
recession_duration1 = recession_periods1.size()
recession_severity1 = recession_periods1['GDP Growth'].sum()

# Plot duration and severity of recessions for both DataFrames
fig = go.Figure()
fig.add_trace(go.Bar(x=recession_duration.index, y=recession_duration,
                     name='Recession Duration (US)', marker_color='blue'))
fig.add_trace(go.Bar(x=recession_severity.index, y=recession_severity,
                     name='Recession Severity (US)', marker_color='lightblue'))
fig.add_trace(go.Bar(x=recession_duration1.index, y=recession_duration1,
                     name='Recession Duration (UK)', marker_color='red'))
fig.add_trace(go.Bar(x=recession_severity1.index, y=recession_severity1,
                     name='Recession Severity (UK)', marker_color='salmon'))

fig.update_layout(title='Duration and Severity of Recession',
                  xaxis_title='Recession Periods',
                  yaxis_title='Duration/Severity')

fig.show()


# PREDICTION USING ARIMA

# In[16]:


#US
import matplotlib.pyplot as plt 
plt.xlabel('Date')
plt.ylabel('GDP')
plt.plot(data)


# In[17]:


#check whether stationary or not
rol_mean=data.rolling(window=12).mean()
rol_std=data.rolling(window=12).std()
plt.plot(data,c='blue')
plt.plot(rol_mean,c='black')
plt.plot(rol_std,c='red')
#since mean and std 0bserved is not constant hence not stationary


# In[18]:


from statsmodels.tsa.stattools import adfuller
dftest=adfuller(data['GDP Growth'])
print(dftest)


# In[19]:


from statsmodels.tsa.seasonal import seasonal_decompose
decompose=seasonal_decompose(data)
trend=decompose.trend
season=decompose.seasonal


# In[20]:


plt.subplot(211)#21 is the width while 1 and 2 distributing in the same plot
plt.plot(trend)
plt.subplot(212)
plt.plot(season)


# In[21]:


data_new=data['GDP Growth']-data['GDP Growth'].shift(2)


# In[22]:


adfuller(data_new.dropna())#now its stationary as we get a value closer to 0.05


# In[23]:


plt.plot(data_new.dropna())


# In[24]:


from statsmodels.tsa.stattools import acf,pacf
acf_plot=acf(data_new.dropna())
pacf_plot=pacf(data_new.dropna())


# In[25]:


plt.subplot(211)#21 is the width while 1 and 2 distributing in the same plot
plt.plot(acf_plot)
plt.subplot(212)
plt.plot(pacf_plot)


# In[26]:


from statsmodels.tsa.arima.model import ARIMA
model=ARIMA(data,seasonal_order=(4,2,2,12))
model_fit=model.fit()


# In[27]:


res=model_fit.forecast(36)
res=pd.DataFrame(res)
res.columns=['GDP Growth']


# In[28]:


plt.plot(data,c='pink')#gives based on the current situation
plt.plot(res,c='purple')#gives forcast for next 24 months


# In[29]:


#UK
import matplotlib.pyplot as plt 
plt.xlabel('Date')
plt.ylabel('GDP')
plt.plot(data1)


# In[30]:


from statsmodels.tsa.arima.model import ARIMA
model=ARIMA(data1,seasonal_order=(4,2,2,12))
model_fit=model.fit()


# In[31]:


res=model_fit.forecast(36)
res=pd.DataFrame(res)
res.columns=['GDP Growth']


# In[32]:


plt.plot(data1,c='pink')#gives based on the current situation
plt.plot(res,c='purple')#gives forcast for next 24 months

