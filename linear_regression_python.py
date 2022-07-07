#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sys


# ## Load the dataset

# In[5]:


df = pd.read_csv(sys.argv[1])



# ## Scatter plot for original data

# In[6]:


plt.scatter(df[['x']], df[['y']], color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('py_orig.png')
plt.clf()


# ## Model the data

# In[7]:


model = LinearRegression()
model.fit(df[['x']], df[['y']])


# ## Linear model plot with original data

# In[9]:


plt.scatter(df[['x']], df[['y']], color='red')
plt.plot(df[['x']], model.predict(df[['x']]), color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Feature vs Prediction')
plt.savefig('py_lm.png')
plt.clf()

