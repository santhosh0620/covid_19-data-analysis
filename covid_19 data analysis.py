#!/usr/bin/env python
# coding: utf-8

# In[1]:


#project 1
#This simple basic project explains the data visualization of covid_19
#created on april 3,2021. 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import os 


# In[2]:


df=pd.read_csv('D:\\data science aftab\\covid_19_india.csv')
df.tail(50)


# In[3]:


cureds=df['Cured']
print(cureds)
print("*****************")
c1=cureds.tail(10)
print(c1)


# In[4]:


death =df['Deaths'] 
print(death)
print("*****************")
d1=death.tail(10)
print(d1)


# In[5]:


confirm= df['Confirmed']
print(confirm)
print("*****************")
u1=confirm.tail(10)
print(u1)


# In[6]:


# visualization of cured cases
plt.plot(c1,color='g',marker='*')
plt.title("cured cases")
plt.grid()
plt.show()


# In[7]:


# visualization of death cases
plt.plot(d1,color='red',marker='*')
plt.title("death cases")
plt.grid()
plt.show()


# In[8]:


# visualization of confirmed cases
plt.plot(u1,marker='*')
plt.title("confirmed cases")
plt.grid()
plt.show()


# In[9]:


#maximum cases analysed
x=cureds.max()
print("max cured",x)
y=death.max()
print("max death",y)
z=confirm.max()
print("max confirm",z)


# In[10]:


#minimum cases analysed
x=cureds.min()
print("min cured",x)
y=death.min()
print("min death",y)
z=confirm.min()
print("min confirm",z)


# In[ ]:


#maximum cases analysed(index)
x=cureds.argmax()
print("max cured index",x)
y=death.argmax()
print("max death index",y)
z=confirm.argmax()
print("max confirm index",z)

