# Exploratory-Data-Analysis-with-Seaborn
Exploratory Data Analysis with Seaborn

#!/usr/bin/env python
# coding: utf-8

# <h2 align=center>Tumor Diagnosis: Exploratory Data Analysis</h2>
# <img src="https://storage.googleapis.com/kaggle-datasets-images/180/384/3da2510581f9d3b902307ff8d06fe327/dataset-cover.jpg">
# 

# ### About the Dataset:

# The [Breast Cancer Diagnostic data](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) is available on the UCI Machine Learning Repository. This database is also available through the [UW CS ftp server](http://ftp.cs.wisc.edu/math-prog/cpo-dataset/machine-learn/cancer/WDBC/).
# 
# Features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. n the 3-dimensional space is that described in: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34].

# **Attribute Information**:
# 
# - ID number
# - Diagnosis (M = malignant, B = benign) 3-32)

# Ten real-valued features are computed for each cell nucleus:
# 
# 1. radius (mean of distances from center to points on the perimeter) 
# 2. texture (standard deviation of gray-scale values) 
# 3. perimeter 
# 4. area 
# 5. smoothness (local variation in radius lengths) 
# 6. compactness (perimeter^2 / area - 1.0) 
# 7. concavity (severity of concave portions of the contour) 
# 8. concave points (number of concave portions of the contour)
# 9. symmetry
# 10. fractal dimension ("coastline approximation" - 1)
# 
# The mean, standard error and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.
# 
# All feature values are recoded with four significant digits.
# 
# Missing attribute values: none
# 
# Class distribution: 357 benign, 212 malignant

# ### Task 1: Loading Libraries and Data

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns  # data visualization library  
import matplotlib.pyplot as plt
import time


# In[2]:


data = pd.read_csv('/kaggle/input/dataset/data.csv')


#  

# <h2 align=center> Exploratory Data Analysis </h2>
# 
# ---

#  

# ### Task 2: Separate Target from Features
# 

# In[3]:


data.head()


# In[4]:


col = data.columns
print (col)


# In[5]:


y = data.diagnosis
drop_cols = ['Unnamed: 32','id','diagnosis']
x = data.drop(drop_cols, axis = 1)
x.head()


#  

# ### Task 3: Plot Diagnosis Distributions
# 

# In[6]:


B, M = y.value_counts()
print('Number of Benign Trumors', B)
print('Number of Malignant Tumors', M)


# In[7]:


x.describe()


#  

# <h2 align=center> Data Visualization </h2>
# 
# ---

#  

# ### Task 4: Visualizing Standardized Data with Seaborn
# 

# In[8]:


data = x
data_std = (data - data.mean())/data.std()
data = pd.concat([y, data_std.iloc[:, 0:10]], axis = 1)
data = pd.melt(data, id_vars='diagnosis', var_name='features', value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x='features',y='value',hue='diagnosis', data= data, split=True, inner='quart')
plt.xticks(rotation=45); 


# In[ ]:





# In[ ]:





#  

# ### Task 5: Violin Plots and Box Plots
# 

# In[9]:


data = x
data_std = (data - data.mean())/data.std()
data = pd.concat([y, data_std.iloc[:, 10:20]], axis = 1)
data = pd.melt(data, id_vars='diagnosis', var_name='features', value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x='features',y='value',hue='diagnosis', data= data, split=True, inner='quart')
plt.xticks(rotation=45); 


# In[10]:


data = x
data_std = (data - data.mean())/data.std()
data = pd.concat([y, data_std.iloc[:, 20:30]], axis = 1)
data = pd.melt(data, id_vars='diagnosis', var_name='features', value_name='value')
plt.figure(figsize=(10,10))
sns.violinplot(x='features',y='value',hue='diagnosis', data= data, split=True, inner='quart')
plt.xticks(rotation=45); 


# In[11]:


sns.boxplot(x='features', y='value', hue='diagnosis', data=data)
plt.xticks(rotation=45); 


#  

# 
# 

# In[ ]:





#  

#  

# ### Task 6: Observing the Distribution of Values and their Variance with Swarm Plots
# 

# In[12]:


sns.set(style='whitegrid', palette='muted')
data = x
data_std = (data - data.mean())/data.std()
data = pd.concat([y, data_std.iloc[:, 0:10]], axis = 1)
data = pd.melt(data, id_vars='diagnosis', var_name='features', value_name='value')
plt.figure(figsize=(10,10))
sns.swarmplot(x='features',y='value',hue='diagnosis', data= data)
plt.xticks(rotation=45); 


# In[13]:


sns.set(style='whitegrid', palette='muted')
data = x
data_std = (data - data.mean())/data.std()
data = pd.concat([y, data_std.iloc[:, 10:20]], axis = 1)
data = pd.melt(data, id_vars='diagnosis', var_name='features', value_name='value')
plt.figure(figsize=(10,10))
sns.swarmplot(x='features',y='value',hue='diagnosis', data= data)
plt.xticks(rotation=45); 


# In[14]:


sns.set(style='whitegrid', palette='muted')
data = x
data_std = (data - data.mean())/data.std()
data = pd.concat([y, data_std.iloc[:, 20:30]], axis = 1)
data = pd.melt(data, id_vars='diagnosis', var_name='features', value_name='value')
plt.figure(figsize=(10,10))
sns.swarmplot(x='features',y='value',hue='diagnosis', data= data)
plt.xticks(rotation=45); 


#  

# ### Task 7: Observing all Pair-wise Correlations
# 

# In[15]:


f, ax = plt.subplots(figsize=(18,18))
sns.heatmap(x.corr(), annot=True, linewidth=.5, fmt='.1f', ax=ax);


# In[ ]:





# In[ ]:





# In[ ]:



