#!/usr/bin/env python
# coding: utf-8

# In[2]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn


# In[17]:


Air=pd.read_excel('C:/Users/prate/Downloads/Assignment/Clustering/EastWestAirlines.xlsx')


# In[18]:


Air


# In[19]:


Air.isnull().sum()


# In[20]:


Air.describe()


# In[22]:


# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[24]:


norm_Air=norm_func(Air)


# In[25]:


norm_Air.head()


# In[26]:


# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(norm_Air, method='single'))


# In[27]:


# create clusters
hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'single')


# In[28]:


# save clusters for chart
y_hc = hc.fit_predict(norm_Air)
Clusters=pd.DataFrame(y_hc,columns=['Clusters'])


# In[29]:


Clusters


# In[ ]:


# k means


# In[30]:


from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 


# In[32]:


# Selecting 6 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=6) 
model.fit(norm_Air)

model.labels_ # getting the labels of clusters assigned to each row 


# In[33]:


kd=pd.Series(model.labels_)  # converting numpy array into pandas series object 
Air['clust']=kd # creating a  new column and assigning it to new column 
norm_Air.head()


# In[35]:


Air.iloc[:,1:12].groupby(Air.clust).mean()


# In[36]:


Air


# In[37]:


model1 = KMeans(n_clusters=11).fit(norm_Air)


# In[38]:


#DBSCAN
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# In[39]:


dbscan = DBSCAN(eps=0.8, min_samples=11)
dbscan.fit(norm_Air)


# In[40]:


#Noisy samples are given the label -1.
dbscan.labels_


# In[ ]:




