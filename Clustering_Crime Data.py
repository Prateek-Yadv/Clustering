#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn


# In[5]:


crime_rate=pd.read_csv('C:/Users/prate/Downloads/Assignment/Clustering/crime_data.csv')


# In[6]:


crime_rate.head()


# In[8]:


crime_rate.isnull().sum()


# In[9]:


crime_rate.describe()


# In[10]:


# Normalization function 
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)


# In[13]:


norm_CR=norm_func(crime_rate.iloc[:,1:])


# In[14]:


norm_CR.head()


# In[15]:


# create dendrogram
dendrogram = sch.dendrogram(sch.linkage(norm_CR, method='single'))


# In[16]:


# create clusters
hc = AgglomerativeClustering(n_clusters=4, affinity = 'euclidean', linkage = 'single')


# In[17]:


# save clusters for chart
y_hc = hc.fit_predict(norm_CR)
Clusters=pd.DataFrame(y_hc,columns=['Clusters'])


# In[20]:


Clusters


# In[ ]:


# k means


# In[21]:


from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 


# In[22]:


# Selecting 4 clusters from the above scree plot which is the optimum number of clusters 
model=KMeans(n_clusters=4) 
model.fit(norm_CR)

model.labels_ # getting the labels of clusters assigned to each row 


# In[23]:


kd=pd.Series(model.labels_)  # converting numpy array into pandas series object 
crime_rate['clust']=kd # creating a  new column and assigning it to new column 
norm_CR.head()


# In[24]:


crime_rate.iloc[:,1:7].groupby(crime_rate.clust).mean()


# In[26]:


crime_rate


# In[31]:


model1 = KMeans(n_clusters=4).fit(norm_CR)

norm_CR.plot(x="Rape",y = "Murder",c=model1.labels_,kind="scatter",s=10,cmap=plt.cm.coolwarm)


# In[32]:


#DBSCAN
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# In[35]:


dbscan = DBSCAN(eps=0.8, min_samples=4)
dbscan.fit(norm_CR)


# In[36]:


#Noisy samples are given the label -1.
dbscan.labels_


# In[ ]:




