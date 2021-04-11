#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from cluster_func import kmeans
from cluster_func import em


# In[26]:


#CENSUS DATA
bank_df  = pd.read_csv("data/census.csv", delimiter=',')
bank_df = bank_df.drop(['native-country'], axis=1)
cat_vars = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex']
bank_df_dummies = pd.get_dummies(bank_df, columns=cat_vars)
bank_df_dummies['sallary'] = bank_df_dummies['sallary'].map({'>50K':0, '<=50K': 1})

labels = bank_df_dummies[['sallary']]
features = bank_df_dummies.drop(['sallary'], axis=1)

X = features.iloc[:,:-1]
y = labels.iloc[:,-1]

dataset = "census"


# In[27]:


#CANCER DATA
# from sklearn.datasets import load_breast_cancer
# data = load_breast_cancer()

# X = data.data
# X = np.nan_to_num(X)
# y = data.target
# dataset = "cancer"


# In[28]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# In[29]:


#Preprocessing the data between 0 and 1
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[30]:


means_init = np.array([X[y == i].mean(axis=0) for i in range(2)])


# In[31]:


##############################################################################################################################
#For Expected Maximization
em(dataset,X_train, X_test, y_train, y_test, init_means = means_init, component_list = [3,4,5,6,7,8,9,10,11,12,13,14,15], num_class = 2)


# In[32]:


#############################################################################################################################
#For KMeans
kmeans(dataset,X_train, X_test, y_train, y_test, init_means = means_init, component_list = [3,4,5,6,7,8,9,10,11,12,13,14,15], num_class = 2)


# In[ ]:





# In[ ]:




