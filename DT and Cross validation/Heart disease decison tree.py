#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing the models
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix 


# In[3]:


# reading the dataset from the url
df1 = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data',header=None)


# In[4]:


df1.head()


# In[5]:


df1.columns


# In[7]:


#renaming the columns
df1_cols = ['age','sex','cp','restbp','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','hd']


# In[8]:


df1.columns = df1_cols


# In[9]:


df1.head()


# In[10]:


#finding missing data
# find out the data type for each column
df1.dtypes


# In[11]:


#so i have to find the unique values in those colums
df1['ca'].unique()


# In[12]:


df1['thal'].unique()


# In[13]:


# so in the nodes capes and breast quad they was mssing data "?".
# so im going to check how many roles contain missing values
len(df1.loc[(df1['ca'] == '?')
            |
            (df1['thal'] == '?')])


# In[14]:


#since we have only 9 rows missing. I am going to print out the rows with the missing values
df1.loc[(df1['ca'] == '?')
            |
            (df1['thal'] == '?')]


# In[15]:


# count the number of row is dataset
len(df1)


# In[17]:


#Since the number of rows with missing data is small compared to the number of rows is small compared to the decison tree im 
# going to remove the rows with the missing data
df1_no_missing = df1.loc[(df1['ca'] != '?')
                        &
                        (df1['thal'] != '?')]


# In[18]:


#length of the db when the datasets are removed
len(df1_no_missing)


# In[19]:


#confirm if those rows have no missing valuues again
df1_no_missing['ca'].unique()


# In[20]:


df1_no_missing['thal'].unique()


# In[21]:


#Format the data for classification tree using one hot encoding
#Before that we have to split the data into and X and Y. X being the value we are going to use to predict Y.
x = df1_no_missing.drop('hd',axis=1).copy()
x.head()


# In[23]:


# copy of Y which is the data we want to predict. Which is the hd column
y = df1_no_missing['hd'].copy()
y.head()


# In[24]:


# now we are going to format x so its suitbale for make predicting Y using one hot encoding.
x.dtypes


# In[26]:


x['cp'].unique()


# In[28]:


#so we are going to convert most classes into categorical starting with breast quad
x_encoded = pd.get_dummies(x, columns=['cp','restecg','slope','thal'])

x_encoded.head()


# In[29]:


y.unique()


# In[30]:


#making a simple classification tree
y_not_zero_index = y > 0
y[y_not_zero_index] = 1
y.unique()


# In[33]:


#split the data into training and testing sets
x_train, x_test, y_train,y_test = train_test_split(x_encoded, y, random_state=42)


# In[38]:


# creating decison tree and plotting it
clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt = clf_dt.fit(x_train, y_train)

plt.figure(figsize=(15,7.5))
plot_tree(clf_dt,
          filled=True,
          rounded=True,
          class_names=["No HD", "Yes HD"],
          feature_names=x_encoded.columns);


# In[39]:


#plotting confusion matrix to see how the decison tree performs
plot_confusion_matrix(clf_dt, x_test, y_test, display_labels=["Does not have HD", "Has HD"])


# In[56]:


#I have to prune the value to make the model as accurate as it can be
path = clf_dt.cost_complexity_pruning_path(x_train, y_train)
ccp_alphas = path.ccp_alphas
ccp_alphas = ccp_alphas[:-1]

clf_dts = []

for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=0,ccp_alpha=ccp_alpha)
    clf_dts.append(clf_dt)


# In[62]:


#using cross validation to get the most optimal graph
alpha_loop_values = []

#For each candidate value for alpha, we will run 5-fold cross validaton.
#After that we will store the mean and SD of scores (the accuracy) for each call
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    scores = cross_val_score(clf_dt, x_train, y_train, cv=5)
    alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])
    


# In[63]:


#Drawing the graph of the mean and STD
alpha_results = pd.DataFrame(alpha_loop_values,
                             columns=['alpha', 'mean_accuracy', 'std'])

alpha_results.plot(x='alpha',
                   y='mean_accuracy',
                   yerr='std',
                   marker='o',
                   linestyle='--')


# In[65]:


alpha_results[(alpha_results['alpha'] > 0.014)
              &
              (alpha_results['alpha'] < 0.015)]


# In[69]:


ideal_ccp_alpha = alpha_results[(alpha_results['alpha'] > 0.014)
                               &
                                (alpha_results['alpha'] < 0.015)]['alpha']
ideal_ccp_alpha


# In[71]:


#Since python thinks alpha is a series bevause we got 2 values when we printed it out. we need o convert it to a float
ideal_ccp_alpha = float(ideal_ccp_alpha)
ideal_ccp_alpha


# In[74]:


#so since we have the value of alpha we need to build a new decision tree using the optimal value for alpha we got above
clf_dt_pruned = DecisionTreeClassifier(random_state=42,
                                      ccp_alpha=ideal_ccp_alpha)
clf_dt_pruned = clf_dt_pruned.fit(x_train, y_train)


# In[75]:


#Drawing another confusion matrix to show the difference
plot_confusion_matrix(clf_dt_pruned,
                      x_test,
                      y_test,
                      display_labels=["Does not have HD", "Has HD"])


# In[79]:


#Plotting final decison tree
plt.figure(figsize=(15, 7.5))
plot_tree(clf_dt_pruned,
          filled=True,
          rounded=True,
          class_names=["No HD", "Yes HD"],
          feature_names=x_encoded.columns);

