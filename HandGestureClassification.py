#!/usr/bin/env python
# coding: utf-8

# # Loading Data/ Libraries

# In[1]:


# Loading the necessary libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore") 


# In[2]:


# Data Preprocessing
### Prosthetic hand EMG sensor files
r_data = pd.read_csv('https://raw.githubusercontent.com/cmrad/handGesturesClassification/master/0.csv',sep = ",", header = None) #rock gesture signals
s_data = pd.read_csv('https://raw.githubusercontent.com/cmrad/handGesturesClassification/master/1.csv', sep = ",", header = None) #scissors gesture signals
p_data = pd.read_csv('https://raw.githubusercontent.com/cmrad/handGesturesClassification/master/2.csv', sep = ",", header = None) #paper gesture signals
ok_data = pd.read_csv('https://raw.githubusercontent.com/cmrad/handGesturesClassification/master/3.csv',sep = ",", header = None) #ok gesture signals


# # General Data Information

# In[3]:


print(r_data.head())
print("Rock Shape: ",r_data.shape,
      "\nScissor Shape: ",s_data.shape,
      "\nPaper Shape: ",p_data.shape,
      "\nOK Shape: ",ok_data.shape)


# # Data Visualization

# In[4]:


def plot_sensor(data,name,color):
    color_list=["navy","darkmagenta","red","black"]
    fig, ax = plt.subplots(2,4, figsize=(20,12))
    sns.set(style="white")
    sns.set(style="whitegrid")
    x=0
    for i in range(2):
        for j in range(4):
            plt.suptitle(name)
            #r_data.iloc[:,i].plot.hist(bins=10,ax=ax[i][j],grid=True)
            sns.distplot(data.iloc[:,x],kde=False,ax=ax[i][j],color=color_list[color],bins=15);
            x+=1
            if i==1:
                ax[i][j].set_title("Sensor_"+str(j+5))
            else:
                ax[i][j].set_title("Sensor_"+str(j+1))
    plt.show()


# In[5]:


plot_sensor(r_data,"Rock_Data",0)


# In[6]:


plot_sensor(s_data,"Scissor_Data",1)


# In[7]:


plot_sensor(p_data,"Paper_Data",2)


# In[8]:


plot_sensor(ok_data,"OK_Data",3)


# In[9]:


#Time Series
colors=["forestgreen","teal","crimson","chocolate","darkred","lightseagreen","orangered","chartreuse"]
time_rock = r_data.iloc[:,0:8]
time_rock.index=pd.to_datetime(time_rock.index)
time_rock.iloc[:170,:].plot(subplots=True,figsize=(10,10),colors=colors);


# In[10]:


time_scis=s_data.iloc[:,0:8]
time_scis.index=pd.to_datetime(time_scis.index)
time_scis.iloc[:170,:].plot(subplots=True,figsize=(10,10),colors=colors);


# # Data Concatenation
# 

# In[11]:


completeData=pd.concat([r_data,s_data,p_data,ok_data],ignore_index=True)
df=completeData.copy()
# column names into string 
l=[str(x) for x in range(65)]
df.columns=l
df.head()


# # Building Models 

# ### Naive Bayes

# In[12]:


from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
X=df.drop(["64"],axis=1)
y=df["64"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

naive=GaussianNB().fit(X_train,y_train)
naive


# In[ ]:


y_pred=naive.predict(X_test)
print("Accuracy:",accuracy_score(y_test,y_pred))


# In[ ]:


y_test.head()


# In[ ]:


y_pred[0:5]


# # KNN 

# In[14]:


from sklearn.neighbors import KNeighborsClassifier
kneigh=KNeighborsClassifier()
k_model=kneigh.fit(X_train,y_train)
k_model


# In[15]:


y_pred=k_model.predict(X_test)
print("Accuracy:",accuracy_score(y_test,y_pred))


# # KNN Model Tuning 

# In[ ]:


params={"n_neighbors": np.arange(1,10)}
knn=KNeighborsClassifier()
knn_cv=GridSearchCV(knn,params,cv=10)
knn_cv.fit(X_train,y_train)


# In[ ]:


knn_cv.best_params_


# In[ ]:


knn_model=KNeighborsClassifier(n_neighbors=9)
knn_tuned=knn_model.fit(X_train,y_train)
knn_tuned


# In[ ]:


y_pred=knn_tuned.predict(X_test)
print("Accuracy:",accuracy_score(y_test,y_pred))


# # Artificial Neural Networks

# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)

from sklearn.neural_network import MLPClassifier
mlp=MLPClassifier().fit(X_train_scaled,y_train)
mlp


# In[ ]:


y_pred=mlp.predict(X_test_scaled)
accuracy_score(y_test,y_pred)


# # CatBoost

# In[ ]:


from catboost import CatBoostClassifier

cat_model=CatBoostClassifier(silent=True).fit(X_train,y_train)

y_pred=cat_model.predict(X_test)
accuracy_score(y_test,y_pred)

