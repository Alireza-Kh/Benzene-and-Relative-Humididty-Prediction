# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 16:33:54 2020

Air quality time-series data prediction

@author: Alireza Kheradmand

"""

# Import packages

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

#-------------------------------Functions-------------------------------------

# Function for feature visualization
def visualization(Num_Data):# Only for dataframe
    
    for col in Num_Data.columns:        
        plt.figure()
        plt.plot(Num_Data[col],label=col)
        plt.xlabel('Time Index')
        plt.ylabel(col)
        plt.legend()

#Function for visualizing normalized training data
def Graph(X_train,features):
    
    fig, axs = plt.subplots(5, 2, sharex=True, sharey=True)
    k = 0
    for i in range(5):
       for j in range(2):
           
           axs[i, j].plot(X_train[:,k])
           axs[i,j].set(xlabel='Time', ylabel=features[k])
           plt.show()
           k = k + 1
           
# Function for outlier detection and removal
def Outlier(X):
  from scipy.stats import zscore
  Z_scores = np.abs(zscore(X))

  for i in range(len(X)):
    
      for j in range(0,len(X.columns)):
        
          if Z_scores[i,j] > 3:
            
              X.iloc[i,j] =  0.5*( X.iloc[i-1,j] + X.iloc[i+1,j] )
            
  return(X) 
#------------------------------Main Script------------------------------------
# Reading data

Data = pd.read_excel('AirQualityUCI.xlsx') # Reading data as a dataframe
Date_Time = Data.iloc[:,0:2] # Separating time indexes of the data
Num_Data = Data.iloc[:,2:] # Numeric reading of data from sensors

'''
visualization(Num_Data) # Visualizing data 
'''

'''
Based on visualization of non-metalic hydrocabon concentration, the values
of this sensor are missed most of the times. So this feature will be removed.
'''
Num_Data = Num_Data.drop(labels='NMHC(GT)',axis=1)

# Missing data imputation: 
# an interpolate function in Pandas library is used.The method for 
# interpolation is linear estimation.

# Missing points are recorded as -200. For simplicity, -200 are replaced with nan.
Num_Data = Num_Data.replace(-200,np.nan)

# Linear interpolation of missing values
Num_Data = Num_Data.interpolate(method='linear',axis=0)

# Outlier detection and removal
Num_Data = Outlier(Num_Data)

Correlation_matrix = Num_Data.corr()
#Use heatmap to see corelation between variables
sns.heatmap(Correlation_matrix,annot=True,cmap='viridis')
plt.title('Heatmap of correlation between variables',fontsize=16)
plt.show()

# Choose C6H6 and RH as outputs:

X = Num_Data.drop(labels=['C6H6(GT)','RH'],axis =1)
Y = Num_Data[['C6H6(GT)','RH']]

#Splitting the data into training and testing

from sklearn.model_selection import TimeSeriesSplit
splitter = TimeSeriesSplit(n_splits=2)

for train_index, test_index in splitter.split(X):
     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
     
     Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

# Standardizing the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
Y = Y.values
Y_train = Y_train.values
Y_test = Y_test.values

# Visualizing normalized data to check if there is similar patterns
Graph(X_train,X.columns)
Graph(X_test,X.columns)

# Linear Regression modeling
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
lin_mdl = LinearRegression().fit(X_train, Y_train)
Y_train_lin_pred  = lin_mdl.predict(X_train)
Y_test_lin_pred  = lin_mdl.predict(X_test)
R2_lin = np.asarray([[r2_score(Y_train[:,0],Y_train_lin_pred[:,0]),r2_score(Y_train[:,1],Y_train_lin_pred[:,1])],
          [r2_score(Y_test[:,0],Y_test_lin_pred[:,0]), r2_score(Y_test[:,1],Y_test_lin_pred[:,1])]])

# C6H6(GT) real value and prediction (training, testing) comparision
plt.figure()
plt.plot(Y[:,0],label='True Value',color='r')
plt.plot(train_index,Y_train_lin_pred[:,0],label='Training Prediction',color='b')
plt.plot(test_index,Y_test_lin_pred[:,0],label='Testing Prediction',color='y')
plt.xlabel('Time')
plt.ylabel('C6H6(GT) Level')
plt.legend()

plt.figure()
plt.scatter(Y_train[:,0], Y_train_lin_pred[:,0],label='Training')
plt.scatter(Y_test[:,0], Y_test_lin_pred[:,0],label='Testing')
plt.xlabel('Real Value')
plt.ylabel('Prediction')
plt.title('C6H6(GT) Prediction')
plt.legend()
plt.text(35,15, f"Training R2 = {str(np.round(R2_lin[0,0], decimals =4))}", fontsize=10)
plt.text(35,5, f"Testing R2 = {str(np.round(R2_lin[0,1], decimals =4))}", fontsize =10)




# Relative humidity (RH) real value and prediction (training, testing) comparision
plt.figure()
plt.plot(Y[:,1],label='True Value',color='r')
plt.plot(train_index,Y_train_lin_pred[:,1],label='Training Prediction',color='b')
plt.plot(test_index,Y_test_lin_pred[:,1],label='Testing Prediction',color='y')
plt.xlabel('Time')
plt.ylabel('Relative Humididty')
plt.legend()