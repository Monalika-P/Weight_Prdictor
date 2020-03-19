# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 13:44:34 2020
@author: MONALIKA P
"""
#importing the libaries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading the data set
dataset = pd.read_csv("weight-height.csv")

 # Label Encoding-1 for male and 0 for female
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 2].values
from sklearn.preprocessing import LabelEncoder
labelEncoder_gender =  LabelEncoder()
X[:,0] = labelEncoder_gender.fit_transform(X[:,0])

#for coverting into float datatype
X = X[:, :].astype(np.float)

# Spliting data into training test and test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Fitting the regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

#Making prediction using the test data
lin_pred = lin_reg.predict(X_test)

#Model Accuracy
from sklearn import metrics
print('R square = ',metrics.r2_score(y_test, lin_pred))
print('Mean squared Error = ',metrics.mean_squared_error(y_test, lin_pred))
print('Mean absolute Error = ',metrics.mean_absolute_error(y_test, lin_pred))

# Weight prediction 
my_weight_pred = lin_reg.predict([[0,190.54]])
print('My predicted weight = ',my_weight_pred)
