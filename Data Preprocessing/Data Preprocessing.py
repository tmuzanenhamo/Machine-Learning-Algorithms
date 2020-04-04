# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 13:02:53 2020

@author: tmuza
"""

# Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# import dataset
dataset = pd.read_csv('Data.csv')  # read the data set
x = dataset.iloc[:, :-1].values  # independent variables
y = dataset.iloc[:, 3].values  # dpended variables

# deal with missing data
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(x[:, 1:3])  # fitting the data with missing values from index 1-2
x[:, 1:3] = imputer.transform(x[:, 1:3])

# encode categorical data
labelencoder_x = LabelEncoder()
x[:, 0] = labelencoder_x.fit_transform(x[:, 0])  # assining the countries to encoded values.

onehotencoder = OneHotEncoder(categorical_features=[0])  # specify which column you want to encode
x = onehotencoder.fit_transform(x).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)  # encode y values

# Splitting the dataset into the train set and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Feature scalling

scale_x = StandardScaler()
x_train = scale_x.fit_transform(x_train)
x_test = scale_x.transform(x_test)
