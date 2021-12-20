#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')
import csv

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        


# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


#Reading training file
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
#Selecting numerical features
train_numerical = train.select_dtypes(include='int64')
y = train_numerical['SalePrice']
X = train_numerical.iloc[:,:-1]
list(X.columns)


# In[ ]:


#Reading test file
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
#Selecting numerical features
test_numerical = test[X.columns]
#Filling NA cells with mean values
test_numerical = test_numerical.fillna(test_numerical.mean())
test_numerical.shape


# In[ ]:


#Modelling decision tree classifier
model = linear_model.LinearRegression()
model = model.fit(X, y)
train_predictions = model.predict(test_numerical)
#test_predictions = model.predict(X_2)


# In[ ]:


print(train_predictions)


# In[ ]:


#Writing prediction values to CSV file
output_array = np.array(train_predictions)
np.savetxt("sales_price_pred_linear.csv", output_array, delimiter=",")

output_array = np.array(test_numerical)
np.savetxt("Id.csv", output_array, delimiter=",")

