# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 16:32:32 2021

@author: Akhila C V
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

covid=pd.read_csv('covid.csv')


covid.head()
covid.describe()
covid.shape   #looking at the shape of the data
covid.columns #looking the header

#handling the null values
covid.isna().any() 
covid.isnull().sum()
covid["location"].value_counts() # Checking the number of locations in the data base
covid.location.unique()
covid.location.value_counts()
covid.total_cases.fillna(covid.total_cases.mean(),inplace=True)

india_case=covid[covid["location"]=="India"] 

#Total cases per day
sns.set(rc={'figure.figsize':(15,10)})
sns.lineplot(x="date",y="total_cases",data=india_case)
plt.show()

#Making a dataframe for last 5 days
india_last_5days_cases=india_case.tail()

sns.set(rc={'figure.figsize':(15,10)})
sns.lineplot(x="date",y="total_cases",data=india_last_5days_cases)
plt.show()

#converting string date to date-time

from dateutil.parser import parse
dateparse=lambda dates:parse(date)

import datetime as dt
india_case['date'] = pd.to_datetime(india_case['date']) 
india_case.head()


#converting date-time to ordinal
india_case['date']=india_case['date'].map(dt.datetime.toordinal)
india_case.head()


X=india_case['date']
Y=india_case['total_cases']



x=india_case['date']
y=india_case['total_cases']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3) #splittingthe data

lr = LinearRegression()
lr.fit(np.array(x_train).reshape(-1,1),np.array(y_train).reshape(-1,1))
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
india_case.tail()



from sklearn.metrics import mean_squared_error
mean_squared_error(x_test,y_pred) # 3056.1136363636365

y_pred_test = lr.predict(np.array(x_test).reshape(-1,1))
y_pred

y_pred_train= lr.predict(np.array(x_train).reshape(-1,1))



lr.score(np.array(x_train).reshape(-1,1),np.array(y_train).reshape(-1,1)) #-544.943269280076
lr.score(np.array(x_test).reshape(-1,1),np.array(y_test).reshape(-1,1)) #-653.4017505239746


from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(2)
X_train_transform = poly.fit_transform(np.array(x_train).reshape(-1,1))

poly_model = LinearRegression()
poly_model.fit(X_train_transform,np.array(y_train).reshape(-1,1))

# get the predictions
# convert our test x into polynomial x - 
X_test_transform = poly.fit_transform(np.array(x_test).reshape(-1,1))

poly_model.score(X_test_transform,y_test) #0.8259893564280216

# to get predcition for the graph line
X_transform = poly.fit_transform(x)
pred = poly_model.predict(X_transform)

pred_p_train = poly_model.predict(X_train_transform)
pred_p_test = poly_model.predict(X_test_transform)


pred_l_train = model.predict(x_train)
pred_l_test = model.predict(x_test)

#graph
plt.scatter(x_train,y_train)
plt.scatter(x_test,y_test)



from sklearn.metrics import mean_squared_error
mean_squared_error(y_train, y_pred_train, squared = False) #linear regression - 22577.803753532637

from sklearn.metrics import mean_squared_error
mean_squared_error(y_train, pred_p_train, squared = False) #polynomial regression - 11324.077579255256


,111,1)