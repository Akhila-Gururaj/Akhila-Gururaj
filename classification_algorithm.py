# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 18:55:30 2021

@author: Akhila C V
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier



data1=pd.read_csv("Heart Failure Dataset.csv")
data1.columns = ['age', 'anaemia', 'creatinine_ph', 'diabetes','ejection_fraction', 'high_bp', 'platelets','serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time','DEATH_EVENT']
data1.isnull().sum()
data1.isnull().mean()
sns.boxplot(data=data1)

data1.boxplot(grid='false', color='blue',fontsize=10, rot=30)

plt.boxplot(data1.creatinine_ph)
data1.info

data1.shape

X = data1.iloc[:,:-1]
y = data1.DEATH_EVENT

data1.describe()
data1.DEATH_EVENT.describe()
data1.DEATH_EVENT.unique()
data1.DEATH_EVENT = data1.DEATH_EVENT.replace(0,"alive").replace(1,"dead")

X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=.3,random_state=123)


#knn model


for i in range(1, 299):
    model = KNeighborsClassifier(i)
    model.fit(X_train,y_train)
    model.score(X_test,y_test)
    result=result.append({"i":K,"score_test": knmodel.score(X_test,y_test),"score_train":knmodel.score(X_train,y_train)},ignore_index=True)
    plt.plot(result.k,result.score_test)
    plt.plot(result.k,result.score_train)
    
pred = model.predict(X_test)
print(confusion_matrix(y_test,pred))


X_train.shape

confusion_matrix(y_test,pred)
actual = y_test.copy()
pd.crosstab(pred,actual)

# Decision Tree
dtmodel = DecisionTreeClassifier()
dtmodel.fit(X_train,y_train)
dtmodel.score(X_test,y_test) #testing accuracy
dtmodel.score(X_train,y_train) #training accuracy
pred = dtmodel.predict(X_test)

confusion_matrix(y_test,pred)

#displaying the DT
tree.plot_tree(dtmodel)

#random forest

rfmodel = RandomForestClassifier()
rfmodel.fit(X_train,y_train)
rfmodel.score(X_test,y_test) #testing accuracy
rfmodel.score(X_train,y_train) #training accuracy

from sklearn.ensemble import GradientBoostingClassifier as GB

gbmodel = GB()
gbmodel.fit(X_train,y_train)
gbmodel.score(X_test,y_test) #testing accuracy
gbmodel.score(X_train,y_train) #training accuracy

for number in range(5):
    print("Thank you")
