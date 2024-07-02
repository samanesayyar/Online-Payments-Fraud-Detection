# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 17:39:46 2024

@author: Samane
"""

import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import plotly.io as pio
pio.templates.default = "plotly_white"
pio.renderers.default='browser'
data=pd.read_csv("archive/PS_20174392719_1491204439457_log.csv")
#print(data.head())
#print(data.isnull().sum())
##Exploring transaction type
#print(data.type.value_counts())
type=data["type"].value_counts()
transaction=type.index
quantity=type.values
##Distribution of Transaction Type
# figure=px.pie(data_frame=data,values=quantity, 
#              names=transaction ,hole = 0.5, 
#              title="Distribution of Transaction Type")
# figure.show()

# Checking correlation
# correlation=data.corr(numeric_only=True)
# print(correlation.isFraud.sort_values(ascending=False))

#convert categorical to numerical
data.type= data.type.map({"CASH_OUT":1,"PAYMENT": 2, 
                                 "CASH_IN": 3, "TRANSFER": 4,
                                 "DEBIT": 5})
data["isFraud"] = data["isFraud"].map({0: "No Fraud", 1: "Fraud"})
#print(data.head())
x=np.array(data[["type", "amount", "oldbalanceOrg", "newbalanceOrig"]])
y=np.array(data[["isFraud"]])
xtrain, xtest, ytrain, ytest= train_test_split(x,y, test_size=0.2, random_state=42)
model=DecisionTreeClassifier()
model.fit(xtrain, ytrain)
print(model.score(xtest, ytest))

# prediction
#features = [type, amount, oldbalanceOrg, newbalanceOrig]
features = np.array([[4, 9000.60, 9000.60, 0.0]])
print(model.predict(features))

