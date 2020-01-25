import pandas as pd
import numpy as np
from numpy import reshape
from sklearn import datasets, linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

#Data preprocessing and analysis
data_car =pd.read_excel('ACCIDENTAL DATA FILE.xlsx')
data_car.insert(3,'label','0')
data_car.drop(["MS"], axis = 1, inplace = True) 

data_phone=pd.read_excel('PHONE FALL DATA FILE.xls')
data_phone.insert(4,'label','1')
data_phone.drop(["ACTUAL TIME (ms)", "TIME INTERVAL (ms)"], axis = 1, inplace = True) 

data_phone.rename(columns={'Accel X':'X','Accel Y':'Y'},inplace=True)
data=data_phone.append(data_car,ignore_index=True)

y=data['label']
X=data.drop('label',axis=1)

#train and test set
X_train,X_test,y_train,y_test=train_test_split(X, y,test_size=0.3,random_state=42)

#fitting a model
clf=RandomForestClassifier(n_estimators=100,bootstrap=True)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)

print("accuracy:",metrics.accuracy_score(y_test, y_pred))


X1=input("x axis acceleration " )
y1=input("y axis acceleration " )
y_predict=clf.predict([[X1,y1]])
if y_predict=='1':
    print("Phone has fallen")
else:
    print("Car accident")





