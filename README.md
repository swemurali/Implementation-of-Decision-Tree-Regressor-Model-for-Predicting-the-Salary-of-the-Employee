# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values present in the dataset and apply label encoder.
3. Determine test and training data set and apply decison tree regression in dataset.
4. calculate Mean square error,data prediction and r2.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: M.Suwetha
RegisterNumber:  212221230112
*/
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
x.head()
y=data[["Salary"]]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
```

## Output:
## dataset
![c1](https://user-images.githubusercontent.com/94165336/203975099-cb2e749b-fdc4-4970-9993-a2bc039d7ff5.png)
## null values
![c2](https://user-images.githubusercontent.com/94165336/203975111-4663c47c-4ef8-4549-8fb1-8e3502648405.png)
![c3](https://user-images.githubusercontent.com/94165336/203975336-6c4a29f7-b9c9-426e-b6da-3e23be2f012f.png)
## applying label encoder
![c4](https://user-images.githubusercontent.com/94165336/203975360-a87a364b-0d2d-4b86-ae9f-ce0692d5096a.png)
## X
![c5](https://user-images.githubusercontent.com/94165336/203975371-1161ea32-fee0-4171-b8c0-379aed755953.png)
## Mean square error
![c6](https://user-images.githubusercontent.com/94165336/203975386-f7e73291-8f01-4669-8cb0-f9218fe94ba6.png)
## R2
![c7](https://user-images.githubusercontent.com/94165336/203975422-43cebd4b-0990-4151-ba56-2b3db35abd94.png)
## data prediction
![c8](https://user-images.githubusercontent.com/94165336/203975439-74ba733d-adfb-4f2b-8c95-4d30351f92ad.png)



## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
