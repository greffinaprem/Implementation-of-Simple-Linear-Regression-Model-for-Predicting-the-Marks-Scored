# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: GREFFINA SANCHEZ P
RegisterNumber: 212222040048



import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt

dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())

#assigning hours to X & scores to Y
X = dataset.iloc[:,:-1].values
print(X)
Y = dataset.iloc[:,-1].values
print(Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=1/3,random_state=0)
print(X_train)
print(X_test)
print(Y_train)
print(Y_test)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
Y_pred =reg.predict(X_test)
print(Y_pred)
print(Y_test)

#Graph plot for training data
plt.scatter(X_train,Y_train,color='blue')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

#Graph plot for test data
plt.scatter(X_test,Y_test,color='red')
plt.plot(X_train,reg.predict(X_train),color='purple')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_absolute_error(Y_test,Y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
 
*/
```

## Output:

**1. df.head()**
 
![image](https://github.com/greffinaprem/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475603/ff61db94-e7dd-417e-9cbe-41451e349371)

**2. df.tail()**
 
![image](https://github.com/greffinaprem/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475603/25c96d3f-a82a-4aa0-b445-9da17ea603cb)

**3. Array value of X**
 
![image](https://github.com/greffinaprem/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475603/a19d5d36-26d7-4b96-94fa-f72f9d6286e8)

**4. Array value of Y**
 
![image](https://github.com/greffinaprem/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475603/c86fd129-74fc-463a-a72d-ec6d3ecdccfb)

**5. Values of Y prediction**

![image](https://github.com/greffinaprem/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475603/63ce9f37-0f52-495f-9f24-3788c7695e63)

**6. Array values of Y test**
 
![image](https://github.com/greffinaprem/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475603/8e3cd84a-e132-4599-9c8e-dc5c62d7ebb7)

**7. Training Set Graph**
 
![image](https://github.com/greffinaprem/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475603/3a40e720-b989-4fd1-9886-461fb042b086)

**8. Test Set Graph**
 
![image](https://github.com/greffinaprem/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475603/e4d91f64-bf05-478b-aa3b-6f54ef4d9b4d)


**9. Values of MSE, MAE and RMSE**
 
![image](https://github.com/greffinaprem/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119475603/90bd3555-2069-4f86-b51e-7367d4587c26)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
