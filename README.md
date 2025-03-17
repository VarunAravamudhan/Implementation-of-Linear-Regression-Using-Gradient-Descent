# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Varun A
RegisterNumber:212224240178
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.01,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(X).dot(theta).reshape(-1,1)
        errors=(predictions - y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv('50_Startups.csv',header=None)
print(data.head())

X=(data.iloc[1:, :-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_scaled=scaler.fit_transform(X1)
Y1_scaled=scaler.fit_transform(y)
print(X1_scaled)
print(Y1_scaled)
theta=linear_regression(X1_scaled,Y1_scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"Predicted value: {pre}")


*/
```

## Output:
![scree_page-0001](https://github.com/user-attachments/assets/e6d99b8c-5eb2-498f-a1bb-04b0deb4eed6)

![scree_page-0002](https://github.com/user-attachments/assets/270a2f71-00e4-45f8-8711-fc44614553bd)

![scree_page-0003](https://github.com/user-attachments/assets/818d0c08-166a-4651-b276-59572797cd86)!
[scree_page-0004](https://github.com/user-attachments/assets/eddc8241-7306-4441-b798-1e0dec6e6823)

![scree_page-0005](https://github.com/user-attachments/assets/7b60139c-0656-44d1-bfb3-1c6d0f0bb50b)

![scree_page-0006](https://github.com/user-attachments/assets/385d963f-8015-4e3e-b681-c9b8e4e5f919)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
