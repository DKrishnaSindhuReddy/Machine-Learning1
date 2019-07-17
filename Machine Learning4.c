import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets import load_breast_cancer #TO LOAD BREAST_CANCER DATASET FROM EXISTING ONE'S. 
dataset=load_breast_cancer() #BREAST_CANCER TABLE DATA IS ASSIGNED TO DATASET.
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
from numpy import *
df['MED']=dataset.target
x=df[['mean texture']] #ASSIGNING MEAN_TEXTURE DATA TO "X" VARIABLE
y=df['MED'] #ASSIGNING "MED" DATA TO "Y" VARIABLE
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y) #USED TO FIT THE MODEL
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y) #USED TO FIT THE MODEL
model2=LinearRegression()
model2.fit(x_poly,y)
y_poly_pred=model2.predict(x_poly)
polynomial_feature1=PolynomialFeatures(degree=3)
x_poly1=polynomial_feature1.fit_transform(x)
print(x_poly1)
model3=LinearRegression()
model3.fit(x_poly1,y)
y_poly_pred1=model3.predict(x_poly1)
plt.plot(x,y_poly_pred1,'bo') #POLYNOMIAL CURVE FOR GIVEN BREAST_CANCER DATA
print(model3.coef_) #GIVES COEFFICIENT VALUES
print(model3.intercept_) #GIVES INTERCEPT VALUE
print(model.score(x,y))
print(model2.score(x_poly,y))
print(model3.score(x_poly1,y)) #GIVES "r^2 VALUE THAT FINDS ACCURACY "