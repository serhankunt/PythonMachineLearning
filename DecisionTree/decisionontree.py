# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 12:08:13 2024

@author: hserh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

veriler = pd.read_csv('maaslar.csv')

#data frame dilimleme(slice)
x = veriler.iloc[:,1:2]
y= veriler.iloc[:,2:]

#numpy array dönüşümü
X = x.values
Y = y.values

#Linear Regression
#Doğrusal Model oluşturma
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

plt.scatter(X, Y, color='red')
plt.plot(x,lin_reg.predict(X), color = 'blue')
plt.show()

#polynomial regression
#Doğrusal olmayan (nonlinear) model oluşturma
#2.dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)
print(x_poly)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,y)

plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.show()

#4.dereceden polinom
from sklearn.preprocessing import PolynomialFeatures
poly_reg3 = PolynomialFeatures(degree = 4)
x_poly3 = poly_reg3.fit_transform(X)
print(x_poly3)
lin_reg3 = LinearRegression()
lin_reg3.fit(x_poly3,y)

plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg3.predict(poly_reg3.fit_transform(X)), color = 'blue')
plt.show()

#tahminler

print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))
print(lin_reg2.predict(poly_reg.fit_transform([[11]])))

from sklearn.preprocessing import StandardScaler

sc1= StandardScaler()
x_olcekli = sc1.fit_transform(X)

sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y).ravel()

from sklearn.svm import SVR

svr_reg = SVR(kernel = 'rbf')
svr_reg.fit(x_olcekli,y_olcekli)

plt.scatter(x_olcekli,y_olcekli,color='red')
plt.plot(x_olcekli,svr_reg.predict(x_olcekli),color='blue')
plt.show()

print(svr_reg.predict(sc1.transform([[14]])))
print(svr_reg.predict(sc1.transform([[11]])))
print(svr_reg.predict(sc1.transform([[10.6]])))
print(svr_reg.predict(sc1.transform([[9.6]])))
print(svr_reg.predict(sc1.transform([[8.6]])))
print(svr_reg.predict(sc1.transform([[7.6]])))
print(svr_reg.predict(sc1.transform([[6.6]])))
print(svr_reg.predict(sc1.transform([[5.6]])))
print(svr_reg.predict(sc1.transform([[4.6]])))
print(svr_reg.predict(sc1.transform([[3.6]])))
print(svr_reg.predict(sc1.transform([[2.6]])))
print(svr_reg.predict(sc1.transform([[1.6]])))

from sklearn.tree import DecisionTreeRegressor
r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

Z = X + 0.5
K = X + 0.5

plt.scatter(X,Y,color='red')
plt.plot(X,r_dt.predict(X),color='blue')

print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))
print(r_dt.predict([[15]]))
print(r_dt.predict([[5.3]]))

plt.plot(r_dt.predict(Z),color='green')
plt.plot(r_dt.predict(K),color='yellow')

plt.show()

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=10,random_state=0)
rf_reg.fit(X,Y.ravel())

print(rf_reg.predict([[6.5]]))

plt.scatter(X, Y,color='red')
plt.plot(X,rf_reg.predict(X),color='blue')

plt.plot(X,rf_reg.predict(Z),color='green')
plt.plot(x,r_dt.predict(K),color='yellow')




